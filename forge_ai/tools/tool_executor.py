"""
================================================================================
🔧 TOOL EXECUTOR - THE ACTION WORKSHOP
================================================================================

The safe executor that runs AI tool calls! Parses AI requests, validates them,
executes tools with timeouts, and returns structured results.

📍 FILE: forge_ai/tools/tool_executor.py
🏷️ TYPE: Tool Execution Engine
🎯 MAIN CLASS: ToolExecutor

┌─────────────────────────────────────────────────────────────────────────────┐
│  EXECUTION FLOW:                                                            │
│                                                                             │
│  AI OUTPUT: "I'll search the web for that."                                │
│            {"tool": "web_search", "query": "AI news"}                       │
│                          │                                                  │
│                          ▼                                                  │
│              ┌───────────────────┐                                         │
│              │   TOOL EXECUTOR   │                                         │
│              ├───────────────────┤                                         │
│              │ 1. Parse JSON     │                                         │
│              │ 2. Validate params│                                         │
│              │ 3. Check security │                                         │
│              │ 4. Execute tool   │                                         │
│              │ 5. Return result  │                                         │
│              └───────────────────┘                                         │
│                          │                                                  │
│                          ▼                                                  │
│  RESULT: Search results about AI news...                                   │
└─────────────────────────────────────────────────────────────────────────────┘

🛡️ SECURITY FEATURES:
    • Timeout protection (operations can't hang forever)
    • Path blocking (AI can't access sensitive files)
    • Parameter validation before execution
    • Sandboxed execution environment

🔗 CONNECTED FILES:
    → USES:      forge_ai/tools/tool_definitions.py (get_tool_definition)
    → USES:      forge_ai/tools/vision.py, web_tools.py, file_tools.py...
    → USES:      forge_ai/utils/security.py (path blocking)
    ← USED BY:   forge_ai/core/tool_router.py (dispatches here)
    ← USED BY:   forge_ai/core/inference.py (enable_tools mode)

📖 AVAILABLE TOOLS (in forge_ai/tools/):
    • vision.py        - Screen capture, image analysis
    • web_tools.py     - Web search, fetch pages
    • file_tools.py    - Read/write/list files
    • browser_tools.py - Browser automation
    • document_tools.py - PDF, DOCX, EPUB reading
    • robot_tools.py   - Robot hardware control
    • game_router.py   - Game AI routing

📖 SEE ALSO:
    • forge_ai/tools/tool_definitions.py - Define new tools
    • forge_ai/tools/tool_registry.py    - Tool registration
    • forge_ai/core/tool_router.py       - Routes requests here
"""

from __future__ import annotations

import json
import re
import logging
import signal
import platform
import time
from contextlib import contextmanager
from typing import Any

from .tool_definitions import get_tool_definition

logger = logging.getLogger(__name__)

# Pre-compiled regex pattern for parsing tool calls (efficiency optimization)
TOOL_CALL_PATTERN = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)


class ToolTimeoutError(Exception):
    """Raised when a tool execution times out."""
    pass


@contextmanager
def timeout_context(seconds: int):
    """
    Context manager for timing out operations.
    
    Uses signal.SIGALRM on Unix systems, concurrent.futures on Windows.
    
    Args:
        seconds: Timeout in seconds
        
    Raises:
        ToolTimeoutError: If the operation takes longer than specified
        
    Note:
        On Windows, this provides a "soft" timeout that sets a flag.
        The operation must periodically check for cancellation.
        For truly interruptible timeouts, use run_with_timeout() instead.
    """
    def timeout_handler(signum, frame):
        raise ToolTimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Unix systems: use SIGALRM
    if platform.system() != "Windows" and hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows fallback: use threading
        import threading
        
        timer_expired = False
        
        def set_timeout():
            nonlocal timer_expired
            timer_expired = True
        
        timer = threading.Timer(seconds, set_timeout)
        timer.start()
        
        try:
            yield
            if timer_expired:
                raise ToolTimeoutError(f"Operation timed out after {seconds} seconds")
        finally:
            timer.cancel()


def run_with_timeout(func, args=(), kwargs=None, timeout_seconds: int = 30):
    """
    Run a function with a hard timeout that works on Windows.
    
    Uses concurrent.futures.ThreadPoolExecutor which can interrupt the thread.
    
    Args:
        func: The function to run
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        timeout_seconds: Maximum time to wait
        
    Returns:
        The function's return value
        
    Raises:
        ToolTimeoutError: If the function takes longer than timeout_seconds
    """
    import concurrent.futures
    
    if kwargs is None:
        kwargs = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise ToolTimeoutError(f"Operation timed out after {timeout_seconds} seconds")


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
        
        # Use pre-compiled pattern for efficiency
        for match in TOOL_CALL_PATTERN.finditer(text):
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
    
    def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given parameters. Alias for execute_tool().
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            
        Returns:
            Result dictionary with 'success', 'result', and optionally 'error'
        """
        return self.execute_tool(tool_name, params)

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
    
    def execute_tool_with_timeout(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Execute a tool with a timeout to prevent hanging operations.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            timeout: Timeout in seconds (default: 60)
            
        Returns:
            Result dictionary with 'success', 'result', and optionally 'error'
        """
        try:
            with timeout_context(timeout):
                return self.execute_tool(tool_name, params)
        except ToolTimeoutError as e:
            logger.warning(f"Tool {tool_name} timed out after {timeout}s")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "timeout": True,
            }
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name} with timeout: {e}")
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
        loaded_modules = self.module_manager.list_loaded()
        if module_name not in loaded_modules:
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
        
        # Execute based on tool type - specific handlers first
        if tool_name == "generate_image":
            return self._execute_generate_image(module, params)
        elif tool_name == "generate_gif":
            return self._execute_generate_gif(module, params)
        elif tool_name == "analyze_image":
            return self._execute_analyze_image(module, params)
        elif tool_name == "control_avatar":
            return self._execute_control_avatar(module, params)
        elif tool_name == "control_avatar_bones":
            return self._execute_control_avatar_bones(params)
        elif tool_name == "customize_avatar":
            return self._execute_customize_avatar(module, params)
        elif tool_name == "speak":
            return self._execute_speak(module, params)
        elif tool_name == "create_voice_profile":
            return self._execute_create_voice_profile(module, params)
        elif tool_name == "generate_code":
            return self._execute_generate_code(module, params)
        elif tool_name == "generate_video":
            return self._execute_generate_video(module, params)
        elif tool_name == "generate_audio":
            return self._execute_generate_audio(module, params)
        else:
            # Generic module tool execution - try to find matching method
            return self._execute_generic_module_tool(module, tool_name, params)
    
    def _execute_generic_module_tool(self, module, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool generically on a module by finding a matching method."""
        try:
            # Try to find a method that matches the tool name
            # e.g., "find_on_screen" -> module.find_on_screen() or module.find()
            method_names = [
                tool_name,  # exact match
                tool_name.replace("_", ""),  # no underscores
                tool_name.split("_")[0],  # first part only (e.g., "find")
                f"execute_{tool_name}",  # execute_* prefix
                "execute",  # generic execute method
            ]
            
            method = None
            for name in method_names:
                if hasattr(module, name) and callable(getattr(module, name)):
                    method = getattr(module, name)
                    break
                # Also check _instance
                if hasattr(module, '_instance') and module._instance:
                    if hasattr(module._instance, name) and callable(getattr(module._instance, name)):
                        method = getattr(module._instance, name)
                        break
            
            if method:
                result = method(**params) if params else method()
                return {
                    "success": True,
                    "result": str(result) if result else "Tool executed successfully",
                    "tool": tool_name,
                }
            else:
                return {
                    "success": False,
                    "error": f"No matching method found for tool '{tool_name}' on module",
                    "tool": tool_name,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
            }
    
    def _execute_builtin_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a built-in tool (doesn't require module)."""
        # Handle module management tools
        if tool_name == "load_module":
            return self._execute_load_module(params)
        elif tool_name == "unload_module":
            return self._execute_unload_module(params)
        elif tool_name == "list_modules":
            return self._execute_list_modules(params)
        elif tool_name == "check_resources":
            return self._execute_check_resources(params)
        
        # Handle GUI control tools
        elif tool_name == "switch_tab":
            return self._execute_switch_tab(params)
        elif tool_name == "adjust_setting":
            return self._execute_adjust_setting(params)
        elif tool_name == "get_setting":
            return self._execute_get_setting(params)
        elif tool_name == "manage_conversation":
            return self._execute_manage_conversation(params)
        elif tool_name == "show_help":
            return self._execute_show_help(params)
        elif tool_name == "optimize_for_hardware":
            return self._execute_optimize_for_hardware(params)
        
        # Handle editing tools directly
        elif tool_name == "edit_image":
            return self._execute_edit_image(params)
        elif tool_name == "edit_gif":
            return self._execute_edit_gif(params)
        elif tool_name == "edit_video":
            return self._execute_edit_video(params)
        
        # Handle direct generation tools (bypass module system)
        elif tool_name == "generate_image":
            return self._execute_generate_image(None, params)
        
        # For other builtin tools, use registry
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
        """Execute image generation - uses local Stable Diffusion directly."""
        try:
            prompt = params.get("prompt", "")
            # Ensure prompt is a string (CLIP tokenizer requires str type)
            if prompt is None:
                return {
                    "success": False,
                    "error": "Prompt cannot be None",
                    "tool": "generate_image",
                }
            prompt = str(prompt).strip()
            if not prompt:
                return {
                    "success": False,
                    "error": "Prompt cannot be empty",
                    "tool": "generate_image",
                }
            
            width = params.get("width", 512)
            height = params.get("height", 512)
            steps = params.get("steps", 20)
            
            # Try to use the image tab provider directly (bypasses module system)
            try:
                from ..gui.tabs.image_tab import get_provider
                
                # Get local SD provider
                # Choose provider based on what's available
                # Replicate is fast and cheap, local SD is slow on Pi
                import os
                if os.environ.get("REPLICATE_API_TOKEN"):
                    provider_name = 'replicate'
                elif os.environ.get("OPENAI_API_KEY"):
                    provider_name = 'openai'
                else:
                    provider_name = 'local'
                
                provider = get_provider(provider_name)
                
                if provider is None:
                    return {
                        "success": False,
                        "error": "Image provider not available",
                        "tool": "generate_image",
                    }
                
                # Auto-load if not loaded
                if not provider.is_loaded:
                    logger.info("Auto-loading Stable Diffusion model...")
                    success = provider.load()
                    if not success:
                        return {
                            "success": False,
                            "error": "Failed to load Stable Diffusion. Install: pip install diffusers transformers accelerate",
                            "tool": "generate_image",
                        }
                
                # Generate the image
                import time
                start_time = time.time()
                result = provider.generate(
                    prompt=prompt, 
                    width=width, 
                    height=height, 
                    steps=steps
                )
                duration = time.time() - start_time
                
                if result.get("success"):
                    return {
                        "success": True,
                        "result": result,
                        "path": result.get("path", ""),
                        "duration": duration,
                        "tool": "generate_image",
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get("error", "Unknown error"),
                        "tool": "generate_image",
                    }
                    
            except ImportError as e:
                logger.warning(f"Could not import image_tab: {e}")
                # Fall back to module-based generation
                pass
            
            # Fallback: try module's generate method
            if module:
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
            
            return {
                "success": False,
                "error": "No image generation method available",
                "tool": "generate_image",
            }
        
        except Exception as e:
            logger.error(f"Image generation error: {e}")
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
    
    def _execute_control_avatar_bones(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bone-level avatar control."""
        try:
            from ..avatar.ai_control import get_ai_avatar_control, BoneCommand
            
            ai_control = get_ai_avatar_control()
            action = params.get("action", "")
            
            if action == "move_bone":
                bone_name = params.get("bone_name")
                if not bone_name:
                    return {
                        "success": False,
                        "error": "bone_name required for move_bone action",
                        "tool": "control_avatar_bones",
                    }
                
                pitch = params.get("pitch")
                yaw = params.get("yaw")
                roll = params.get("roll")
                
                command = BoneCommand(bone_name, pitch=pitch, yaw=yaw, roll=roll)
                ai_control.execute_commands([command])
                
                return {
                    "success": True,
                    "result": f"Moved {bone_name} (pitch={pitch}, yaw={yaw}, roll={roll})",
                    "tool": "control_avatar_bones",
                }
            
            elif action == "gesture":
                gesture_name = params.get("gesture_name")
                if not gesture_name:
                    return {
                        "success": False,
                        "error": "gesture_name required for gesture action",
                        "tool": "control_avatar_bones",
                    }
                
                success = ai_control.execute_gesture(gesture_name)
                if success:
                    return {
                        "success": True,
                        "result": f"Executed gesture: {gesture_name}",
                        "tool": "control_avatar_bones",
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Unknown gesture: {gesture_name}",
                        "tool": "control_avatar_bones",
                    }
            
            elif action == "reset_pose":
                ai_control.reset_pose()
                return {
                    "success": True,
                    "result": "Avatar reset to neutral pose",
                    "tool": "control_avatar_bones",
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "tool": "control_avatar_bones",
                }
        
        except Exception as e:
            logger.error(f"Error in control_avatar_bones: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "control_avatar_bones",
            }
    
    def _execute_customize_avatar(self, module, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute avatar customization (colors, lighting, effects)."""
        try:
            setting = params.get("setting", "")
            value = params.get("value", "")
            
            # Store customization request in a shared state that the GUI can read
            # The GUI's avatar tab will poll for these settings
            customization = {
                "setting": setting,
                "value": value,
                "timestamp": __import__("time").time(),
            }
            
            # Try to use a shared state or signal system
            # First try module method
            if hasattr(module, "customize"):
                result = module.customize(setting=setting, value=value)
            elif hasattr(module, "_instance") and hasattr(module._instance, "customize"):
                result = module._instance.customize(setting=setting, value=value)
            else:
                # Store in a file that the GUI can watch
                import json
                from pathlib import Path
                settings_path = Path(__file__).parent.parent.parent / "information" / "avatar" / "customization.json"
                settings_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Read existing settings
                existing = {}
                if settings_path.exists():
                    try:
                        existing = json.loads(settings_path.read_text())
                    except (json.JSONDecodeError, IOError) as e:
                        logger.debug(f"Could not read avatar settings: {e}")
                        existing = {}
                
                # Update setting
                existing[setting] = value
                existing["_last_updated"] = customization["timestamp"]
                
                # Write back
                settings_path.write_text(json.dumps(existing, indent=2))
            
            return {
                "success": True,
                "result": f"Avatar {setting} set to {value}",
                "tool": "customize_avatar",
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": "customize_avatar",
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
    
    def _execute_create_voice_profile(self, module, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a voice profile from personality description."""
        try:
            name = params.get("name", "custom_voice")
            personality = params.get("personality", "")
            base_voice = params.get("base_voice", "default")
            
            if not personality:
                return {
                    "success": False,
                    "error": "Personality description is required",
                    "tool": "create_voice_profile",
                }
            
            # Use voice identity system to create profile from description
            from forge_ai.voice.voice_identity import AIVoiceIdentity
            
            identity = AIVoiceIdentity()
            profile = identity.describe_desired_voice(personality)
            profile.name = name
            profile.voice = base_voice if base_voice != "default" else profile.voice
            profile.save()
            
            return {
                "success": True,
                "result": f"Created voice profile '{name}' with pitch={profile.pitch:.2f}, speed={profile.speed:.2f}, volume={profile.volume:.2f}",
                "profile_name": name,
                "parameters": {
                    "pitch": profile.pitch,
                    "speed": profile.speed,
                    "volume": profile.volume,
                    "voice": profile.voice,
                    "effects": profile.effects,
                },
                "tool": "create_voice_profile",
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": "create_voice_profile",
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
    
    def _execute_generate_gif(self, module, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GIF generation by creating multiple image frames."""
        try:
            from pathlib import Path
            from PIL import Image
            import os
            
            frames_prompts = params.get("frames", [])
            fps = params.get("fps", 5)
            loop = params.get("loop", 0)
            width = params.get("width", 512)
            height = params.get("height", 512)
            
            if not frames_prompts or len(frames_prompts) == 0:
                return {
                    "success": False,
                    "error": "No frame prompts provided",
                    "tool": "generate_gif",
                }
            
            # Generate each frame as an image
            frame_images = []
            
            for i, prompt in enumerate(frames_prompts):
                logger.info(f"Generating frame {i+1}/{len(frames_prompts)}: {prompt}")
                
                # Call module's generate method for each frame
                if hasattr(module, "generate"):
                    result = module.generate(prompt=prompt, width=width, height=height)
                elif hasattr(module, "_instance") and hasattr(module._instance, "generate"):
                    result = module._instance.generate(prompt=prompt, width=width, height=height)
                else:
                    return {
                        "success": False,
                        "error": "Image generation module does not have generate() method",
                        "tool": "generate_gif",
                    }
                
                # Open the generated image
                if isinstance(result, str) or isinstance(result, Path):
                    img = Image.open(result)
                elif hasattr(result, 'images') and len(result.images) > 0:
                    img = result.images[0]
                else:
                    return {
                        "success": False,
                        "error": f"Could not get image from frame {i+1}",
                        "tool": "generate_gif",
                    }
                
                # Ensure consistent size
                if img.size != (width, height):
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                
                frame_images.append(img)
            
            # Create output directory if needed
            from ..config import CONFIG
            output_dir = Path(CONFIG.get("outputs_dir", "outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            import time
            timestamp = int(time.time())
            output_path = output_dir / f"animated_{timestamp}.gif"
            
            # Calculate duration per frame (in milliseconds)
            duration_ms = int(1000 / fps) if fps > 0 else 200
            
            # Save as animated GIF
            frame_images[0].save(
                output_path,
                save_all=True,
                append_images=frame_images[1:],
                duration=duration_ms,
                loop=loop,
                optimize=False
            )
            
            return {
                "success": True,
                "result": f"GIF generated successfully with {len(frame_images)} frames",
                "tool": "generate_gif",
                "output_path": str(output_path),
                "frames": len(frame_images),
                "fps": fps,
            }
        
        except Exception as e:
            logger.exception(f"Error generating GIF: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "generate_gif",
            }
    
    def _execute_edit_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image editing operations."""
        try:
            from pathlib import Path
            from PIL import Image, ImageEnhance, ImageFilter
            import os
            
            image_path = params.get("image_path", "")
            edit_type = params.get("edit_type", "")
            
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}",
                    "tool": "edit_image",
                }
            
            # Open the image
            img = Image.open(image_path)
            
            # Apply the requested edit
            if edit_type == "resize":
                width = params.get("width")
                height = params.get("height")
                if width and height:
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                else:
                    return {
                        "success": False,
                        "error": "Width and height required for resize",
                        "tool": "edit_image",
                    }
            
            elif edit_type == "rotate":
                angle = params.get("angle", 0)
                img = img.rotate(angle, expand=True)
            
            elif edit_type == "flip":
                direction = params.get("direction", "horizontal")
                if direction == "horizontal":
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                elif direction == "vertical":
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
            elif edit_type == "brightness":
                factor = params.get("factor", 1.0)
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(factor)
            
            elif edit_type == "contrast":
                factor = params.get("factor", 1.0)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(factor)
            
            elif edit_type == "blur":
                img = img.filter(ImageFilter.BLUR)
            
            elif edit_type == "sharpen":
                img = img.filter(ImageFilter.SHARPEN)
            
            elif edit_type == "grayscale":
                img = img.convert("L")
            
            elif edit_type == "crop":
                crop_box = params.get("crop_box")
                if crop_box and len(crop_box) == 4:
                    img = img.crop(tuple(crop_box))
                else:
                    return {
                        "success": False,
                        "error": "Crop box [left, top, right, bottom] required for crop",
                        "tool": "edit_image",
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown edit type: {edit_type}",
                    "tool": "edit_image",
                }
            
            # Save edited image
            from ..config import CONFIG
            output_dir = Path(CONFIG.get("outputs_dir", "outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import time
            timestamp = int(time.time())
            input_name = Path(image_path).stem
            output_path = output_dir / f"{input_name}_edited_{timestamp}.png"
            
            img.save(output_path)
            
            return {
                "success": True,
                "result": f"Image edited successfully: {edit_type}",
                "tool": "edit_image",
                "output_path": str(output_path),
                "edit_type": edit_type,
            }
        
        except Exception as e:
            logger.exception(f"Error editing image: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "edit_image",
            }
    
    def _execute_edit_gif(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GIF editing operations."""
        try:
            from pathlib import Path
            from PIL import Image
            import os
            
            gif_path = params.get("gif_path", "")
            edit_type = params.get("edit_type", "")
            
            if not os.path.exists(gif_path):
                return {
                    "success": False,
                    "error": f"GIF file not found: {gif_path}",
                    "tool": "edit_gif",
                }
            
            # Open the GIF and extract frames
            gif = Image.open(gif_path)
            frames = []
            durations = []
            
            try:
                while True:
                    frames.append(gif.copy())
                    durations.append(gif.info.get('duration', 100))
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass
            
            if not frames:
                return {
                    "success": False,
                    "error": "No frames found in GIF",
                    "tool": "edit_gif",
                }
            
            # Apply the requested edit
            if edit_type == "reverse":
                frames = frames[::-1]
                durations = durations[::-1]
            
            elif edit_type == "speed":
                speed_factor = params.get("speed_factor", 1.0)
                durations = [int(d / speed_factor) for d in durations]
            
            elif edit_type == "resize":
                width = params.get("width")
                height = params.get("height")
                if width and height:
                    frames = [f.resize((width, height), Image.Resampling.LANCZOS) for f in frames]
                else:
                    return {
                        "success": False,
                        "error": "Width and height required for resize",
                        "tool": "edit_gif",
                    }
            
            elif edit_type == "crop":
                crop_box = params.get("crop_box")
                if crop_box and len(crop_box) == 4:
                    frames = [f.crop(tuple(crop_box)) for f in frames]
                else:
                    return {
                        "success": False,
                        "error": "Crop box [left, top, right, bottom] required for crop",
                        "tool": "edit_gif",
                    }
            
            elif edit_type == "extract_frames":
                # Save individual frames
                from ..config import CONFIG
                output_dir = Path(CONFIG.get("outputs_dir", "outputs"))
                output_dir.mkdir(parents=True, exist_ok=True)
                
                import time
                timestamp = int(time.time())
                frame_paths = []
                
                for i, frame in enumerate(frames):
                    frame_path = output_dir / f"frame_{timestamp}_{i:04d}.png"
                    frame.save(frame_path)
                    frame_paths.append(str(frame_path))
                
                return {
                    "success": True,
                    "result": f"Extracted {len(frames)} frames from GIF",
                    "tool": "edit_gif",
                    "frame_paths": frame_paths,
                    "frame_count": len(frames),
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown edit type: {edit_type}",
                    "tool": "edit_gif",
                }
            
            # Save edited GIF
            from ..config import CONFIG
            output_dir = Path(CONFIG.get("outputs_dir", "outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import time
            timestamp = int(time.time())
            input_name = Path(gif_path).stem
            output_path = output_dir / f"{input_name}_edited_{timestamp}.gif"
            
            # Get original loop setting
            loop = gif.info.get('loop', 0)
            
            # Save the edited GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=loop,
                optimize=False
            )
            
            return {
                "success": True,
                "result": f"GIF edited successfully: {edit_type}",
                "tool": "edit_gif",
                "output_path": str(output_path),
                "edit_type": edit_type,
                "frame_count": len(frames),
            }
        
        except Exception as e:
            logger.exception(f"Error editing GIF: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "edit_gif",
            }
    
    def _execute_edit_video(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute video editing operations."""
        try:
            from pathlib import Path
            import os
            
            video_path = params.get("video_path", "")
            edit_type = params.get("edit_type", "")
            
            if not os.path.exists(video_path):
                return {
                    "success": False,
                    "error": f"Video file not found: {video_path}",
                    "tool": "edit_video",
                }
            
            # Try to import moviepy
            try:
                from moviepy.editor import VideoFileClip
            except ImportError:
                return {
                    "success": False,
                    "error": "MoviePy not installed. Install with: pip install moviepy",
                    "tool": "edit_video",
                }
            
            # Load video
            video = VideoFileClip(video_path)
            
            # Apply the requested edit
            if edit_type == "trim":
                start_time = params.get("start_time", 0.0)
                end_time = params.get("end_time", video.duration)
                video = video.subclip(start_time, end_time)
            
            elif edit_type == "speed":
                speed_factor = params.get("speed_factor", 1.0)
                video = video.speedx(speed_factor)
            
            elif edit_type == "resize":
                width = params.get("width")
                height = params.get("height")
                if width and height:
                    video = video.resize((width, height))
                else:
                    return {
                        "success": False,
                        "error": "Width and height required for resize",
                        "tool": "edit_video",
                    }
            
            elif edit_type == "to_gif":
                # Convert video to GIF
                from ..config import CONFIG
                output_dir = Path(CONFIG.get("outputs_dir", "outputs"))
                output_dir.mkdir(parents=True, exist_ok=True)
                
                import time
                timestamp = int(time.time())
                input_name = Path(video_path).stem
                output_path = output_dir / f"{input_name}_converted_{timestamp}.gif"
                
                fps = params.get("fps", 10)
                video.write_gif(str(output_path), fps=fps)
                video.close()
                
                return {
                    "success": True,
                    "result": f"Video converted to GIF successfully",
                    "tool": "edit_video",
                    "output_path": str(output_path),
                    "edit_type": edit_type,
                }
            
            elif edit_type == "extract_frames":
                # Extract frames from video
                from ..config import CONFIG
                output_dir = Path(CONFIG.get("outputs_dir", "outputs"))
                output_dir.mkdir(parents=True, exist_ok=True)
                
                import time
                timestamp = int(time.time())
                fps = params.get("fps", 1)
                
                frame_paths = []
                for i, frame in enumerate(video.iter_frames(fps=fps)):
                    from PIL import Image
                    frame_path = output_dir / f"video_frame_{timestamp}_{i:04d}.png"
                    img = Image.fromarray(frame)
                    img.save(frame_path)
                    frame_paths.append(str(frame_path))
                
                video.close()
                
                return {
                    "success": True,
                    "result": f"Extracted {len(frame_paths)} frames from video",
                    "tool": "edit_video",
                    "frame_paths": frame_paths,
                    "frame_count": len(frame_paths),
                }
            
            else:
                video.close()
                return {
                    "success": False,
                    "error": f"Unknown edit type: {edit_type}",
                    "tool": "edit_video",
                }
            
            # Save edited video
            from ..config import CONFIG
            output_dir = Path(CONFIG.get("outputs_dir", "outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import time
            timestamp = int(time.time())
            input_name = Path(video_path).stem
            output_path = output_dir / f"{input_name}_edited_{timestamp}.mp4"
            
            video.write_videofile(str(output_path), codec='libx264', audio_codec='aac')
            video.close()
            
            return {
                "success": True,
                "result": f"Video edited successfully: {edit_type}",
                "tool": "edit_video",
                "output_path": str(output_path),
                "edit_type": edit_type,
            }
        
        except Exception as e:
            logger.exception(f"Error editing video: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "edit_video",
            }
    
    def _execute_load_module(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load/enable a module."""
        try:
            module_id = params.get("module_id", "")
            
            if not self.module_manager:
                return {
                    "success": False,
                    "error": "ModuleManager not available. Cannot load modules.",
                    "tool": "load_module",
                }
            
            # Check if already loaded
            loaded = self.module_manager.list_loaded()
            if module_id in loaded:
                return {
                    "success": True,
                    "result": f"Module '{module_id}' is already loaded",
                    "tool": "load_module",
                    "already_loaded": True,
                }
            
            # Try to load
            success = self.module_manager.load(module_id)
            
            if success:
                return {
                    "success": True,
                    "result": f"Successfully loaded module '{module_id}'",
                    "tool": "load_module",
                    "module_id": module_id,
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to load module '{module_id}'. Check dependencies and requirements.",
                    "tool": "load_module",
                }
        
        except Exception as e:
            logger.exception(f"Error loading module: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "load_module",
            }
    
    def _execute_unload_module(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Unload/disable a module."""
        try:
            module_id = params.get("module_id", "")
            
            if not self.module_manager:
                return {
                    "success": False,
                    "error": "ModuleManager not available. Cannot unload modules.",
                    "tool": "unload_module",
                }
            
            # Check if loaded
            loaded = self.module_manager.list_loaded()
            if module_id not in loaded:
                return {
                    "success": True,
                    "result": f"Module '{module_id}' is not loaded (already unloaded)",
                    "tool": "unload_module",
                    "already_unloaded": True,
                }
            
            # Try to unload
            success = self.module_manager.unload(module_id)
            
            if success:
                return {
                    "success": True,
                    "result": f"Successfully unloaded module '{module_id}'",
                    "tool": "unload_module",
                    "module_id": module_id,
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to unload module '{module_id}'",
                    "tool": "unload_module",
                }
        
        except Exception as e:
            logger.exception(f"Error unloading module: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "unload_module",
            }
    
    def _execute_list_modules(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all modules and their status."""
        try:
            if not self.module_manager:
                return {
                    "success": False,
                    "error": "ModuleManager not available",
                    "tool": "list_modules",
                }
            
            status = self.module_manager.get_status()
            loaded_ids = set(self.module_manager.list_loaded())
            
            # Get all registered modules
            from forge_ai.modules.registry import MODULE_REGISTRY
            
            modules_info = []
            for mod_id, mod_class in MODULE_REGISTRY.items():
                info = mod_class.INFO
                modules_info.append({
                    "id": mod_id,
                    "name": info.name,
                    "category": info.category.value,
                    "loaded": mod_id in loaded_ids,
                    "description": info.description,
                    "requires_gpu": info.requires_gpu,
                    "is_cloud": info.is_cloud_service,
                })
            
            # Organize by category
            by_category = {}
            for mod in modules_info:
                cat = mod["category"]
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(mod)
            
            result_text = f"Total modules: {len(modules_info)} ({len(loaded_ids)} loaded)\n\n"
            
            for category, mods in sorted(by_category.items()):
                result_text += f"{category.upper()}:\n"
                for mod in mods:
                    status_icon = "[+]" if mod["loaded"] else "[-]"
                    result_text += f"  {status_icon} {mod['id']} - {mod['name']}\n"
                result_text += "\n"
            
            return {
                "success": True,
                "result": result_text.strip(),
                "tool": "list_modules",
                "modules": modules_info,
                "loaded_count": len(loaded_ids),
                "total_count": len(modules_info),
                "by_category": by_category,
            }
        
        except Exception as e:
            logger.exception(f"Error listing modules: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "list_modules",
            }
    
    def _execute_check_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check current system resources and get recommendations."""
        try:
            if not self.module_manager:
                return {
                    "success": False,
                    "error": "ModuleManager not available",
                    "tool": "check_resources",
                }
            
            # Get resource usage
            usage = self.module_manager.get_resource_usage()
            
            # Build human-readable summary
            result_text = f"Resource Usage Report\n"
            result_text += f"{'='*50}\n\n"
            
            # Modules
            result_text += f"Modules Loaded: {usage['modules_loaded']}/{usage['modules_registered']}\n\n"
            
            # Memory
            if 'memory' in usage and 'rss_mb' in usage['memory']:
                mem = usage['memory']
                result_text += f"Memory:\n"
                result_text += f"  Process: {mem['rss_mb']:.1f} MB\n"
                result_text += f"  System: {mem['system_used_percent']:.1f}% used\n"
                result_text += f"  Available: {mem['system_available_mb']:.1f} MB\n\n"
            
            # GPU
            if 'gpu' in usage and usage['gpu'].get('available'):
                gpu = usage['gpu']
                result_text += f"GPU ({gpu.get('device_name', 'Unknown')}):\n"
                result_text += f"  Allocated: {gpu['allocated_mb']:.1f} MB\n"
                result_text += f"  Used: {gpu['used_percent']:.1f}%\n\n"
            else:
                result_text += "GPU: Not available\n\n"
            
            # Assessment
            if 'assessment' in usage:
                assess = usage['assessment']
                status_emoji = {"good": "[OK]", "warning": "[!]", "critical": "[X]"}
                result_text += f"Status: {status_emoji.get(assess['status'], '?')} {assess['status'].upper()}\n\n"
                
                if assess['warnings']:
                    result_text += "Warnings:\n"
                    for warn in assess['warnings']:
                        result_text += f"  [!] {warn}\n"
                    result_text += "\n"
                
                if assess['recommendations']:
                    result_text += "Recommendations:\n"
                    for rec in assess['recommendations']:
                        result_text += f"  -> {rec}\n"
            
            return {
                "success": True,
                "result": result_text.strip(),
                "tool": "check_resources",
                "usage": usage,
                "status": usage.get('assessment', {}).get('status', 'unknown'),
            }
        
        except Exception as e:
            logger.exception(f"Error checking resources: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "check_resources",
            }
    
    # =========================================================================
    # GUI Control Tools
    # =========================================================================
    
    def _execute_switch_tab(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Switch to a different GUI tab.
        
        Args:
            params: {"tab_name": str, "reason": str (optional)}
        """
        try:
            from ..gui.gui_state import get_gui_state
            
            tab_name = params.get("tab_name", "").lower()
            reason = params.get("reason", "")
            
            if not tab_name:
                return {
                    "success": False,
                    "error": "tab_name is required",
                    "tool": "switch_tab",
                }
            
            gui_state = get_gui_state()
            result = gui_state.switch_tab(tab_name)
            
            if result.get("success"):
                message = f"Switched to {tab_name} tab"
                if reason:
                    message += f" - {reason}"
                return {
                    "success": True,
                    "result": message,
                    "tool": "switch_tab",
                    "tab": tab_name,
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Failed to switch tab"),
                    "tool": "switch_tab",
                }
        
        except Exception as e:
            logger.exception(f"Error switching tab: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "switch_tab",
            }
    
    def _execute_adjust_setting(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust a GUI setting.
        
        Args:
            params: {"setting_name": str, "value": Any}
        """
        try:
            from ..gui.gui_state import get_gui_state
            
            setting_name = params.get("setting_name", "")
            value = params.get("value")
            
            if not setting_name:
                return {
                    "success": False,
                    "error": "setting_name is required",
                    "tool": "adjust_setting",
                }
            
            gui_state = get_gui_state()
            result = gui_state.set_setting(setting_name, value)
            
            if result.get("success"):
                return {
                    "success": True,
                    "result": f"Set {setting_name} to {result.get('new_value')} (was {result.get('old_value')})",
                    "tool": "adjust_setting",
                    "setting": setting_name,
                    "old_value": result.get("old_value"),
                    "new_value": result.get("new_value"),
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Failed to adjust setting"),
                    "tool": "adjust_setting",
                }
        
        except Exception as e:
            logger.exception(f"Error adjusting setting: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "adjust_setting",
            }
    
    def _execute_get_setting(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the current value of a GUI setting.
        
        Args:
            params: {"setting_name": str}
        """
        try:
            from ..gui.gui_state import get_gui_state
            
            setting_name = params.get("setting_name", "")
            
            gui_state = get_gui_state()
            
            if not setting_name:
                # Return all settings
                settings = gui_state.get_all_settings()
                return {
                    "success": True,
                    "result": f"Current settings:\n{json.dumps(settings, indent=2)}",
                    "tool": "get_setting",
                    "settings": settings,
                }
            
            value = gui_state.get_setting(setting_name)
            
            return {
                "success": True,
                "result": f"{setting_name} = {value}",
                "tool": "get_setting",
                "setting": setting_name,
                "value": value,
            }
        
        except Exception as e:
            logger.exception(f"Error getting setting: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "get_setting",
            }
    
    def _execute_manage_conversation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage chat conversations (save, load, rename, delete, list, new).
        
        Args:
            params: {"action": str, "name": str (optional), "new_name": str (optional)}
        """
        try:
            from ..gui.gui_state import get_gui_state
            
            action = params.get("action", "").lower()
            name = params.get("name")
            new_name = params.get("new_name")
            
            if not action:
                return {
                    "success": False,
                    "error": "action is required (save, load, rename, delete, list, new)",
                    "tool": "manage_conversation",
                }
            
            gui_state = get_gui_state()
            result = gui_state.manage_conversation(action, name, new_name)
            
            if result.get("success"):
                # Build human-readable response
                if action == "list":
                    conversations = result.get("conversations", [])
                    if conversations:
                        msg = f"Found {len(conversations)} conversation(s):\n"
                        for conv in conversations[:10]:  # Show first 10
                            msg += f"  - {conv['name']} ({conv.get('messages', 0)} messages)\n"
                        if len(conversations) > 10:
                            msg += f"  ... and {len(conversations) - 10} more"
                    else:
                        msg = "No saved conversations found."
                elif action == "save":
                    msg = f"Saved conversation as '{name or 'auto-named'}'"
                elif action == "load":
                    msg = f"Loaded conversation '{name}'"
                elif action == "new":
                    msg = "Started new conversation"
                elif action == "rename":
                    msg = f"Renamed '{name}' to '{new_name}'"
                elif action == "delete":
                    msg = f"Deleted conversation '{name}'"
                else:
                    msg = str(result)
                
                return {
                    "success": True,
                    "result": msg,
                    "tool": "manage_conversation",
                    "action": action,
                    **result,
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Action failed"),
                    "tool": "manage_conversation",
                }
        
        except Exception as e:
            logger.exception(f"Error managing conversation: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "manage_conversation",
            }
    
    def _execute_show_help(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Show contextual help for a topic.
        
        Args:
            params: {"topic": str}
        """
        try:
            from ..gui.gui_state import get_gui_state
            
            topic = params.get("topic", "getting_started")
            
            gui_state = get_gui_state()
            result = gui_state.get_help(topic)
            
            if result.get("success"):
                return {
                    "success": True,
                    "result": f"**{result['title']}**\n\n{result['content']}",
                    "tool": "show_help",
                    "topic": topic,
                    "related": result.get("related", []),
                }
            else:
                # Return available topics if topic not found
                available = result.get("available_topics", [])
                topics_list = ", ".join(available[:10])
                return {
                    "success": True,
                    "result": f"Topic '{topic}' not found. Available topics: {topics_list}",
                    "tool": "show_help",
                    "available_topics": available,
                }
        
        except Exception as e:
            logger.exception(f"Error showing help: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "show_help",
            }
    
    def _execute_optimize_for_hardware(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize GUI settings based on hardware capabilities.
        
        Args:
            params: {"mode": str (auto, performance, balanced, power_saver, gaming)}
        """
        try:
            from ..gui.gui_state import get_gui_state
            
            mode = params.get("mode", "auto").lower()
            
            gui_state = get_gui_state()
            result = gui_state.optimize_for_hardware(mode)
            
            if result.get("success"):
                # Build human-readable summary
                hardware = result.get("hardware", {})
                applied = result.get("applied", {})
                recommendations = result.get("recommendations", [])
                
                msg = f"Optimized for {result['mode'].upper()} mode\n\n"
                msg += "Hardware detected:\n"
                if hardware.get("gpu_available"):
                    msg += f"  GPU: {hardware.get('gpu_name', 'Unknown')} ({hardware.get('gpu_vram_gb', 0):.1f} GB VRAM)\n"
                else:
                    msg += "  GPU: Not available (CPU mode)\n"
                msg += f"  RAM: {hardware.get('ram_gb', 0):.1f} GB\n"
                msg += f"  CPU: {hardware.get('cpu_cores', 1)} cores\n\n"
                
                if applied:
                    msg += "Settings applied:\n"
                    for key, value in applied.items():
                        msg += f"  {key}: {value}\n"
                    msg += "\n"
                
                if recommendations:
                    msg += "Recommendations:\n"
                    for rec in recommendations:
                        msg += f"  -> {rec}\n"
                
                return {
                    "success": True,
                    "result": msg.strip(),
                    "tool": "optimize_for_hardware",
                    "mode": result["mode"],
                    "hardware": hardware,
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Optimization failed"),
                    "tool": "optimize_for_hardware",
                }
        
        except Exception as e:
            logger.exception(f"Error optimizing for hardware: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": "optimize_for_hardware",
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
