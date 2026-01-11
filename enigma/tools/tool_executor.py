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
import signal
import platform
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager

from .tool_definitions import get_tool_definition

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
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
        TimeoutError: If the operation takes longer than specified
        
    Note:
        On Windows, this provides a "soft" timeout that sets a flag.
        The operation must periodically check for cancellation.
        For truly interruptible timeouts, use run_with_timeout() instead.
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
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
                raise TimeoutError(f"Operation timed out after {seconds} seconds")
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
        TimeoutError: If the function takes longer than timeout_seconds
    """
    import concurrent.futures
    
    if kwargs is None:
        kwargs = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")


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
        except TimeoutError as e:
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
        elif tool_name == "speak":
            return self._execute_speak(module, params)
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
        
        # Handle editing tools directly
        elif tool_name == "edit_image":
            return self._execute_edit_image(params)
        elif tool_name == "edit_gif":
            return self._execute_edit_gif(params)
        elif tool_name == "edit_video":
            return self._execute_edit_video(params)
        
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
        """Execute image generation."""
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
            from enigma.modules.registry import MODULE_REGISTRY
            
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
