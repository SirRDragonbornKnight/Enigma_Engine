"""
Universal Tool Router
=====================

Provides keyword-based tool routing that works with ANY model:
- Enigma AI Engine trained models
- Enigma AI Engine untrained models  
- HuggingFace models
- Any external model/API

This routes based on USER INPUT keywords, not model output.
The model doesn't need to know how to call tools - this handles it.

Usage:
    from enigma_engine.core.universal_router import UniversalToolRouter
    
    router = UniversalToolRouter()
    
    # Route and execute based on user message
    result = router.route_and_execute("draw me a cat")
    # Returns: {"tool": "image", "success": True, "result": "..."}
    
    # Just detect intent without executing
    intent = router.detect_intent("search for python tutorials")
    # Returns: "web_search"
    
    # Use with any model's chat method
    response = router.chat_with_routing(
        user_message="draw a sunset",
        chat_function=my_model.chat,  # Any model's chat function
        fallback_to_chat=True
    )
"""

import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class UniversalToolRouter:
    """
    Universal tool router that works with ANY model.
    
    Routes based on keywords in user input, not model output.
    This means even a completely untrained model can use tools.
    """
    
    # Tool detection patterns: (tool_name, keywords, priority)
    # Higher priority = checked first
    TOOL_PATTERNS = [
        # Image generation
        ("image", ["draw", "paint", "sketch", "illustrate", "picture of", "image of", 
                   "create image", "generate image", "make image", "artwork of",
                   "photo of", "photograph of"], 10),
        
        # Video generation
        ("video", ["video of", "animate", "create video", "generate video", 
                   "make video", "clip of", "animation of"], 10),
        
        # Audio/Speech
        ("audio", ["speak", "say out loud", "read aloud", "text to speech",
                   "generate audio", "create sound", "make music", "tts"], 10),
        
        # 3D generation
        ("3d", ["3d model", "create 3d", "generate 3d", "make 3d", 
                "mesh of", "sculpt"], 10),
        
        # GIF generation  
        ("gif", ["create gif", "make gif", "generate gif", "animated gif"], 10),
        
        # Code generation - expanded patterns
        ("code", ["write code", "create code", "generate code", "program",
                  "write a function", "write a script", "code for", 
                  "implement", "write python", "write javascript",
                  "code to", "write some code", "coding", "function to",
                  "algorithm", "sort a list", "parse", "create a class"], 8),
        
        # Web search
        ("web_search", ["search for", "search the web", "google", "look up",
                        "find information", "search online", "web search"], 8),
        
        # Time/Date utilities
        ("get_time", ["what time", "current time", "time is it", "what's the time",
                      "tell me the time", "date today", "what date", "current date"], 9),
        
        # File operations
        ("read_file", ["read file", "open file", "show file", "cat file",
                       "what's in the file", "read the"], 7),
        ("write_file", ["write to file", "save to file", "create file",
                        "write file"], 7),
        ("list_directory", ["list files", "list directory", "show files",
                            "what files", "ls", "dir"], 7),
        
        # Screenshot/Vision
        ("screenshot", ["screenshot", "capture screen", "take screenshot",
                        "screen capture", "what's on my screen"], 9),
        ("vision", ["look at", "analyze image", "describe image", 
                    "what do you see", "identify"], 8),
        
        # System info
        ("system_info", ["system info", "cpu usage", "memory usage", "disk space",
                         "hardware info", "system status"], 7),
        
        # Default chat (lowest priority)
        ("chat", ["explain", "tell me", "what is", "how do", "why", 
                  "help me", "can you"], 1),
    ]
    
    def __init__(self):
        """Initialize the universal router."""
        self._tool_executor = None
        self._tool_router = None
        
    def _get_tool_executor(self):
        """Lazy load tool executor."""
        if self._tool_executor is None:
            try:
                from ..tools.tool_executor import ToolExecutor
                self._tool_executor = ToolExecutor()
            except ImportError as e:
                logger.warning(f"Could not load ToolExecutor: {e}")
        return self._tool_executor
    
    def _get_tool_router(self):
        """Lazy load the main tool router."""
        if self._tool_router is None:
            try:
                from .tool_router import get_router
                self._tool_router = get_router()
            except ImportError as e:
                logger.warning(f"Could not load ToolRouter: {e}")
        return self._tool_router
    
    def detect_intent(self, user_input: str) -> str:
        """
        Detect which tool the user wants based on keywords.
        
        Args:
            user_input: The user's message
            
        Returns:
            Tool name (e.g., "image", "code", "chat")
        """
        input_lower = user_input.lower()
        
        # Check patterns in priority order
        matches = []
        for tool_name, keywords, priority in self.TOOL_PATTERNS:
            for keyword in keywords:
                if keyword in input_lower:
                    matches.append((tool_name, priority, keyword))
                    break  # One match per tool is enough
        
        if not matches:
            return "chat"
        
        # Return highest priority match
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[0][0]
    
    def extract_params(self, user_input: str, tool_name: str) -> dict[str, Any]:
        """
        Extract parameters for a tool from user input.
        
        Args:
            user_input: The user's message
            tool_name: The detected tool
            
        Returns:
            Dictionary of parameters for the tool
        """
        params = {"prompt": user_input}
        
        if tool_name == "image":
            # Extract image description
            patterns = [
                r'(?:draw|paint|create|generate|make)\s+(?:me\s+)?(?:a\s+)?(?:picture|image|photo)?\s*(?:of\s+)?(.+)',
                r'(?:picture|image|photo)\s+of\s+(.+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    params["prompt"] = match.group(1).strip()
                    break
            params["width"] = 512
            params["height"] = 512
            
        elif tool_name == "web_search":
            # Extract search query
            patterns = [
                r'(?:search|google|look up|find)\s+(?:for\s+)?(.+)',
                r'what is\s+(.+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    params["query"] = match.group(1).strip()
                    break
            if "query" not in params:
                params["query"] = user_input
                
        elif tool_name == "read_file":
            # Extract file path
            match = re.search(r'(?:read|open|show|cat)\s+(?:file\s+)?["\']?([^\s"\']+)["\']?', 
                            user_input, re.IGNORECASE)
            if match:
                params["path"] = match.group(1)
                
        elif tool_name == "write_file":
            # Extract path and content
            match = re.search(r'(?:write|save)\s+["\']?(.+?)["\']?\s+to\s+["\']?([^\s"\']+)["\']?',
                            user_input, re.IGNORECASE)
            if match:
                params["content"] = match.group(1)
                params["path"] = match.group(2)
                
        elif tool_name == "list_directory":
            # Extract directory path
            match = re.search(r'(?:list|show|ls|dir)\s+(?:files\s+)?(?:in\s+)?["\']?([^\s"\']+)?["\']?',
                            user_input, re.IGNORECASE)
            if match and match.group(1):
                params["path"] = match.group(1)
            else:
                params["path"] = "."
                
        elif tool_name in ("audio", "speak"):
            # Extract text to speak
            patterns = [
                r'(?:speak|say|read)\s+["\'](.+)["\']',
                r'(?:speak|say|read)\s+(.+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    params["text"] = match.group(1).strip()
                    break
                    
        return params
    
    def route_and_execute(self, user_input: str) -> dict[str, Any]:
        """
        Detect intent, extract params, and execute the appropriate tool.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dictionary with:
                - tool: Name of tool used
                - success: Whether execution succeeded
                - result: Tool output or error message
                - routed: Whether a tool was used (False = went to chat)
        """
        # Detect which tool to use
        tool_name = self.detect_intent(user_input)
        
        logger.info(f"Universal router detected: {tool_name}")
        
        # Chat doesn't need tool execution
        if tool_name == "chat":
            return {
                "tool": "chat",
                "success": True,
                "result": None,
                "routed": False,
                "message": "No specific tool detected, use chat model"
            }
        
        # Extract parameters
        params = self.extract_params(user_input, tool_name)
        
        # Map tool names to executor tool names
        tool_mapping = {
            "image": "generate_image",
            "video": "generate_video",
            "audio": "speak_text",
            "3d": "generate_3d",
            "gif": "generate_gif",
            "code": "generate_code",
            "web_search": "web_search",
            "read_file": "read_file",
            "write_file": "write_file",
            "list_directory": "list_directory",
            "screenshot": "screenshot",
            "vision": "analyze_image",
        }
        
        executor_tool = tool_mapping.get(tool_name, tool_name)
        
        # Execute the tool
        executor = self._get_tool_executor()
        if not executor:
            return {
                "tool": tool_name,
                "success": False,
                "result": "Tool executor not available",
                "routed": True
            }
        
        try:
            result = executor.execute(executor_tool, params)
            success = result.get("success", False) or "error" not in result
            
            return {
                "tool": tool_name,
                "success": success,
                "result": result,
                "routed": True,
                "params": params
            }
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "tool": tool_name,
                "success": False,
                "result": str(e),
                "routed": True,
                "params": params
            }
    
    def chat_with_routing(
        self,
        user_message: str,
        chat_function: Callable[[str], str],
        fallback_to_chat: bool = True,
        **chat_kwargs
    ) -> str:
        """
        Route user message to tools or chat, works with ANY model.
        
        This is the main method to use for adding tool support to any model.
        
        Args:
            user_message: The user's input
            chat_function: Any function that takes a prompt and returns a response
                          (e.g., model.chat, model.generate, engine.generate)
            fallback_to_chat: If tool fails, try chat instead
            **chat_kwargs: Additional kwargs for the chat function
            
        Returns:
            Response string (from tool or chat)
            
        Example:
            # With HuggingFace model
            router = UniversalToolRouter()
            response = router.chat_with_routing(
                "draw me a cat",
                hf_model.chat
            )
            
            # With EnigmaEngine
            response = router.chat_with_routing(
                "search for python tutorials",
                forge_engine.generate
            )
            
            # With any callable
            response = router.chat_with_routing(
                "what is AI?",
                lambda x: "I don't know"  # Even this works
            )
        """
        # Try to route to a tool
        result = self.route_and_execute(user_message)
        
        # If no tool was matched, use chat
        if not result.get("routed"):
            logger.info("No tool matched, using chat function")
            return chat_function(user_message, **chat_kwargs)
        
        # Tool was matched
        tool_name = result.get("tool", "unknown")
        
        if result.get("success"):
            # Tool succeeded
            tool_result = result.get("result", {})
            
            # Format the response nicely
            if isinstance(tool_result, dict):
                if "path" in tool_result or "output_path" in tool_result:
                    path = tool_result.get("path") or tool_result.get("output_path")
                    return f"Done! I used the {tool_name} tool.\n\nSaved to: {path}"
                elif "content" in tool_result:
                    return f"Here's the result:\n\n{tool_result['content']}"
                elif "items" in tool_result:
                    items = tool_result["items"]
                    if isinstance(items, list):
                        item_list = "\n".join(f"  - {item.get('name', item) if isinstance(item, dict) else item}" 
                                             for item in items[:20])
                        return f"Found {len(items)} items:\n{item_list}"
                elif "results" in tool_result:
                    return f"Search results:\n{tool_result['results']}"
                else:
                    return f"Tool '{tool_name}' completed:\n{tool_result}"
            else:
                return f"Tool '{tool_name}' result: {tool_result}"
        else:
            # Tool failed
            error = result.get("result", "Unknown error")
            
            if fallback_to_chat:
                logger.info(f"Tool {tool_name} failed, falling back to chat")
                # Add context about the failed tool
                enhanced_prompt = f"{user_message}\n\n(Note: I tried to use the {tool_name} tool but it wasn't available. Please respond conversationally.)"
                return chat_function(enhanced_prompt, **chat_kwargs)
            else:
                return f"I tried to use the {tool_name} tool but encountered an error: {error}"
    
    def get_available_tools(self) -> list[str]:
        """Get list of tools this router can handle."""
        return list({t[0] for t in self.TOOL_PATTERNS if t[0] != "chat"})


# Singleton instance
_universal_router: Optional[UniversalToolRouter] = None


def get_universal_router() -> UniversalToolRouter:
    """Get the singleton universal router instance."""
    global _universal_router
    if _universal_router is None:
        _universal_router = UniversalToolRouter()
    return _universal_router


def chat_with_tools(
    user_message: str,
    chat_function: Callable[[str], str],
    **kwargs
) -> str:
    """
    Convenience function: Route message to tools or chat.
    
    Works with ANY model's chat/generate function.
    
    Args:
        user_message: User input
        chat_function: Any function(str) -> str
        **kwargs: Passed to chat_function
        
    Returns:
        Response string
        
    Example:
        from enigma_engine.core.universal_router import chat_with_tools
        
        # Works with any model
        response = chat_with_tools("draw a cat", my_model.generate)
    """
    router = get_universal_router()
    return router.chat_with_routing(user_message, chat_function, **kwargs)
