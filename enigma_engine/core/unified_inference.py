"""
Unified Inference Layer
=======================

This module provides a UNIFIED interface for all model types (internal and external).
All models behave the same way - the AI can control tools regardless of model source.

Architecture:
┌────────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED INFERENCE LAYER                                 │
│                                                                                 │
│  User Input ──┬──────────────────────────────────────────────────────────────► │
│               │                                                                 │
│               ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │    INTENT DETECTION (runs for ALL inputs)                                │  │
│  │    - Native function calling (if model supports it)                      │  │
│  │    - Keyword-based detection (UniversalToolRouter)                       │  │
│  │    - Output parsing (<tool_call> tags)                                   │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│               │                                                                 │
│               ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │    MODEL INFERENCE (picks best method for model type)                    │  │
│  │    - GGUF: llama.cpp chat_with_tools() or chat()                        │  │
│  │    - HuggingFace: transformers generate() with chat template            │  │
│  │    - Enigma: EnigmaEngine.chat()                                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│               │                                                                 │
│               ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │    TOOL EXECUTION (if tools detected)                                    │  │
│  │    - Execute tools & show results to AI                                  │  │
│  │    - Agent loop: AI sees results, can call more tools                   │  │
│  │    - Repeat until AI is done or max iterations                          │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│               │                                                                 │
│               ▼                                                                 │
│  Final Response ◄─────────────────────────────────────────────────────────────  │
└────────────────────────────────────────────────────────────────────────────────┘

Usage:
    from enigma_engine.core.unified_inference import UnifiedEngine
    
    # Wrap any model
    engine = UnifiedEngine(my_model, model_type="gguf")  # or "huggingface", "enigma"
    
    # Generate with automatic tool handling
    response = engine.generate_with_tools(
        "Draw me a picture of a sunset",
        agent_loop=True,  # AI can iterate with tool results
        max_iterations=5
    )
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL DEFINITIONS - Available tools for AI control
# =============================================================================

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a text description",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Description of image to generate"},
                    "style": {"type": "string", "description": "Art style (realistic, anime, cartoon, oil_painting)"},
                    "width": {"type": "integer", "description": "Image width in pixels"},
                    "height": {"type": "integer", "description": "Image height in pixels"}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_code",
            "description": "Generate code in any programming language",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Description of code to generate"},
                    "language": {"type": "string", "description": "Programming language"}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_video",
            "description": "Generate a video from text description",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Description of video to generate"},
                    "duration": {"type": "number", "description": "Duration in seconds"}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_audio",
            "description": "Convert text to speech or generate audio",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to speak or audio description"},
                    "voice": {"type": "string", "description": "Voice to use"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "screenshot",
            "description": "Take a screenshot of the screen",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Screen region (full, active, left_half, etc.)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "switch_tab",
            "description": "Switch to a different GUI tab",
            "parameters": {
                "type": "object",
                "properties": {
                    "tab_name": {"type": "string", "description": "Tab name (chat, image, code, video, audio, settings)"}
                },
                "required": ["tab_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "control_avatar",
            "description": "Control the avatar (move, speak, emote)",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Action (wave, nod, smile, speak, move)"},
                    "text": {"type": "string", "description": "Text to speak (for speak action)"}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a system command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current date and time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


class UnifiedEngine:
    """
    Unified inference engine that wraps ANY model type.
    
    Provides consistent tool-calling behavior regardless of whether you're using:
    - An internal Enigma model
    - A GGUF model via llama.cpp
    - A HuggingFace model
    - An API model (OpenAI, Anthropic, etc.)
    
    The AI has the same level of control with ALL model types.
    """
    
    def __init__(
        self,
        model: Any,
        model_type: str = "auto",
        tools: Optional[List[Dict]] = None,
        enable_agent_loop: bool = True,
        keyword_routing: bool = True
    ):
        """
        Initialize unified engine.
        
        Args:
            model: The underlying model (EnigmaEngine, GGUF model, HF model, etc.)
            model_type: "enigma", "gguf", "huggingface", "api", or "auto" to detect
            tools: Custom tool definitions (uses defaults if None)
            enable_agent_loop: Enable multi-step tool execution
            keyword_routing: Enable keyword-based intent detection as fallback
        """
        self.model = model
        self.model_type = self._detect_model_type(model) if model_type == "auto" else model_type
        self.tools = tools or AVAILABLE_TOOLS
        self.enable_agent_loop = enable_agent_loop
        self.keyword_routing = keyword_routing
        
        # Tool executor
        self._tool_executor = None
        
        # Universal router for keyword detection
        self._universal_router = None
        
        logger.info(f"UnifiedEngine initialized with model_type={self.model_type}")
    
    def _detect_model_type(self, model: Any) -> str:
        """Auto-detect model type."""
        # Check for GGUF (llama.cpp)
        if hasattr(model, '_llama') or hasattr(model, 'create_chat_completion'):
            return "gguf"
        
        # Check for HuggingFace
        if hasattr(model, 'generate') and hasattr(model, 'config'):
            return "huggingface"
        
        # Check for EnigmaEngine
        if hasattr(model, 'chat') and hasattr(model, 'generate'):
            return "enigma"
        
        # Default
        return "unknown"
    
    def _get_tool_executor(self):
        """Lazy load tool executor."""
        if self._tool_executor is None:
            try:
                from ..tools.tool_executor import ToolExecutor
                self._tool_executor = ToolExecutor()
            except ImportError as e:
                logger.warning(f"Could not load ToolExecutor: {e}")
        return self._tool_executor
    
    def _get_universal_router(self):
        """Lazy load universal router."""
        if self._universal_router is None:
            try:
                from .universal_router import UniversalToolRouter
                self._universal_router = UniversalToolRouter()
            except ImportError as e:
                logger.warning(f"Could not load UniversalToolRouter: {e}")
        return self._universal_router
    
    # =========================================================================
    # MAIN API
    # =========================================================================
    
    def generate_with_tools(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        agent_loop: bool = True,
        max_iterations: int = 5,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response with automatic tool handling.
        
        This is the MAIN method that makes all models behave the same.
        
        Args:
            user_message: User's input text
            history: Conversation history
            system_prompt: System prompt
            agent_loop: Enable multi-step tool execution
            max_iterations: Max tool→result→AI cycles
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            {
                "response": str,           # Final text response
                "tool_calls": List[Dict],  # All tool calls made
                "tool_results": List[Dict],# Results from tool execution
                "iterations": int,         # Number of agent loop iterations
            }
        """
        all_tool_calls = []
        all_tool_results = []
        iteration = 0
        current_message = user_message
        current_history = history or []
        response = ""  # Initialize response
        
        while iteration < max_iterations:
            # Step 1: Generate response from model
            response, tool_calls = self._generate_and_extract_tools(
                current_message,
                current_history,
                system_prompt,
                max_tokens,
                temperature,
                **kwargs
            )
            
            all_tool_calls.extend(tool_calls)
            
            # Step 2: If no tool calls, we're done
            if not tool_calls:
                # Try keyword-based routing as fallback (only on first iteration)
                if iteration == 0 and self.keyword_routing:
                    keyword_tools = self._detect_tools_from_keywords(user_message)
                    if keyword_tools:
                        tool_calls = keyword_tools
                        all_tool_calls.extend(tool_calls)
                        logger.info(f"Detected {len(keyword_tools)} tool(s) from keywords")
                
                if not tool_calls:
                    break
            
            # Step 3: Execute tools
            tool_results = self._execute_tools(tool_calls)
            all_tool_results.extend(tool_results)
            
            # Step 4: If not in agent loop mode, stop
            if not agent_loop or not self.enable_agent_loop:
                break
            
            # Step 5: Feed results back to AI
            iteration += 1
            if iteration < max_iterations:
                # Update history with AI's response and tool results
                current_history = current_history + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": self._format_tool_results(tool_results)}
                ]
                current_message = "Based on the tool results above, continue or provide your final response."
        
        return {
            "response": response,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "iterations": iteration
        }
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Simple chat without tool handling."""
        return self._model_chat(message, history, system_prompt, **kwargs)
    
    # =========================================================================
    # MODEL-SPECIFIC INFERENCE
    # =========================================================================
    
    def _generate_and_extract_tools(
        self,
        message: str,
        history: List[Dict[str, str]],
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Tuple[str, List[Dict]]:
        """Generate response and extract any tool calls."""
        
        if self.model_type == "gguf":
            return self._generate_gguf(message, history, system_prompt, max_tokens, temperature)
        elif self.model_type == "huggingface":
            return self._generate_hf(message, history, system_prompt, max_tokens, temperature)
        elif self.model_type == "enigma":
            return self._generate_enigma(message, history, system_prompt, max_tokens, temperature)
        else:
            # Unknown model - try basic generation
            response = self._model_chat(message, history, system_prompt, max_tokens=max_tokens)
            tool_calls = self._parse_tool_calls_from_response(response)
            return response, tool_calls
    
    def _generate_gguf(
        self,
        message: str,
        history: List[Dict[str, str]],
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> Tuple[str, List[Dict]]:
        """Generate with GGUF model (llama.cpp)."""
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": message})
        
        # Try native function calling first
        if hasattr(self.model, 'chat_with_tools'):
            result = self.model.chat_with_tools(
                messages=messages,
                tools=self.tools,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            response = result.get('content', '')
            native_calls = result.get('tool_calls', [])
            
            # Convert native format to our format
            tool_calls = []
            for tc in native_calls:
                func = tc.get('function', {})
                tool_calls.append({
                    "tool": func.get('name', ''),
                    "params": json.loads(func.get('arguments', '{}')) if isinstance(func.get('arguments'), str) else func.get('arguments', {})
                })
            
            return response, tool_calls
        
        # Fallback to regular chat + parse
        elif hasattr(self.model, 'chat'):
            response = self.model.chat(messages=messages, max_tokens=max_tokens, temperature=temperature)
            if not isinstance(response, str):
                response = str(response)
            tool_calls = self._parse_tool_calls_from_response(response)
            return response, tool_calls
        
        return "", []
    
    def _generate_hf(
        self,
        message: str,
        history: List[Dict[str, str]],
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> Tuple[str, List[Dict]]:
        """Generate with HuggingFace model."""
        
        # HuggingFace models typically use chat() method
        if hasattr(self.model, 'chat'):
            response = self.model.chat(
                message,
                history=history,
                system_prompt=system_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            if not isinstance(response, str):
                response = str(response)
            tool_calls = self._parse_tool_calls_from_response(response)
            return response, tool_calls
        
        return "", []
    
    def _generate_enigma(
        self,
        message: str,
        history: List[Dict[str, str]],
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> Tuple[str, List[Dict]]:
        """Generate with internal Enigma model."""
        
        if hasattr(self.model, 'chat'):
            response = self.model.chat(
                message=message,
                history=history,
                system_prompt=system_prompt,
                max_gen=max_tokens,
                temperature=temperature
            )
            tool_calls = self._parse_tool_calls_from_response(response)
            return response, tool_calls
        
        return "", []
    
    def _model_chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str],
        **kwargs
    ) -> str:
        """Call model's chat method directly."""
        if hasattr(self.model, 'chat'):
            return self.model.chat(
                message=message if self.model_type == "enigma" else message,
                history=history,
                system_prompt=system_prompt,
                **kwargs
            )
        return ""
    
    # =========================================================================
    # TOOL PARSING & EXECUTION
    # =========================================================================
    
    def _parse_tool_calls_from_response(self, response: str) -> List[Dict]:
        """Parse <tool_call> tags from response."""
        tool_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(tool_pattern, response, re.DOTALL)
        
        tool_calls = []
        for match in matches:
            try:
                data = json.loads(match)
                tool_calls.append({
                    "tool": data.get('tool', ''),
                    "params": data.get('params', {})
                })
            except json.JSONDecodeError:
                pass
        
        return tool_calls
    
    def _detect_tools_from_keywords(self, user_input: str) -> List[Dict]:
        """Detect tools from keywords in user input."""
        router = self._get_universal_router()
        if not router:
            return []
        
        intent = router.detect_intent(user_input)
        
        # Map intents to tool calls
        intent_to_tool = {
            "image": {"tool": "generate_image", "params": {"prompt": user_input}},
            "video": {"tool": "generate_video", "params": {"prompt": user_input}},
            "audio": {"tool": "generate_audio", "params": {"text": user_input}},
            "3d": {"tool": "generate_3d", "params": {"prompt": user_input}},
            "code": {"tool": "generate_code", "params": {"prompt": user_input}},
            "web_search": {"tool": "web_search", "params": {"query": user_input}},
            "screenshot": {"tool": "screenshot", "params": {}},
            "get_time": {"tool": "get_time", "params": {}},
        }
        
        if intent in intent_to_tool:
            return [intent_to_tool[intent]]
        
        return []
    
    def _execute_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls and return results."""
        executor = self._get_tool_executor()
        results = []
        
        for tc in tool_calls:
            tool_name = tc.get('tool', '')
            params = tc.get('params', {})
            
            try:
                if executor:
                    result = executor.execute_tool(tool_name, params)
                else:
                    result = {"success": False, "error": "No tool executor available"}
            except Exception as e:
                result = {"success": False, "error": str(e)}
            
            result['tool'] = tool_name
            results.append(result)
        
        return results
    
    def _format_tool_results(self, results: List[Dict]) -> str:
        """Format tool results for AI to understand."""
        lines = ["Tool execution results:"]
        for r in results:
            tool = r.get('tool', 'unknown')
            if r.get('success'):
                result_text = str(r.get('result', r.get('path', 'Success')))[:500]
                lines.append(f"- {tool}: SUCCESS - {result_text}")
            else:
                error = r.get('error', 'Unknown error')
                lines.append(f"- {tool}: FAILED - {error}")
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def wrap_model(model: Any, **kwargs) -> UnifiedEngine:
    """
    Convenience function to wrap any model in UnifiedEngine.
    
    Example:
        from enigma_engine.core.unified_inference import wrap_model
        
        engine = wrap_model(my_gguf_model)
        response = engine.generate_with_tools("Draw a cat")
    """
    return UnifiedEngine(model, **kwargs)


def generate_with_tools(
    model: Any,
    message: str,
    **kwargs
) -> Dict[str, Any]:
    """
    One-shot tool-enabled generation with any model.
    
    Example:
        result = generate_with_tools(my_model, "Search for Python tutorials")
        print(result['response'])
        print(result['tool_results'])
    """
    engine = UnifiedEngine(model)
    return engine.generate_with_tools(message, **kwargs)
