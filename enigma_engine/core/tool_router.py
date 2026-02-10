"""
================================================================================
            CHAPTER 3: THE DISPATCHER - EVERY REQUEST FINDS ITS HOME
================================================================================

    "In the grand crossroads, all paths begin and all intentions are revealed."

Welcome to the ROUTING HUB! When a user speaks to Enigma AI Engine, this file decides
WHERE that request should go. Is it a question? Chat handles it. A drawing
request? Image generation. Code? The coding assistant.

WHY THIS FILE MATTERS:
    Enigma AI Engine has many specialized capabilities, but users just type naturally.
    "Draw me a cat" and "Explain quantum physics" need VERY different handling.
    The Tool Router reads intent and dispatches to the right specialist.

THE CROSSROADS:
    ┌─────────────────────────────────────────────────────────────────┐
    │  USER: "Draw me a cat"                                          │
    │         │                                                       │
    │         ▼                                                       │
    │  ┌─────────────────┐                                           │
    │  │  TOOL ROUTER    │  "Hmm, 'draw'... that's IMAGE territory"  │
    │  └────────┬────────┘                                           │
    │           │                                                     │
    │     ┌─────┴─────┬──────────┬──────────┬──────────┐             │
    │     ▼           ▼          ▼          ▼          ▼             │
    │  [CHAT]     [IMAGE]    [CODE]    [VIDEO]    [AUDIO]            │
    │             SELECTED                                            │
    │                ▼                                                │
    │  [Image Provider] → Cat picture appears!                       │
    └─────────────────────────────────────────────────────────────────┘

AVAILABLE ROUTES (Tools):
    | Tool   | Keywords                    | Destination              |
    |--------|-----------------------------|--------------------------| 
    | chat   | explain, help, what, why    | General conversation     |
    | image  | draw, paint, picture, art   | Image generation tab     |
    | code   | program, script, function   | Code generation tab      |
    | video  | animate, clip, movie        | Video generation tab     |
    | audio  | speak, voice, music, say    | Audio/TTS tab            |
    | 3d     | model, mesh, sculpt         | 3D generation tab        |

YOUR QUEST HERE:
    Want to add a new tool? Add routing keywords and a handler.
    Want to change which model handles a tool? Use router.assign_model().

CONNECTED PATHS:
    You came from → inference.py (uses routing for smart responses)
    Routes to    → tool_executor.py (actually runs the tool)
                 → gui/tabs/*_tab.py (generation UIs)
    Configured by → gui/tabs/model_router_tab.py (visual config)

QUICK START:
    >>> from enigma_engine.core.tool_router import get_router
    >>> router = get_router()
    >>> router.auto_route("Draw me a sunset")  # Returns image!

SEE ALSO:
    - enigma_engine/tools/tool_executor.py   - Executes the tool calls
    - enigma_engine/tools/tool_definitions.py - Define new tools
    - data/tool_routing.json            - Saved routing config
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RoutingRule:
    """
    Definition of a routing rule for intent-based tool selection.
    
    📖 WHAT THIS IS:
    A routing rule defines how to detect when a user wants to use
    a specific capability (image generation, code writing, web search, etc.)
    This is separate from ToolDefinition in tool_definitions.py which defines
    the full tool with parameters and versioning.
    
    📐 KEYWORD MATCHING:
    When user says "draw me a cat", the router looks for keywords:
    - "draw" matches image tool → route to image generator
    - "explain" matches chat tool → route to chat model
    """
    name: str                    # Tool identifier (e.g., "image")
    description: str             # What the tool does
    keywords: List[str]          # Words that trigger this tool
    parameters: Dict[str, str]   # Required parameters and descriptions
    handler: Optional[Callable] = None  # Function that executes the tool
    

@dataclass 
class ModelAssignment:
    """
    A model assigned to handle a specific tool.
    
    📖 WHAT THIS IS:
    Each tool can have multiple models assigned to it.
    The router tries them in priority order.
    
    📐 MODEL ID FORMAT:
    - "forge:name"        → Enigma AI Engine model
    - "huggingface:repo"  → HuggingFace model
    - "local:name"        → Local system tool
    - "api:service"       → External API
    
    📐 EXAMPLE:
    For the "image" tool, you might assign:
    - "local:stable-diffusion" (priority=10, fast)
    - "api:dall-e" (priority=5, high quality but slow)
    """
    model_id: str                         # e.g., "forge:name", "local:name"
    model_type: str                       # "enigma_engine", "huggingface", "local", "api"
    priority: int = 0                     # Higher = try first
    config: Dict[str, Any] = field(default_factory=dict)


# Built-in routing rules for intent classification
ROUTING_RULES = {
    "chat": RoutingRule(
        name="chat",
        description="General conversation and reasoning",
        keywords=["chat", "talk", "explain", "help", "what", "why", "how"],
        parameters={"prompt": "The user message to respond to"}
    ),
    "image": RoutingRule(
        name="image",
        description="Generate images from text descriptions",
        keywords=["draw", "paint", "create image", "generate image", "picture", "photo", "illustration", "artwork"],
        parameters={"prompt": "Description of image to generate", "width": "Image width", "height": "Image height"}
    ),
    "code": RoutingRule(
        name="code",
        description="Generate or analyze code",
        keywords=["code", "program", "script", "function", "debug", "fix code", "write code"],
        parameters={"prompt": "Code task description", "language": "Programming language"}
    ),
    # Language-specific code generation tools
    "code_python": RoutingRule(
        name="code_python",
        description="Generate Python code (specialized model)",
        keywords=["python", "py", "python script", "python code"],
        parameters={"prompt": "Python code task description"}
    ),
    "code_javascript": RoutingRule(
        name="code_javascript",
        description="Generate JavaScript/TypeScript code (specialized model)",
        keywords=["javascript", "js", "typescript", "ts", "node", "react"],
        parameters={"prompt": "JavaScript code task description"}
    ),
    "code_rust": RoutingRule(
        name="code_rust",
        description="Generate Rust code (specialized model)",
        keywords=["rust", "rs", "cargo"],
        parameters={"prompt": "Rust code task description"}
    ),
    "code_cpp": RoutingRule(
        name="code_cpp",
        description="Generate C/C++ code (specialized model)",
        keywords=["cpp", "c++", "c language", "c code"],
        parameters={"prompt": "C/C++ code task description"}
    ),
    "code_java": RoutingRule(
        name="code_java",
        description="Generate Java code (specialized model)",
        keywords=["java", "spring", "kotlin"],
        parameters={"prompt": "Java code task description"}
    ),
    "video": RoutingRule(
        name="video",
        description="Generate video clips",
        keywords=["video", "animate", "animation", "clip", "movie"],
        parameters={"prompt": "Video description", "duration": "Length in seconds"}
    ),
    "audio": RoutingRule(
        name="audio",
        description="Generate audio or speech",
        keywords=["speak", "say", "voice", "audio", "sound", "music", "song", "read aloud"],
        parameters={"text": "Text to speak or audio description"}
    ),
    "3d": RoutingRule(
        name="3d",
        description="Generate 3D models",
        keywords=["3d", "model", "mesh", "object", "sculpt"],
        parameters={"prompt": "3D model description"}
    ),
    "gif": RoutingRule(
        name="gif",
        description="Create animated GIFs",
        keywords=["gif", "animated", "animation loop", "meme"],
        parameters={"prompt": "GIF description", "frames": "Number of frames"}
    ),
    "web": RoutingRule(
        name="web",
        description="Search the web or fetch information",
        keywords=["search", "google", "look up", "find", "website", "browse"],
        parameters={"query": "Search query or URL"}
    ),
    "memory": RoutingRule(
        name="memory",
        description="Remember or recall information",
        keywords=["remember", "recall", "forget", "memory", "save this"],
        parameters={"action": "save/recall/list", "content": "What to remember"}
    ),
    "embeddings": RoutingRule(
        name="embeddings",
        description="Semantic search and similarity matching",
        keywords=["search", "find similar", "semantic", "embedding", "vector search"],
        parameters={"query": "Text to search for", "top_k": "Number of results"}
    ),
    "camera": RoutingRule(
        name="camera",
        description="Capture from webcam or camera",
        keywords=["camera", "webcam", "photo", "capture", "take picture", "snap"],
        parameters={"action": "capture/record/analyze"}
    ),
    "vision": RoutingRule(
        name="vision",
        description="Analyze images or screen captures",
        keywords=["see", "look at", "analyze image", "what's on screen", "screenshot", "describe image"],
        parameters={"image_path": "Path to image or 'screen' for screenshot"}
    ),
    "avatar": RoutingRule(
        name="avatar",
        description="Control the AI avatar display",
        keywords=["avatar", "face", "expression", "show me", "look"],
        parameters={"expression": "Avatar expression to show"}
    ),
    "robot": RoutingRule(
        name="robot",
        description="Control physical robots and servos",
        keywords=["robot", "servo", "motor", "arm", "gripper", "move robot", "animatronic"],
        parameters={"robot": "Robot name", "action": "move/grip/speak", "value": "Action value"}
    ),
    "game": RoutingRule(
        name="game",
        description="Game AI assistance and control",
        keywords=["game", "play", "gaming", "minecraft", "strategy", "quest", "level"],
        parameters={"game": "Game name", "action": "help/command/strategy"}
    ),
    "iot": RoutingRule(
        name="iot",
        description="Control smart home and IoT devices",
        keywords=["light", "lights", "turn on", "turn off", "smart home", "home assistant", "thermostat", "sensor", "switch"],
        parameters={"device": "Device name", "action": "on/off/set", "value": "Value to set"}
    ),
    "file": RoutingRule(
        name="file",
        description="Read, write, and manage files",
        keywords=["file", "read file", "write file", "save", "open file", "create file", "delete file"],
        parameters={"action": "read/write/list/delete", "path": "File path"}
    ),
    "document": RoutingRule(
        name="document",
        description="Process PDFs, Word docs, and documents",
        keywords=["pdf", "document", "word", "docx", "extract text", "read document"],
        parameters={"path": "Document path", "action": "read/extract/summarize"}
    ),
    "system": RoutingRule(
        name="system",
        description="System commands and terminal operations",
        keywords=["terminal", "command", "run", "execute", "shell", "system info"],
        parameters={"command": "Command to run"}
    ),
    "task": RoutingRule(
        name="task",
        description="Task management and reminders",
        keywords=["task", "todo", "reminder", "checklist", "schedule", "remind me"],
        parameters={"action": "add/list/complete", "content": "Task description"}
    ),
    "voice_clone": RoutingRule(
        name="voice_clone",
        description="Clone voices and create voice profiles",
        keywords=["clone voice", "voice clone", "copy voice", "voice profile", "mimic voice"],
        parameters={"action": "clone/list/use", "audio_path": "Audio sample path"}
    ),
    "automation": RoutingRule(
        name="automation",
        description="Automation, scheduling, macros, and clipboard",
        keywords=["schedule", "automate", "macro", "clipboard", "copy", "paste", "record macro", "run at"],
        parameters={"action": "schedule/macro/clipboard", "content": "What to automate"}
    ),
    "knowledge": RoutingRule(
        name="knowledge",
        description="Search Wikipedia, ArXiv, and knowledge bases",
        keywords=["wikipedia", "arxiv", "research", "paper", "article", "encyclopedia", "scientific"],
        parameters={"query": "Topic to search", "source": "wikipedia/arxiv"}
    ),
    "data": RoutingRule(
        name="data",
        description="Analyze CSV, JSON, SQL, and create charts",
        keywords=["csv", "json", "sql", "data", "chart", "graph", "plot", "analyze data", "query"],
        parameters={"action": "analyze/query/plot", "path": "Data file path"}
    ),
    "browser": RoutingRule(
        name="browser",
        description="Control browser tabs and media playback",
        keywords=["browser", "tab", "pause video", "mute", "youtube", "media", "play", "volume"],
        parameters={"action": "pause/mute/skip/focus", "target": "Tab or media target"}
    ),
    "productivity": RoutingRule(
        name="productivity",
        description="Git, Docker, SSH, and process management",
        keywords=["git", "commit", "push", "pull", "docker", "container", "ssh", "process", "kill process"],
        parameters={"action": "git/docker/ssh/process", "command": "Specific command"}
    ),
    "self": RoutingRule(
        name="self",
        description="AI self-modification: personality, preferences, memory",
        keywords=["personality", "be more", "act like", "preference", "configure yourself", "your settings"],
        parameters={"setting": "What to change", "value": "New value"}
    ),
}


class ToolRouter:
    """
    Routes requests to appropriate tools and models.
    
    📖 WHAT THIS CLASS DOES:
    The ToolRouter is the TRAFFIC CONTROLLER for all AI requests!
    It looks at what the user wants and sends the request to the right place.
    
    📐 ROUTING FLOW:
    
        User: "Draw me a sunset over mountains"
                │
                ▼
        ┌───────────────────────────────────────────────────┐
        │            TOOL ROUTER                            │
        │  1. Analyze: "draw" keyword detected              │
        │  2. Classify: This is an IMAGE request            │
        │  3. Check: Can we handle locally?                 │
        │  4. If not: Check remote peers                    │
        │  5. Execute: Run best available handler           │
        │  6. Return: Result to user                        │
        └───────────────────────────────────────────────────┘
                │
                ▼
        🎨 sunset_image.png
    
    📐 AI-TO-AI COLLABORATION:
    When enabled, the router can delegate tasks to connected AI instances:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Local (Pi)              Remote (Desktop)           Remote (Cloud)      │
    │  ┌─────────────┐        ┌─────────────────┐        ┌───────────────┐   │
    │  │ chat, code  │◄──────►│ image, audio    │◄──────►│ video, 3d     │   │
    │  │ (pi_5 model)│        │ (large + GPU)   │        │ (xl model)    │   │
    │  └─────────────┘        └─────────────────┘        └───────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘
    
    📐 SPECIALIZED MODELS:
    For even smarter routing, you can train specialized models:
    - Router Model: Classifies user intent (better than keywords)
    - Vision Model: Describes what's in images
    - Code Model: Generates code (optimized for programming)
    
    📐 MODEL CACHING:
    Loading models is slow, so we keep recently-used models in memory.
    When cache is full, oldest models are unloaded (LRU eviction).
    
    🔗 CONNECTS TO:
      → Uses tool_executor.py to run tools
      → Uses generation tabs for image/video/audio
      → Uses ai_collaboration.py for remote execution
      ← Called by inference.py when enable_tools=True
      ← Configured in GUI via model_router_tab.py
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # CACHE LIMITS: How many models to keep loaded at once
    # ─────────────────────────────────────────────────────────────────────────
    MAX_CACHE_SIZE = 5              # Main model cache
    MAX_SPECIALIZED_CACHE_SIZE = 3  # Specialized model cache
    
    def __init__(self, config_path: Optional[Path] = None, use_specialized: bool = False,
                 enable_networking: bool = False):
        """
        Initialize the Tool Router.
        
        Args:
            config_path: Where to save/load routing configuration
            use_specialized: Enable specialized models (router, vision, code)
            enable_networking: Enable AI-to-AI collaboration over network
        """
        from ..config import CONFIG
        
        self.config_path = config_path or Path(CONFIG.get("data_dir", "data")) / "tool_routing.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ─────────────────────────────────────────────────────────────────────
        # ASSIGNMENTS: Which models handle which tools
        # Tool name → List of assigned models (in priority order)
        # ─────────────────────────────────────────────────────────────────────
        self.assignments: Dict[str, List[ModelAssignment]] = {}
        
        # ─────────────────────────────────────────────────────────────────────
        # ROUTING RULES: Define how to detect intent and route to tools
        # ─────────────────────────────────────────────────────────────────────
        self.routing_rules = ROUTING_RULES.copy()
        
        # ─────────────────────────────────────────────────────────────────────
        # MODULE MANAGER INTEGRATION: Check what modules are loaded
        # ─────────────────────────────────────────────────────────────────────
        self._module_manager = None
        self._init_module_manager()
        
        # Tool -> Module mapping (which module provides which capability)
        self._tool_to_module = {
            "image": ["image_gen_local", "image_gen_api"],
            "code": ["code_gen_local", "code_gen_api"],
            "video": ["video_gen_local", "video_gen_api"],
            "audio": ["audio_gen_local", "audio_gen_api"],
            "3d": ["threed_gen_local", "threed_gen_api"],
            "embeddings": ["embedding_local", "embedding_api"],
            "vision": ["vision"],
            "camera": ["camera"],
            "voice": ["voice_input", "voice_output"],
            "avatar": ["avatar"],
        }
        
        # ─────────────────────────────────────────────────────────────────────
        # AI-TO-AI NETWORKING: For distributed task execution
        # ─────────────────────────────────────────────────────────────────────
        self._network_node: Optional[Any] = None
        self._remote_peers: Dict[str, dict] = {}  # peer_name -> {url, capabilities}
        self._collaboration_protocol: Optional[Any] = None
        self._routing_preference: str = "local_first"  # local_first, fastest, quality_first, distributed
        self._networking_enabled = enable_networking
        
        if enable_networking:
            self._init_networking()
        
        # ─────────────────────────────────────────────────────────────────────
        # MODEL CACHE: Keep loaded models in memory for speed
        # Uses LRU (Least Recently Used) eviction when cache is full
        # ─────────────────────────────────────────────────────────────────────
        self._model_cache: Dict[str, Any] = {}
        self._model_cache_order: List[str] = []  # Track access order for LRU
        
        # ─────────────────────────────────────────────────────────────────────
        # TOOL HANDLERS: Custom functions for each tool
        # ─────────────────────────────────────────────────────────────────────
        self._handlers: Dict[str, Callable] = {}
        
        # ─────────────────────────────────────────────────────────────────────
        # SPECIALIZED MODELS: Trained models for routing/vision/code
        # ─────────────────────────────────────────────────────────────────────
        self.use_specialized = use_specialized
        self._specialized_models: Dict[str, Any] = {}
        self._specialized_cache_order: List[str] = []
        self._shared_tokenizer = None  # Tokenizer shared across specialized models
        
        # Load saved configuration
        self._load_config()
        
        # Load specialized models config if enabled
        if use_specialized:
            self._load_specialized_config()
        
        # ─────────────────────────────────────────────────────────────────────
        # AUTO-ASSIGN: Models with declared capabilities auto-register
        # This means if you train ONE model for chat+code+vision,
        # it's automatically assigned to all three tools (loaded only ONCE)
        # ─────────────────────────────────────────────────────────────────────
        self._auto_assign_from_capabilities()
    
    def _auto_assign_from_capabilities(self):
        """
        Auto-assign models to tools based on their declared capabilities.
        
        📖 WHY THIS EXISTS:
        If you train ONE model on multiple things (chat, code, vision),
        you shouldn't have to manually assign it 3 times. This function
        reads each model's capabilities and auto-assigns it to matching tools.
        
        📐 HOW IT WORKS:
        1. Scan all models in registry for "capabilities" field
        2. For each model with capabilities, assign it to those tools
        3. The model is CACHED - so loading for "chat" means it's already
           loaded when "code" or "vision" is called!
        
        📐 EXAMPLE:
        Model "my_ai" has capabilities: ["chat", "code", "vision"]
        → Auto-assigns "forge:my_ai" to chat tool
        → Auto-assigns "forge:my_ai" to code tool  
        → Auto-assigns "forge:my_ai" to vision tool
        → When ANY of these tools runs, the SAME cached model is used!
        """
        try:
            from .model_registry import ModelRegistry
            registry = ModelRegistry()
            
            for model_name, info in registry.registry.get("models", {}).items():
                capabilities = info.get("capabilities", [])
                has_weights = info.get("has_weights", False)
                
                # Only auto-assign models that are trained
                if not has_weights:
                    continue
                
                # Auto-assign to each declared capability
                for cap in capabilities:
                    if cap in self.routing_rules:
                        # Check if already assigned (don't override user assignments)
                        existing = self.get_assignments(cap)
                        model_id = f"forge:{model_name}"
                        
                        if not any(a.model_id == model_id for a in existing):
                            # Add with lower priority than manual assignments
                            self.assign_model(cap, model_id, priority=5)
                            logger.debug(f"Auto-assigned {model_id} to {cap}")
            
        except Exception as e:
            logger.debug(f"Could not auto-assign from capabilities: {e}")
    
    def _init_module_manager(self):
        """
        Initialize connection to the ModuleManager.
        
        📖 WHY THIS EXISTS:
        The ModuleManager controls what modules are loaded/available.
        By connecting to it, the router can:
        1. Check if a required module (e.g., image_gen_local) is loaded
        2. Provide better error messages ("Image gen not available - enable in Modules tab")
        3. Dynamically route based on what's actually loaded
        """
        try:
            from ..modules import ModuleManager
            self._module_manager = ModuleManager.get_instance()
            logger.debug("Connected to ModuleManager for capability checking")
        except Exception as e:
            self._module_manager = None
            logger.debug(f"ModuleManager not available: {e}")
    
    def is_module_available(self, tool_name: str) -> bool:
        """
        Check if the module for a tool is loaded.
        
        Args:
            tool_name: Tool name (e.g., "image", "code", "video")
            
        Returns:
            True if at least one module providing this capability is loaded
        """
        if self._module_manager is None:
            # No module manager - assume all tools available (fallback behavior)
            return True
        
        modules = self._tool_to_module.get(tool_name, [])
        if not modules:
            # No modules defined for this tool - assume available
            return True
        
        # Check if ANY of the possible modules are loaded
        for module_id in modules:
            if self._module_manager.is_loaded(module_id):
                return True
        
        return False
    
    def get_available_modules_for_tool(self, tool_name: str) -> list[str]:
        """
        Get list of loaded modules that can handle a tool.
        
        Args:
            tool_name: Tool name (e.g., "image", "code")
            
        Returns:
            List of loaded module IDs that provide this capability
        """
        if self._module_manager is None:
            return []
        
        modules = self._tool_to_module.get(tool_name, [])
        loaded = []
        
        for module_id in modules:
            if self._module_manager.is_loaded(module_id):
                loaded.append(module_id)
        
        return loaded
    
    def get_tool_availability_message(self, tool_name: str) -> str:
        """
        Get a descriptive message about tool availability.
        
        Returns user-friendly message explaining if/why a tool isn't available.
        """
        if self.is_module_available(tool_name):
            modules = self.get_available_modules_for_tool(tool_name)
            if modules:
                return f"{tool_name.title()} available via: {', '.join(modules)}"
            return f"{tool_name.title()} available"
        
        modules = self._tool_to_module.get(tool_name, [])
        if modules:
            return f"{tool_name.title()} not available - enable one of: {', '.join(modules)} in Modules tab"
        return f"{tool_name.title()} not configured"
    
    def _cache_model(self, key: str, model: Any, is_specialized: bool = False):
        """
        Add model to cache with LRU eviction.
        
        📖 LRU (Least Recently Used) CACHING:
        When cache is full and we need to add a new model,
        we remove the OLDEST model (least recently accessed).
        
        📐 EXAMPLE:
        Cache: [A, B, C] (max=3)
        Access B → Cache: [A, C, B] (B moved to end)
        Add D → Cache: [C, B, D] (A evicted, D added)
        """
        if is_specialized:
            cache = self._specialized_models
            order = self._specialized_cache_order
            max_size = self.MAX_SPECIALIZED_CACHE_SIZE
        else:
            cache = self._model_cache
            order = self._model_cache_order
            max_size = self.MAX_CACHE_SIZE
        
        # Update access order (move to end = most recently used)
        if key in order:
            order.remove(key)
        order.append(key)
        
        cache[key] = model
        
        # Evict oldest models if over limit
        while len(order) > max_size:
            oldest = order.pop(0)  # Remove from front (oldest)
            if oldest in cache:
                old_model = cache.pop(oldest)
                self._cleanup_model(old_model)  # Free memory
                logger.info(f"Evicted model from cache: {oldest}")
    
    def _cleanup_model(self, model: Any):
        """
        Clean up a model and free GPU memory.
        
        📖 WHY THIS MATTERS:
        GPU memory is limited! When we're done with a model,
        we need to:
        1. Move it to CPU (frees GPU memory)
        2. Delete the reference
        3. Tell PyTorch to empty its cache
        """
        try:
            if model is None:
                return
            
            # Handle specialized model dict format
            if isinstance(model, dict) and 'model' in model:
                actual_model = model['model']
                del model['model']
                model.clear()
                model = actual_model
            
            # Move model off GPU
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
            
            # Free GPU memory (tell PyTorch to release unused memory)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        except Exception as e:
            logger.warning(f"Error cleaning up model: {e}")
        
    def _load_config(self):
        """Load routing configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                
                for tool_name, models in data.get("assignments", {}).items():
                    self.assignments[tool_name] = [
                        ModelAssignment(**m) for m in models
                    ]
            except Exception as e:
                logger.warning(f"Error loading tool routing config: {e}")
                self.assignments = {}
        else:
            # Default assignments
            self._set_defaults()
            
    def _set_defaults(self):
        """Set default model assignments."""
        self.assignments = {
            "chat": [ModelAssignment("forge:default", "enigma_engine", priority=10)],
            "image": [ModelAssignment("local:stable-diffusion", "local", priority=10)],
            "code": [ModelAssignment("forge:default", "enigma_engine", priority=10)],
            "video": [ModelAssignment("local:animatediff", "local", priority=10)],
            "audio": [ModelAssignment("local:tts", "local", priority=10)],
            "3d": [ModelAssignment("local:shap-e", "local", priority=10)],
            "gif": [ModelAssignment("local:gif-maker", "local", priority=10)],
            "web": [ModelAssignment("local:web-tools", "local", priority=10)],
            "memory": [ModelAssignment("local:memory", "local", priority=10)],
            "embeddings": [ModelAssignment("local:embeddings", "local", priority=10)],
            "camera": [ModelAssignment("local:camera", "local", priority=10)],
            "vision": [ModelAssignment("local:vision", "local", priority=10)],
            "avatar": [ModelAssignment("local:avatar", "local", priority=10)],
        }
        self._save_config()
    
    def _load_specialized_config(self):
        """Load specialized models configuration."""
        try:
            config_path = Path(__file__).parent.parent.parent / "information" / "specialized_models.json"
            if not config_path.exists():
                logger.warning(f"Specialized models config not found: {config_path}")
                return
            
            with open(config_path) as f:
                config = json.load(f)
            
            if not config.get("enabled", False):
                logger.info("Specialized models disabled in config")
                return
            
            self._specialized_config = config
            logger.info(f"Loaded specialized models config: {list(config.get('models', {}).keys())}")
            
        except Exception as e:
            logger.warning(f"Failed to load specialized models config: {e}")
            self._specialized_config = {}
    
    def _get_shared_tokenizer(self):
        """Get or load the shared tokenizer for specialized models."""
        if self._shared_tokenizer is not None:
            return self._shared_tokenizer
        
        try:
            from .tokenizer import get_tokenizer

            # Try to load from configured path
            if hasattr(self, '_specialized_config'):
                tokenizer_path = self._specialized_config.get("shared_tokenizer")
                if tokenizer_path:
                    vocab_path = Path(__file__).parent.parent.parent / tokenizer_path
                    if vocab_path.exists():
                        self._shared_tokenizer = get_tokenizer("bpe", vocab_file=str(vocab_path))
                        logger.info(f"Loaded shared tokenizer from: {vocab_path}")
                        return self._shared_tokenizer
            
            # Fallback to default tokenizer
            self._shared_tokenizer = get_tokenizer()
            logger.info("Using default tokenizer for specialized models")
            return self._shared_tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load shared tokenizer: {e}")
            return None
    
    def _load_specialized_model(self, model_type: str):
        """
        Load a specialized model (router, vision, code, etc.).
        
        Args:
            model_type: Type of model to load (router, vision, code)
        
        Returns:
            Loaded model or None if failed
        """
        if not hasattr(self, '_specialized_config') or not self._specialized_config:
            return None
        
        model_config = self._specialized_config.get('models', {}).get(model_type)
        if not model_config:
            logger.warning(f"No config for specialized model: {model_type}")
            return None
        
        model_path = Path(__file__).parent.parent.parent / model_config['path']
        
        if not model_path.exists():
            logger.warning(f"Specialized model not found: {model_path}")
            logger.info(f"Train it with: python scripts/train_specialized_model.py --type {model_type}")
            return None
        
        try:
            import torch

            from .model import Forge

            # Load model checkpoint (weights_only=False needed for full checkpoint with config)
            # This is safe because we only load from our own trained models directory
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Get tokenizer
            tokenizer = self._get_shared_tokenizer()
            if tokenizer is None:
                logger.error("Cannot load specialized model without tokenizer")
                return None
            
            # Create model from config
            model = Forge.from_config(checkpoint['config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            logger.info(f"Loaded specialized {model_type} model from: {model_path}")
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'config': checkpoint.get('config', {}),
                'model_type': model_type,
                'type': 'specialized'
            }
            
        except Exception as e:
            logger.error(f"Failed to load specialized {model_type} model: {e}")
            return None
    
    def classify_intent(self, text: str) -> str:
        """
        Classify user intent to determine which tool to use.
        
        📖 WHAT THIS DOES:
        Analyzes user input and figures out what they want:
        - "draw a cat" → "image"
        - "explain quantum physics" → "chat"
        - "write a Python function" → "code"
        
        📐 TWO APPROACHES:
        
        1. SPECIALIZED ROUTER MODEL (if use_specialized=True):
           - Uses a trained neural network to classify intent
           - More accurate for ambiguous requests
           - Requires trained router model
        
        2. KEYWORD MATCHING (fallback):
           - Looks for trigger words in user input
           - Fast but less intelligent
           - "draw" → image, "explain" → chat, etc.
        
        📐 FLOW:
        
            User: "Create a painting of a dragon"
                    │
                    ▼
            ┌───────────────────────────────────────┐
            │  TRY: Specialized Router Model        │
            │  Input: "Create a painting of dragon" │
            │  Output: "image"                      │
            └───────────────────────────────────────┘
                    │ (success)
                    ▼
            Return: "image"
        
        Args:
            text: User input text to classify
        
        Returns:
            Intent class (chat, vision, image, code, video, audio, 3d, gif, web)
        """
        # ─────────────────────────────────────────────────────────────────────
        # APPROACH 1: Try specialized router model first (more accurate)
        # ─────────────────────────────────────────────────────────────────────
        if self.use_specialized:
            router_model = self._specialized_models.get('router')
            if router_model is None and 'router' not in self._specialized_models:
                # Try to load it (only once - None means we tried and failed)
                router_model = self._load_specialized_model('router')
                self._specialized_models['router'] = router_model
            
            if router_model:
                try:
                    import torch
                    
                    model = router_model['model']
                    tokenizer = router_model['tokenizer']
                    
                    # Prepare input in the format the router was trained on
                    # Format: "Q: <user question>\nA: [E:tool]"
                    prompt = f"Q: {text}\nA: [E:tool]"
                    input_ids = tokenizer.encode(prompt)
                    input_tensor = torch.tensor([input_ids])
                    
                    # Generate classification
                    with torch.no_grad():
                        output = model.generate(
                            input_tensor,
                            max_new_tokens=10,
                            temperature=0.1,  # Low temperature = deterministic
                        )
                    
                    # Decode output and extract intent
                    result = tokenizer.decode(output[0].tolist())
                    
                    # Extract intent using regex (expected: "Q: ... A: [E:tool]intent")
                    import re
                    match = re.search(r'\[E:tool\](\w+)', result)
                    if match:
                        intent = match.group(1).lower()
                        if intent in self.routing_rules:
                            logger.info(f"Router classified intent: {intent}")
                            return intent
                    
                    # Fallback: try simple string splitting
                    if '[E:tool]' in result:
                        parts = result.split('[E:tool]')
                        if len(parts) > 1:
                            intent_text = parts[-1].strip()
                            intent = intent_text.split()[0].lower() if intent_text else None
                            if intent and intent in self.routing_rules:
                                logger.info(f"Router classified intent (fallback): {intent}")
                                return intent
                    
                except Exception as e:
                    logger.warning(f"Specialized router failed: {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # APPROACH 2: Fallback to keyword-based detection
        # ─────────────────────────────────────────────────────────────────────
        return self.detect_tool(text)
    
    def describe_image(self, features: str) -> str:
        """
        Generate image description using specialized vision model.
        
        Args:
            features: Comma-separated vision features/labels
        
        Returns:
            Natural language description
        """
        if self.use_specialized:
            vision_model = self._specialized_models.get('vision')
            if vision_model is None and 'vision' not in self._specialized_models:
                vision_model = self._load_specialized_model('vision')
                self._specialized_models['vision'] = vision_model
            
            if vision_model:
                try:
                    import torch
                    
                    model = vision_model['model']
                    tokenizer = vision_model['tokenizer']
                    
                    # Prepare input
                    prompt = f"Q: [E:vision] {features}\nA: "
                    input_ids = tokenizer.encode(prompt)
                    input_tensor = torch.tensor([input_ids])
                    
                    # Generate
                    with torch.no_grad():
                        output = model.generate(
                            input_tensor,
                            max_new_tokens=64,
                            temperature=0.7,
                        )
                    
                    # Decode
                    result = tokenizer.decode(output[0].tolist())
                    
                    # Extract description after "A: "
                    if "A: " in result:
                        description = result.split("A: ", 1)[1].strip()
                        logger.info(f"Vision model generated description")
                        return description
                    
                except Exception as e:
                    logger.warning(f"Specialized vision model failed: {e}")
        
        # Fallback to simple description
        return f"I see: {features}"
    
    def generate_code(self, prompt: str) -> str:
        """
        Generate code using specialized code model.
        
        Args:
            prompt: Code generation prompt
        
        Returns:
            Generated code
        """
        if self.use_specialized:
            code_model = self._specialized_models.get('code')
            if code_model is None and 'code' not in self._specialized_models:
                code_model = self._load_specialized_model('code')
                self._specialized_models['code'] = code_model
            
            if code_model:
                try:
                    import torch
                    
                    model = code_model['model']
                    tokenizer = code_model['tokenizer']
                    
                    # Prepare input
                    input_text = f"Q: {prompt}\nA: "
                    input_ids = tokenizer.encode(input_text)
                    input_tensor = torch.tensor([input_ids])
                    
                    # Generate
                    with torch.no_grad():
                        output = model.generate(
                            input_tensor,
                            max_new_tokens=256,
                            temperature=0.7,
                        )
                    
                    # Decode
                    result = tokenizer.decode(output[0].tolist())
                    
                    # Extract code after "A: "
                    if "A: " in result:
                        code = result.split("A: ", 1)[1].strip()
                        logger.info(f"Code model generated response")
                        return code
                    
                except Exception as e:
                    logger.warning(f"Specialized code model failed: {e}")
        
        # Fallback - return prompt
        return f"# {prompt}\n# (No specialized code model available)"
    
    # ─────────────────────────────────────────────────────────────────────
    # TRAINER AI: Generates and curates training data for other AIs
    # ─────────────────────────────────────────────────────────────────────
    
    def get_trainer_ai(self):
        """Get the TrainerAI instance for data generation and curation."""
        from .trainer_ai import get_trainer_ai
        return get_trainer_ai(model=self.default_model)
    
    def generate_training_data(
        self,
        position: str = "chat",
        count: int = 100,
        seed_data: str = None,
        text: str = None,
        style: str = "qa"
    ) -> str:
        """
        Generate training data for any router position using the Trainer AI.
        
        This is the "AI that trains AIs" - it generates properly formatted
        training data for any specialized model in the router system.
        
        Args:
            position: Router position to generate data for (router, vision, code, etc.)
            count: Number of examples to generate
            seed_data: Optional existing data to augment
            text: (Legacy) Raw text to convert to Q&A pairs
            style: (Legacy) Output format for legacy text conversion
        
        Returns:
            Formatted training data for the specified position
        """
        trainer = self.get_trainer_ai()
        
        # Legacy support: if text provided, convert to Q&A format
        if text is not None:
            return self._convert_text_to_qa(text, style)
        
        # Try specialized trainer model first for AI-powered generation
        if self.use_specialized:
            trainer_model = self._specialized_models.get('trainer')
            if trainer_model is None and 'trainer' not in self._specialized_models:
                trainer_model = self._load_specialized_model('trainer')
                self._specialized_models['trainer'] = trainer_model
            
            if trainer_model:
                try:
                    # Use AI-powered generation
                    trainer.model = trainer_model.get('model')
                    trainer.use_ai = True
                except Exception as e:
                    logger.warning(f"Could not enable AI generation: {e}")
        
        # Generate data using TrainerAI
        return trainer.generate_training_data(position, count=count, seed_data=seed_data)
    
    def curate_training_data(
        self,
        position: str,
        raw_data: str,
        remove_duplicates: bool = True,
        validate_format: bool = True,
        filter_low_quality: bool = True,
    ):
        """
        Curate and clean training data for a router position.
        
        Args:
            position: Router position (router, vision, code, etc.)
            raw_data: Raw training data to curate
            remove_duplicates: Remove duplicate entries
            validate_format: Check format validity
            filter_low_quality: Remove low-quality entries
        
        Returns:
            Tuple of (cleaned data string, DataQualityScore)
        """
        trainer = self.get_trainer_ai()
        return trainer.curate_data(
            position, raw_data,
            remove_duplicates=remove_duplicates,
            validate_format=validate_format,
            filter_low_quality=filter_low_quality,
        )
    
    def get_data_template(self, position: str, examples: int = 3) -> str:
        """
        Get example template showing the data format for a position.
        
        Args:
            position: Router position (router, vision, code, avatar, etc.)
            examples: Number of example entries to include
        
        Returns:
            Template string showing the expected format
        """
        trainer = self.get_trainer_ai()
        return trainer.get_template(position, count=examples)
    
    def get_available_positions(self) -> list:
        """Get list of all router positions that can be trained."""
        trainer = self.get_trainer_ai()
        return trainer.get_positions()
    
    def _convert_text_to_qa(self, text: str, style: str) -> str:
        """Legacy: Convert raw text to Q&A training format."""
        # Try specialized trainer model first
        if self.use_specialized:
            trainer_model = self._specialized_models.get('trainer')
            if trainer_model is None and 'trainer' not in self._specialized_models:
                trainer_model = self._load_specialized_model('trainer')
                self._specialized_models['trainer'] = trainer_model
            
            if trainer_model:
                try:
                    return self._trainer_generate(trainer_model, text, style)
                except Exception as e:
                    logger.warning(f"Specialized trainer model failed: {e}")
        
        # Fallback: rule-based Q&A generation
        return self._generate_qa_fallback(text)
    
    def _trainer_generate(self, trainer_model, text: str, style: str) -> str:
        """Generate training data using specialized trainer model."""
        import torch
        
        model = trainer_model['model']
        tokenizer = trainer_model['tokenizer']
        
        prompt = f"Convert this text to {style} training data:\n{text}\n\nTraining data:"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids])
        
        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=512,
                temperature=0.7,
            )
        
        result = tokenizer.decode(output[0].tolist())
        if "Training data:" in result:
            return result.split("Training data:", 1)[1].strip()
        return result
    
    def _generate_qa_fallback(self, text: str) -> str:
        """
        Rule-based Q&A generation when no trainer model is available.
        
        Splits text into paragraphs and creates questions from them.
        """
        
        lines = []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for i, para in enumerate(paragraphs[:20]):  # Limit to 20 pairs
            # Skip very short paragraphs
            if len(para) < 30:
                continue
            
            # Generate a question based on the paragraph
            # Look for key phrases to make relevant questions
            question = self._extract_question(para)
            
            lines.append(f"Q: {question}")
            lines.append(f"A: {para}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _extract_question(self, paragraph: str) -> str:
        """Extract a relevant question from a paragraph."""
        import re
        
        # Try to find the main topic/subject
        sentences = paragraph.split('.')
        first_sentence = sentences[0].strip() if sentences else paragraph[:100]
        
        # Common patterns to convert to questions
        patterns = [
            (r'^(\w+) is ', r'What is \1?'),
            (r'^(\w+) are ', r'What are \1?'),
            (r'^The (\w+)', r'What is the \1?'),
            (r'^A (\w+)', r'What is a \1?'),
            (r'how to ', r'How do you '),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, first_sentence, re.IGNORECASE):
                # Found a pattern, create question
                match = re.search(r'^\w+', first_sentence)
                if match:
                    topic = match.group()
                    return f"What is {topic}?"
        
        # Default: ask about the topic
        words = first_sentence.split()[:5]
        topic = ' '.join(words)
        return f"Tell me about {topic}?"
    
    def evaluate_response(self, question: str, response: str) -> dict:
        """
        Evaluate an AI response quality (Trainer AI as critic).
        
        Args:
            question: The original question
            response: The AI's response to evaluate
        
        Returns:
            Dict with scores and feedback
        """
        # Try specialized trainer model
        if self.use_specialized:
            trainer_model = self._specialized_models.get('trainer')
            if trainer_model:
                try:
                    return self._trainer_evaluate(trainer_model, question, response)
                except Exception as e:
                    logger.warning(f"Trainer evaluation failed: {e}")
        
        # Fallback: simple heuristic evaluation
        return self._evaluate_response_fallback(question, response)
    
    def _trainer_evaluate(self, trainer_model, question: str, response: str) -> dict:
        """Evaluate using trainer model."""
        import torch
        
        model = trainer_model['model']
        tokenizer = trainer_model['tokenizer']
        
        prompt = f"Rate this response (1-10 for quality, relevance, helpfulness):\nQ: {question}\nA: {response}\n\nRating:"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids])
        
        with torch.no_grad():
            output = model.generate(input_tensor, max_new_tokens=50, temperature=0.3)
        
        result = tokenizer.decode(output[0].tolist())
        
        # Parse rating
        try:
            if "Rating:" in result:
                rating_text = result.split("Rating:", 1)[1].strip()
                score = float(rating_text.split()[0])
                return {"score": score / 10.0, "feedback": rating_text}
        except Exception:
            pass
        
        return {"score": 0.5, "feedback": "Could not parse evaluation"}
    
    def _evaluate_response_fallback(self, question: str, response: str) -> dict:
        """Simple heuristic response evaluation."""
        score = 0.5
        feedback = []
        
        # Check length
        if len(response) < 10:
            score -= 0.2
            feedback.append("Response too short")
        elif len(response) > 50:
            score += 0.1
            feedback.append("Good length")
        
        # Check if response relates to question
        q_words = set(question.lower().split())
        r_words = set(response.lower().split())
        overlap = len(q_words & r_words)
        if overlap > 2:
            score += 0.1
            feedback.append("Good relevance")
        
        # Check for repetition
        words = response.split()
        if len(words) != len(set(words)):
            repetition = 1 - (len(set(words)) / len(words))
            if repetition > 0.3:
                score -= 0.15
                feedback.append("Too repetitive")
        
        # Bounds
        score = max(0.0, min(1.0, score))
        
        return {
            "score": score,
            "feedback": "; ".join(feedback) if feedback else "Average response"
        }
    
    def improve_response(self, question: str, response: str, feedback: str = "") -> str:
        """
        Use Trainer AI to improve an AI response.
        
        Args:
            question: Original question
            response: Current response to improve
            feedback: Optional feedback about what's wrong
        
        Returns:
            Improved response
        """
        if self.use_specialized:
            trainer_model = self._specialized_models.get('trainer')
            if trainer_model:
                try:
                    import torch
                    
                    model = trainer_model['model']
                    tokenizer = trainer_model['tokenizer']
                    
                    prompt = f"Improve this response:\nQ: {question}\nOriginal A: {response}\nFeedback: {feedback}\n\nImproved A:"
                    input_ids = tokenizer.encode(prompt)
                    input_tensor = torch.tensor([input_ids])
                    
                    with torch.no_grad():
                        output = model.generate(input_tensor, max_new_tokens=256, temperature=0.7)
                    
                    result = tokenizer.decode(output[0].tolist())
                    if "Improved A:" in result:
                        return result.split("Improved A:", 1)[1].strip()
                    return result
                except Exception as e:
                    logger.warning(f"Trainer improve failed: {e}")
        
        # Fallback: return original
        return response
            
    def _save_config(self):
        """Save routing configuration."""
        data = {
            "assignments": {
                tool: [
                    {"model_id": m.model_id, "model_type": m.model_type, 
                     "priority": m.priority, "config": m.config}
                    for m in models
                ]
                for tool, models in self.assignments.items()
            },
            "updated": datetime.now().isoformat()
        }
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)
            
    def assign_model(self, tool_name: str, model_id: str, priority: int = 10, 
                     config: Optional[Dict] = None):
        """
        Assign a model to a tool.
        
        Args:
            tool_name: Tool to assign to (chat, image, code, etc.)
            model_id: Model identifier:
                - "forge:model_name" - Forge model from registry
                - "huggingface:repo/model" - HuggingFace model
                - "local:name" - Local module (stable-diffusion, etc.)
                - "api:provider" - API provider (openai, replicate)
            priority: Higher = tried first (default 10)
            config: Optional model-specific config
        """
        if tool_name not in self.routing_rules:
            raise ValueError(f"Unknown tool: {tool_name}. Available: {list(self.routing_rules.keys())}")
        
        # Parse model type from ID
        if ":" in model_id:
            model_type = model_id.split(":")[0]
        else:
            model_type = "enigma_engine"  # Default
            model_id = f"forge:{model_id}"
            
        assignment = ModelAssignment(
            model_id=model_id,
            model_type=model_type,
            priority=priority,
            config=config or {}
        )
        
        if tool_name not in self.assignments:
            self.assignments[tool_name] = []
            
        # Check if already assigned (update priority/config)
        for i, existing in enumerate(self.assignments[tool_name]):
            if existing.model_id == model_id:
                self.assignments[tool_name][i] = assignment
                self._save_config()
                return
                
        # Add new assignment
        self.assignments[tool_name].append(assignment)
        # Sort by priority (highest first)
        self.assignments[tool_name].sort(key=lambda x: -x.priority)
        self._save_config()
        
    def unassign_model(self, tool_name: str, model_id: str):
        """Remove a model from a tool."""
        if tool_name in self.assignments:
            self.assignments[tool_name] = [
                m for m in self.assignments[tool_name] if m.model_id != model_id
            ]
            self._save_config()
            
    def get_assignments(self, tool_name: str) -> List[ModelAssignment]:
        """Get all models assigned to a tool."""
        return self.assignments.get(tool_name, [])
    
    def get_all_assignments(self) -> Dict[str, List[ModelAssignment]]:
        """Get all tool assignments."""
        return self.assignments.copy()
    
    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded in cache."""
        return model_id in self._model_cache
    
    def get_loaded_models(self) -> List[str]:
        """Get list of all currently loaded model IDs."""
        return list(self._model_cache.keys())
    
    def get_active_ai(self) -> Optional[Dict[str, Any]]:
        """
        Get info about the currently active AI (chat model).
        
        Returns dict with:
            - model_id: Full model identifier (e.g., "huggingface:microsoft/DialoGPT-medium")
            - model_type: Type (forge, huggingface, api, local)
            - model_name: Short name
            - loaded: Whether model is currently loaded
            - priority: Priority level
        
        Returns None if no chat model configured.
        """
        chat_assignments = self.get_assignments("chat")
        if not chat_assignments:
            return None
        
        primary = chat_assignments[0]  # Highest priority
        model_id = primary.model_id
        
        # Parse model type
        if ":" in model_id:
            model_type, model_name = model_id.split(":", 1)
        else:
            model_type = "unknown"
            model_name = model_id
        
        return {
            "model_id": model_id,
            "model_type": model_type,
            "model_name": model_name,
            "loaded": self.is_model_loaded(model_id),
            "priority": primary.priority,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive router status for display.
        
        Returns dict with:
            - active_ai: Info about primary chat model
            - loaded_count: Number of loaded models
            - assigned_tools: List of tools with assignments
            - total_assignments: Total model assignments
        """
        active = self.get_active_ai()
        loaded = self.get_loaded_models()
        
        assigned_tools = [
            tool for tool, models in self.assignments.items() 
            if models
        ]
        
        total = sum(len(m) for m in self.assignments.values())
        
        return {
            "active_ai": active,
            "loaded_count": len(loaded),
            "loaded_models": loaded,
            "assigned_tools": assigned_tools,
            "total_assignments": total,
        }
    
    def register_handler(self, tool_name: str, handler: Callable):
        """Register a handler function for a tool."""
        self._handlers[tool_name] = handler
        
    def detect_tool(self, text: str) -> Optional[str]:
        """
        Detect which tool should handle a request based on keywords.
        
        Returns tool name or None if unclear (use chat).
        """
        text_lower = text.lower()
        
        scores = {}
        for tool_name, routing_rule in self.routing_rules.items():
            score = 0
            for keyword in routing_rule.keywords:
                if keyword in text_lower:
                    score += 1
                    # Exact phrase match = bonus
                    if f" {keyword} " in f" {text_lower} ":
                        score += 2
            if score > 0:
                scores[tool_name] = score
                
        if not scores:
            return "chat"  # Default
            
        return max(scores, key=lambda k: scores.get(k, 0))
    
    def parse_tool_call(self, ai_output: str) -> Optional[Dict[str, Any]]:
        """
        Parse AI output for tool call syntax.
        
        Supports formats:
        - {"tool": "image", "prompt": "..."}
        - [TOOL: image] prompt: ...
        - /image prompt...
        """
        # JSON format
        try:
            if "{" in ai_output and "}" in ai_output:
                # Find JSON object
                start = ai_output.index("{")
                end = ai_output.rindex("}") + 1
                json_str = ai_output[start:end]
                data = json.loads(json_str)
                if "tool" in data:
                    return data
        except (json.JSONDecodeError, ValueError):
            pass
            
        # [TOOL: name] format
        match = re.search(r'\[TOOL:\s*(\w+)\](.+)', ai_output, re.IGNORECASE | re.DOTALL)
        if match:
            tool_name = match.group(1).lower()
            params_text = match.group(2).strip()
            return {"tool": tool_name, "prompt": params_text}
            
        # /command format
        match = re.search(r'^/(\w+)\s*(.*)$', ai_output.strip(), re.MULTILINE)
        if match:
            tool_name = match.group(1).lower()
            params_text = match.group(2).strip()
            return {"tool": tool_name, "prompt": params_text}
            
        return None
    
    def load_model(self, model_id: str) -> Any:
        """Load a model by ID (cached)."""
        if model_id in self._model_cache:
            return self._model_cache[model_id]
            
        model_type, model_name = model_id.split(":", 1) if ":" in model_id else ("enigma_engine", model_id)
        
        model = None
        
        if model_type == "enigma_engine":
            model = self._load_forge_model(model_name)
        elif model_type == "huggingface":
            model = self._load_huggingface_model(model_name)
        elif model_type == "local":
            model = self._load_local_module(model_name)
        elif model_type == "api":
            model = self._load_api_provider(model_name)
            
        if model:
            self._model_cache[model_id] = model
            
        return model
    
    def _load_forge_model(self, name: str) -> Any:
        """Load a Enigma AI Engine model."""
        try:
            from .model_registry import ModelRegistry
            registry = ModelRegistry()
            
            if name == "default":
                # Get first available model - list_models returns Dict[str, Any]
                models_dict = registry.list_models()
                if models_dict:
                    # Get first key (model name)
                    name = next(iter(models_dict.keys()), "small_enigma_engine")
                else:
                    return None
                    
            model, config = registry.load_model(name)
            return {"model": model, "config": config, "type": "enigma_engine"}
        except Exception as e:
            logger.warning(f"Failed to load Forge model {name}: {e}")
            return None
            
    def _load_huggingface_model(self, repo_id: str) -> Any:
        """Load a HuggingFace model for text generation."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            return {
                "model": model, 
                "tokenizer": tokenizer, 
                "type": "huggingface",
                "device": device
            }
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace model {repo_id}: {e}")
            return None
            
    def _load_local_module(self, name: str) -> Any:
        """Get a local module handler."""
        # These are handled by the module system, just return a marker
        return {"type": "local", "module": name}
        
    def _load_api_provider(self, provider: str) -> Any:
        """Get an API provider handler."""
        return {"type": "api", "provider": provider}
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given parameters.
        
        Returns:
            {"success": bool, "result": Any, "error": Optional[str]}
        """
        if tool_name not in self.routing_rules:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
            
        # Get assigned models
        assignments = self.get_assignments(tool_name)
        if not assignments:
            return {"success": False, "error": f"No model assigned to tool: {tool_name}"}
            
        # Try each model in priority order
        for assignment in assignments:
            try:
                result = self._execute_with_model(tool_name, assignment, params)
                if result.get("success"):
                    return result
            except Exception as e:
                logger.debug(f"Model {assignment.model_id} failed: {e}")
                continue
                
        return {"success": False, "error": "All assigned models failed"}
    
    def _execute_with_model(self, tool_name: str, assignment: ModelAssignment, 
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool with a specific model."""
        
        # Check for registered handler first
        if tool_name in self._handlers:
            return self._handlers[tool_name](params, assignment)
            
        model_type = assignment.model_type
        
        # Route HuggingFace models based on tool type
        if model_type == "huggingface":
            if tool_name == "image":
                return self._execute_hf_image(assignment, params)
            elif tool_name == "audio":
                return self._execute_hf_audio(assignment, params)
            elif tool_name == "video":
                return self._execute_hf_video(assignment, params)
            elif tool_name == "3d":
                return self._execute_hf_3d(assignment, params)
            elif tool_name == "vision":
                return self._execute_hf_vision(assignment, params)
            elif tool_name == "embeddings":
                return self._execute_hf_embeddings(assignment, params)
            elif tool_name in ("chat", "code"):
                return self._execute_text_generation(assignment, params)
            else:
                return self._execute_text_generation(assignment, params)
        elif model_type == "enigma_engine":
            return self._execute_text_generation(assignment, params)
        elif model_type == "local":
            return self._execute_local_module(assignment, params)
        elif model_type == "api":
            return self._execute_api_call(assignment, params)
            
        return {"success": False, "error": f"Unknown model type: {model_type}"}
    
    def _execute_text_generation(self, assignment: ModelAssignment, 
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text generation with Forge or HuggingFace model."""
        model_data = self.load_model(assignment.model_id)
        if not model_data:
            return {"success": False, "error": "Failed to load model"}
            
        prompt = params.get("prompt", "")
        max_tokens = params.get("max_tokens", 256)
        
        if model_data["type"] == "enigma_engine":
            # Use Forge inference
            try:
                from .inference import EnigmaEngine
                engine = EnigmaEngine(model_data["model"], model_data["config"])
                max_gen = params.get("max_tokens", 256)
                result = engine.generate(prompt, max_gen=max_gen)
                return {"success": True, "result": result, "model": assignment.model_id}
            except Exception as e:
                return {"success": False, "error": str(e)}
                
        elif model_data["type"] == "huggingface":
            # Use HuggingFace generation
            try:
                model = model_data["model"]
                tokenizer = model_data["tokenizer"]
                
                inputs = tokenizer(prompt, return_tensors="pt")
                if model_data.get("device") == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt from output
                if result.startswith(prompt):
                    result = result[len(prompt):].strip()
                    
                return {"success": True, "result": result, "model": assignment.model_id}
            except Exception as e:
                return {"success": False, "error": str(e)}
                
        return {"success": False, "error": "Unknown model type"}
    
    def _execute_local_module(self, assignment: ModelAssignment,
                             params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a local module (stable-diffusion, code, audio, video, 3d, etc.)."""
        module_name = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
        
        try:
            # Route to appropriate handler based on module name
            if module_name in ("stable-diffusion", "sd", "image"):
                return self._execute_image_generation(params)
            elif module_name in ("code", "forge-code"):
                return self._execute_code_generation(params)
            elif module_name in ("tts", "audio", "speech"):
                return self._execute_audio_generation(params)
            elif module_name in ("video", "animatediff"):
                return self._execute_video_generation(params)
            elif module_name in ("3d", "shap-e", "threed"):
                return self._execute_3d_generation(params)
            
            # For unknown modules, return marker for GUI to handle
            return {
                "success": True, 
                "result": {"action": "execute_module", "module": module_name, "params": params},
                "model": assignment.model_id
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_image_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image generation using the local SD provider."""
        try:
            # Check if image generation module is available
            if not self.is_module_available("image"):
                return {
                    "success": False, 
                    "error": self.get_tool_availability_message("image")
                }
            
            from ..gui.tabs.image_tab import get_provider
            
            provider = get_provider('local')
            if provider is None:
                return {"success": False, "error": "Image provider not available"}
            
            # Load if needed
            if not provider.is_loaded:
                if not provider.load():
                    return {"success": False, "error": "Failed to load Stable Diffusion model"}
            
            # Generate
            prompt = params.get("prompt", "")
            width = int(params.get("width", 512))
            height = int(params.get("height", 512))
            steps = int(params.get("steps", 30))
            guidance = float(params.get("guidance", 7.5))
            negative_prompt = params.get("negative_prompt", "")
            
            result = provider.generate(
                prompt,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                negative_prompt=negative_prompt
            )
            
            return result
            
        except ImportError as e:
            return {"success": False, "error": f"Image generation not available: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_code_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code generation, routing to language-specific models if available."""
        try:
            # Check if code generation module is available
            if not self.is_module_available("code"):
                return {
                    "success": False, 
                    "error": self.get_tool_availability_message("code")
                }
            
            prompt = params.get("prompt", "")
            language = params.get("language", "python").lower()
            
            # Map language to language-specific tool key
            language_tool_map = {
                "python": "code_python",
                "py": "code_python",
                "javascript": "code_javascript",
                "js": "code_javascript",
                "typescript": "code_javascript",
                "ts": "code_javascript",
                "rust": "code_rust",
                "rs": "code_rust",
                "cpp": "code_cpp",
                "c++": "code_cpp",
                "c": "code_cpp",
                "java": "code_java",
                "kotlin": "code_java",
            }
            
            # Check if language-specific model is assigned
            lang_tool = language_tool_map.get(language)
            if lang_tool:
                lang_assignments = self.get_assignments(lang_tool)
                if lang_assignments:
                    # Use language-specific model
                    for assignment in lang_assignments:
                        try:
                            result = self._execute_with_language_model(
                                assignment, prompt, language
                            )
                            if result.get("success"):
                                result["model_type"] = f"specialized_{language}"
                                return result
                        except Exception as e:
                            logger.debug(f"Language model {assignment.model_id} failed: {e}")
                            continue
            
            # Fallback to general code provider
            from ..gui.tabs.code_tab import get_provider
            
            provider = get_provider('local')
            if provider is None:
                return {"success": False, "error": "Code provider not available"}
            
            if not provider.is_loaded:
                if not provider.load():
                    return {"success": False, "error": "Failed to load code model"}
            
            result = provider.generate(prompt, language=language)
            return result
            
        except ImportError as e:
            return {"success": False, "error": f"Code generation not available: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_with_language_model(self, assignment: ModelAssignment, 
                                    prompt: str, language: str) -> Dict[str, Any]:
        """Execute code generation with a language-specific model."""
        model_data = self.load_model(assignment.model_id)
        if not model_data:
            return {"success": False, "error": "Failed to load language-specific model"}
        
        # Build language-aware prompt
        code_prompt = f"Write {language} code:\n{prompt}\n\n```{language}\n"
        
        if model_data["type"] == "enigma_engine":
            try:
                from .inference import EnigmaEngine
                engine = EnigmaEngine(model_data["model"], model_data["config"])
                result = engine.generate(code_prompt, max_gen=500)
                return {
                    "success": True, 
                    "code": result, 
                    "model": assignment.model_id,
                    "language": language
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif model_data["type"] == "huggingface":
            try:
                model = model_data["model"]
                tokenizer = model_data["tokenizer"]
                
                inputs = tokenizer(code_prompt, return_tensors="pt")
                if model_data.get("device") == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if result.startswith(code_prompt):
                    result = result[len(code_prompt):].strip()
                
                return {
                    "success": True, 
                    "code": result, 
                    "model": assignment.model_id,
                    "language": language
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": f"Unknown model type: {model_data['type']}"}
    
    def _execute_audio_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute audio/TTS generation using local provider."""
        try:
            # Check if audio generation module is available
            if not self.is_module_available("audio"):
                return {
                    "success": False, 
                    "error": self.get_tool_availability_message("audio")
                }
            
            from ..gui.tabs.audio_tab import get_provider
            
            provider = get_provider('local')
            if provider is None:
                return {"success": False, "error": "Audio provider not available"}
            
            if not provider.is_loaded:
                if not provider.load():
                    return {"success": False, "error": "Failed to load TTS engine"}
            
            text = params.get("text", params.get("prompt", ""))
            
            result = provider.generate(text)
            return result
            
        except ImportError as e:
            return {"success": False, "error": f"Audio generation not available: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_video_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute video generation using local AnimateDiff."""
        try:
            # Check if video generation module is available
            if not self.is_module_available("video"):
                return {
                    "success": False, 
                    "error": self.get_tool_availability_message("video")
                }
            
            from ..gui.tabs.video_tab import get_provider
            
            provider = get_provider('local')
            if provider is None:
                return {"success": False, "error": "Video provider not available"}
            
            if not provider.is_loaded:
                if not provider.load():
                    return {"success": False, "error": "Failed to load video model"}
            
            prompt = params.get("prompt", "")
            duration = float(params.get("duration", 2.0))
            fps = int(params.get("fps", 8))
            
            result = provider.generate(prompt, duration=duration, fps=fps)
            return result
            
        except ImportError as e:
            return {"success": False, "error": f"Video generation not available: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_3d_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute 3D model generation using local Shap-E."""
        try:
            # Check if 3D generation module is available
            if not self.is_module_available("3d"):
                return {
                    "success": False, 
                    "error": self.get_tool_availability_message("3d")
                }
            
            from ..gui.tabs.threed_tab import get_provider
            
            provider = get_provider('local')
            if provider is None:
                return {"success": False, "error": "3D provider not available"}
            
            if not provider.is_loaded:
                if not provider.load():
                    return {"success": False, "error": "Failed to load 3D model"}
            
            prompt = params.get("prompt", "")
            guidance_scale = float(params.get("guidance_scale", 15.0))
            num_inference_steps = int(params.get("num_inference_steps", 64))
            
            result = provider.generate(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            return result
            
        except ImportError as e:
            return {"success": False, "error": f"3D generation not available: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # =========================================================================
    # HuggingFace Model Execution Methods
    # =========================================================================
    
    def _execute_hf_image(self, assignment: ModelAssignment, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image generation with HuggingFace diffusion model."""
        try:
            import time
            from pathlib import Path

            import torch
            from diffusers import DiffusionPipeline
            
            model_id = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
            
            # Check cache
            if model_id not in self._model_cache:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Loading HF image model: {model_id}")
                pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    safety_checker=None,
                )
                pipe = pipe.to(device)
                
                if device == "cuda":
                    try:
                        pipe.enable_attention_slicing()
                    except Exception as e:
                        logger.debug(f"Could not enable attention slicing: {e}")
                
                self._cache_model(model_id, {"pipe": pipe, "type": "hf_image", "device": device})
            
            model_data = self._model_cache[model_id]
            pipe = model_data["pipe"]
            
            prompt = params.get("prompt", "")
            width = int(params.get("width", 512))
            height = int(params.get("height", 512))
            steps = int(params.get("steps", 20))
            
            output = pipe(prompt, width=width, height=height, num_inference_steps=steps)
            image = output.images[0]
            
            # Save
            from ..config import CONFIG
            output_dir = Path(CONFIG.get("outputs_dir", "outputs")) / "images"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            filepath = output_dir / f"hf_image_{timestamp}.png"
            image.save(str(filepath))
            
            return {"success": True, "path": str(filepath), "model": model_id}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_hf_audio(self, assignment: ModelAssignment, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute audio/TTS with HuggingFace model."""
        try:
            import time
            from pathlib import Path

            import scipy.io.wavfile as wavfile
            import torch
            
            model_id = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
            text = params.get("text", params.get("prompt", ""))
            
            # Different handling for different TTS models
            if "bark" in model_id.lower():
                from transformers import AutoProcessor, BarkModel
                
                if model_id not in self._model_cache:
                    processor = AutoProcessor.from_pretrained(model_id)
                    model = BarkModel.from_pretrained(model_id)
                    if torch.cuda.is_available():
                        model = model.to("cuda")
                    self._cache_model(model_id, {"processor": processor, "model": model, "type": "bark"})
                
                data = self._model_cache[model_id]
                inputs = data["processor"](text, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                audio = data["model"].generate(**inputs)
                audio_np = audio.cpu().numpy().squeeze()
                sample_rate = 24000
                
            elif "kokoro" in model_id.lower() or "melo" in model_id.lower():
                from transformers import pipeline
                
                if model_id not in self._model_cache:
                    pipe = pipeline("text-to-speech", model=model_id)
                    self._cache_model(model_id, {"pipe": pipe, "type": "tts_pipe"})
                
                data = self._model_cache[model_id]
                output = data["pipe"](text)
                audio_np = output["audio"]
                sample_rate = output.get("sampling_rate", 22050)
            else:
                # Generic TTS pipeline
                from transformers import pipeline
                
                if model_id not in self._model_cache:
                    pipe = pipeline("text-to-speech", model=model_id)
                    self._cache_model(model_id, {"pipe": pipe, "type": "tts_pipe"})
                
                data = self._model_cache[model_id]
                output = data["pipe"](text)
                audio_np = output["audio"]
                sample_rate = output.get("sampling_rate", 22050)
            
            # Save audio
            from ..config import CONFIG
            output_dir = Path(CONFIG.get("outputs_dir", "outputs")) / "audio"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            filepath = output_dir / f"hf_audio_{timestamp}.wav"
            
            import numpy as np
            if audio_np.dtype != np.int16:
                audio_np = (audio_np * 32767).astype(np.int16)
            wavfile.write(str(filepath), sample_rate, audio_np)
            
            return {"success": True, "path": str(filepath), "model": model_id}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_hf_video(self, assignment: ModelAssignment, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute video generation with HuggingFace model."""
        try:
            import time
            from pathlib import Path

            import torch
            from diffusers import DiffusionPipeline
            
            model_id = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
            
            if model_id not in self._model_cache:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Loading HF video model: {model_id}")
                pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
                pipe = pipe.to(device)
                self._cache_model(model_id, {"pipe": pipe, "type": "hf_video", "device": device})
            
            model_data = self._model_cache[model_id]
            pipe = model_data["pipe"]
            
            prompt = params.get("prompt", "")
            num_frames = int(params.get("duration", 2.0) * params.get("fps", 8))
            
            output = pipe(prompt, num_frames=num_frames)
            frames = output.frames[0]
            
            # Save as GIF
            from ..config import CONFIG
            output_dir = Path(CONFIG.get("outputs_dir", "outputs")) / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            filepath = output_dir / f"hf_video_{timestamp}.gif"
            
            frames[0].save(str(filepath), format='GIF', save_all=True,
                          append_images=frames[1:], duration=125, loop=0)
            
            return {"success": True, "path": str(filepath), "model": model_id}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_hf_3d(self, assignment: ModelAssignment, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute 3D generation with HuggingFace Shap-E."""
        try:
            import time
            from pathlib import Path

            import torch
            from diffusers import ShapEPipeline
            
            model_id = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
            
            if model_id not in self._model_cache:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                
                logger.info(f"Loading HF 3D model: {model_id}")
                pipe = ShapEPipeline.from_pretrained(model_id, torch_dtype=dtype)
                pipe = pipe.to(device)
                self._cache_model(model_id, {"pipe": pipe, "type": "hf_3d", "device": device})
            
            model_data = self._model_cache[model_id]
            pipe = model_data["pipe"]
            
            prompt = params.get("prompt", "")
            guidance = float(params.get("guidance_scale", 15.0))
            
            output = pipe(prompt, guidance_scale=guidance)
            
            # Save
            from ..config import CONFIG
            output_dir = Path(CONFIG.get("outputs_dir", "outputs")) / "3d"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            filepath = output_dir / f"hf_3d_{timestamp}.ply"
            
            mesh = output.images[0]
            try:
                import trimesh
                if hasattr(mesh, 'export'):
                    mesh.export(str(filepath))
                else:
                    tmesh = trimesh.Trimesh(vertices=mesh.verts.cpu().numpy())
                    tmesh.export(str(filepath))
            except ImportError:
                import pickle
                filepath = output_dir / f"hf_3d_{timestamp}.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(mesh, f)
            
            return {"success": True, "path": str(filepath), "model": model_id}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_hf_vision(self, assignment: ModelAssignment, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vision/image understanding with HuggingFace model."""
        try:
            from PIL import Image
            from transformers import pipeline
            
            model_id = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
            
            if model_id not in self._model_cache:
                logger.info(f"Loading HF vision model: {model_id}")
                pipe = pipeline("image-to-text", model=model_id)
                self._cache_model(model_id, {"pipe": pipe, "type": "hf_vision"})
            
            data = self._model_cache[model_id]
            
            image_path = params.get("image_path", params.get("image", ""))
            question = params.get("prompt", params.get("question", "Describe this image."))
            
            if image_path:
                image = Image.open(image_path)
            else:
                return {"success": False, "error": "No image provided"}
            
            result = data["pipe"](image, question)
            text = result[0]["generated_text"] if isinstance(result, list) else str(result)
            
            return {"success": True, "result": text, "model": model_id}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_hf_embeddings(self, assignment: ModelAssignment, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute embeddings with HuggingFace sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_id = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
            
            if model_id not in self._model_cache:
                logger.info(f"Loading HF embeddings model: {model_id}")
                model = SentenceTransformer(model_id)
                self._cache_model(model_id, {"model": model, "type": "hf_embed"})
            
            data = self._model_cache[model_id]
            
            text = params.get("text", params.get("query", params.get("prompt", "")))
            if isinstance(text, str):
                text = [text]
            
            embeddings = data["model"].encode(text)
            
            return {"success": True, "embeddings": embeddings.tolist(), "model": model_id}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_api_call(self, assignment: ModelAssignment,
                         params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an API call."""
        provider = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
        
        return {
            "success": True,
            "result": {"action": "api_call", "provider": provider, "params": params},
            "model": assignment.model_id
        }
    
    # =========================================================================
    # AI-TO-AI NETWORKING - Distributed Task Execution
    # =========================================================================
    
    def _init_networking(self):
        """Initialize networking for AI collaboration."""
        try:
            from ..comms.ai_collaboration import get_collaboration_protocol
            self._collaboration_protocol = get_collaboration_protocol()
            logger.info("AI collaboration protocol initialized")
        except ImportError as e:
            logger.warning(f"AI collaboration not available: {e}")
            self._collaboration_protocol = None
    
    def connect_to_ai(self, url: str, name: str = None) -> bool:
        """
        Connect to another Enigma AI Engine instance for collaboration.
        
        📖 WHAT THIS DOES:
        Establishes a connection to another Enigma AI Engine instance running on the
        network. Once connected, you can delegate tasks to this peer.
        
        📐 EXAMPLE:
            router = get_router(enable_networking=True)
            router.connect_to_ai("192.168.1.100:5000", "desktop_pc")
            
            # Now "desktop_pc" is available for task delegation
        
        Args:
            url: URL of the remote Enigma AI Engine instance (e.g., "192.168.1.100:5000")
            name: Friendly name for this peer (auto-detected if not provided)
        
        Returns:
            True if connection successful
        """
        if not self._networking_enabled:
            logger.warning("Networking not enabled. Initialize with enable_networking=True")
            return False
        
        # Ensure network node exists
        if self._network_node is None:
            try:
                from ..comms.network import ForgeNode
                self._network_node = ForgeNode(name="tool_router_node")
                self._network_node.start_server(blocking=False)
            except Exception as e:
                logger.error(f"Failed to start network node: {e}")
                return False
        
        # Connect to peer
        success = self._network_node.connect_to(url, name)
        
        if success:
            peer_name = name or url.split(":")[0]
            self._remote_peers[peer_name] = {
                "url": url if url.startswith("http") else f"http://{url}",
                "connected_at": datetime.now().isoformat(),
                "capabilities": None,  # Fetched lazily
            }
            
            # Connect collaboration protocol
            if self._collaboration_protocol:
                self._collaboration_protocol.connect_to_network(self._network_node)
                self._collaboration_protocol.announce_capabilities()
            
            logger.info(f"Connected to AI peer: {peer_name} at {url}")
        
        return success
    
    def disconnect_from_ai(self, name: str):
        """
        Disconnect from a remote AI peer.
        
        Args:
            name: Name of the peer to disconnect from
        """
        if name in self._remote_peers:
            del self._remote_peers[name]
            if self._network_node and name in self._network_node.peers:
                del self._network_node.peers[name]
            logger.info(f"Disconnected from AI peer: {name}")
    
    def list_connected_ais(self) -> List[str]:
        """
        List all connected AI peers.
        
        Returns:
            List of peer names
        """
        return list(self._remote_peers.keys())
    
    def _can_handle_locally(self, tool_name: str) -> bool:
        """
        Check if this router can handle a tool locally.
        
        Args:
            tool_name: Name of the tool to check
        
        Returns:
            True if local execution is possible
        """
        # Check if we have local assignments for this tool
        assignments = self.get_assignments(tool_name)
        if not assignments:
            return False
        
        # Check if any assignment is actually available locally
        for assignment in assignments:
            model_type = assignment.model_type
            
            # Local modules are always available (they may fail, but we can try)
            if model_type == "local":
                return True
            
            # Check if Forge model is loaded or loadable
            if model_type == "enigma_engine":
                try:
                    # Just check if model exists, don't load it
                    from .model_registry import ModelRegistry
                    registry = ModelRegistry()
                    model_name = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
                    models = registry.list_models()
                    if model_name == "default" or model_name in models:
                        return True
                except Exception as e:
                    logger.debug(f"Could not check model registry for {model_type}: {e}")
        
        return False
    
    def _find_best_peer_for_task(self, tool_name: str) -> Optional[str]:
        """
        Find the most capable connected peer for a task.
        
        📖 WHAT THIS DOES:
        Queries all connected peers to find who can best handle
        the given tool, based on routing preference.
        
        Args:
            tool_name: Tool to find a handler for
        
        Returns:
            Name of best peer, or None if no peer available
        """
        if not self._collaboration_protocol:
            return None
        
        return self._collaboration_protocol.request_task_handling(tool_name, {})
    
    def _execute_remote(self, peer_name: str, tool_name: str, params: dict) -> dict:
        """
        Execute a tool on a remote AI peer.
        
        Args:
            peer_name: Name of the peer to execute on
            tool_name: Tool to execute
            params: Tool parameters
        
        Returns:
            Execution result dictionary
        """
        if not self._collaboration_protocol:
            return {"success": False, "error": "Collaboration protocol not initialized"}
        
        if peer_name not in self._remote_peers:
            return {"success": False, "error": f"Unknown peer: {peer_name}"}
        
        logger.info(f"Executing {tool_name} remotely on {peer_name}")
        
        try:
            result = self._collaboration_protocol.delegate_task(peer_name, tool_name, params)
            result["executed_by"] = peer_name
            result["remote"] = True
            return result
        except Exception as e:
            logger.error(f"Remote execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_peer_capabilities(self, peer_name: str) -> dict:
        """
        Get capabilities of a connected peer.
        
        Args:
            peer_name: Name of the peer
        
        Returns:
            Capability dictionary or empty dict if unavailable
        """
        if not self._collaboration_protocol:
            return {}
        
        capability = self._collaboration_protocol.get_peer_capabilities(peer_name)
        if capability:
            return capability.to_dict()
        return {}
    
    def route_intelligently(self, tool_name: str, params: dict) -> dict:
        """
        Intelligently route a task to the best available handler.
        
        📖 WHAT THIS DOES:
        Smart routing that considers local capability, peer capabilities,
        current load, and routing preference to find the best handler.
        
        📐 ROUTING FLOW:
        ┌─────────────────────────────────────────────────────────────────────┐
        │  1. Check local capability and load                                 │
        │     └─► If local can handle and not overloaded → Execute locally   │
        │                                                                     │
        │  2. Check routing preference                                        │
        │     ├─► local_first: Only use peers if local fails                 │
        │     ├─► fastest: Use lowest-latency option                         │
        │     ├─► quality_first: Use most capable option                     │
        │     └─► distributed: Balance load across all                       │
        │                                                                     │
        │  3. Find best peer if needed                                        │
        │     └─► Query peers, score by capability, pick best                │
        │                                                                     │
        │  4. Execute on chosen handler                                       │
        │                                                                     │
        │  5. Fallback chain: local → fastest_peer → capable_peer → cloud    │
        └─────────────────────────────────────────────────────────────────────┘
        
        Args:
            tool_name: Tool to execute
            params: Tool parameters
        
        Returns:
            Execution result dictionary
        """
        # Check local capability first
        can_local = self._can_handle_locally(tool_name)
        
        # If local_first preference or no networking, try local first
        if self._routing_preference == "local_first" or not self._networking_enabled:
            if can_local:
                result = self.execute_tool(tool_name, params)
                if result.get("success"):
                    result["routed_to"] = "local"
                    return result
            
            # Local failed, try peers
            if self._networking_enabled:
                best_peer = self._find_best_peer_for_task(tool_name)
                if best_peer:
                    result = self._execute_remote(best_peer, tool_name, params)
                    if result.get("success"):
                        return result
            
            # Everything failed
            return {"success": False, "error": f"No handler available for {tool_name}"}
        
        # For fastest/quality_first, evaluate all options
        if self._routing_preference in ("fastest", "quality_first"):
            candidates = []
            
            # Score local
            if can_local:
                local_score = 10 if self._routing_preference == "fastest" else 5
                candidates.append(("local", local_score))
            
            # Score peers
            if self._collaboration_protocol:
                all_caps = self._collaboration_protocol.list_all_capabilities()
                for peer_name, cap in all_caps.items():
                    from ..comms.ai_collaboration import RoutingPreference
                    pref = RoutingPreference.FASTEST if self._routing_preference == "fastest" else RoutingPreference.QUALITY_FIRST
                    score = cap.get_score(tool_name, pref)
                    if score > 0:
                        candidates.append((peer_name, score))
            
            # Sort by score
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Try in order
            for handler, score in candidates:
                if handler == "local":
                    result = self.execute_tool(tool_name, params)
                else:
                    result = self._execute_remote(handler, tool_name, params)
                
                if result.get("success"):
                    result["routed_to"] = handler
                    result["routing_score"] = score
                    return result
            
            return {"success": False, "error": f"All handlers failed for {tool_name}"}
        
        # Distributed: round-robin or load-based
        if self._routing_preference == "distributed":
            # Simple implementation: pick least loaded
            candidates = []
            
            if can_local:
                candidates.append(("local", 0.5))  # Assume 50% local load
            
            if self._collaboration_protocol:
                all_caps = self._collaboration_protocol.list_all_capabilities()
                for peer_name, cap in all_caps.items():
                    if cap.can_handle(tool_name):
                        candidates.append((peer_name, cap.current_load))
            
            # Sort by load (lowest first)
            candidates.sort(key=lambda x: x[1])
            
            for handler, load in candidates:
                if handler == "local":
                    result = self.execute_tool(tool_name, params)
                else:
                    result = self._execute_remote(handler, tool_name, params)
                
                if result.get("success"):
                    result["routed_to"] = handler
                    return result
            
            return {"success": False, "error": f"No handlers available for {tool_name}"}
        
        # Default: local execution
        return self.execute_tool(tool_name, params)
    
    def set_routing_preference(self, preference: str):
        """
        Set the routing preference for task distribution.
        
        Args:
            preference: One of "local_first", "fastest", "quality_first", "distributed"
        """
        valid = {"local_first", "fastest", "quality_first", "distributed"}
        if preference not in valid:
            logger.warning(f"Invalid routing preference: {preference}. Valid: {valid}")
            return
        
        self._routing_preference = preference
        
        if self._collaboration_protocol:
            self._collaboration_protocol.set_routing_preference(preference)
        
        logger.info(f"Routing preference set to: {preference}")
    
    def negotiate_task(self, tool_name: str, params: dict) -> str:
        """
        Ask connected AIs who can best handle a task.
        
        Args:
            tool_name: Tool to execute
            params: Task parameters
        
        Returns:
            Name of best handler ("local" or peer name)
        """
        if not self._collaboration_protocol:
            return "local"
        
        return self._collaboration_protocol.negotiate_task(tool_name, params)
    
    def broadcast_capability_update(self):
        """
        Inform all peers when local capabilities change.
        
        Call this after adding/removing tools or changing model assignments.
        """
        if self._collaboration_protocol:
            self._collaboration_protocol.announce_capabilities()
            logger.info("Broadcasted capability update to peers")
    
    def request_collaboration(self, task: str, sub_tasks: List[dict]) -> dict:
        """
        Split a complex task across multiple AIs.
        
        Args:
            task: Parent task description
            sub_tasks: List of subtask dictionaries
        
        Returns:
            Combined results from all collaborating AIs
        """
        if not self._collaboration_protocol:
            # Execute all locally
            results = []
            for subtask in sub_tasks:
                result = self.execute_tool(
                    subtask.get("tool_name", "chat"),
                    subtask.get("params", {})
                )
                results.append(result)
            
            return {
                "success": any(r.get("success") for r in results),
                "results": results,
                "distributed": False,
            }
        
        return self._collaboration_protocol.request_collaboration(task, sub_tasks)
    
    # =========================================================================
    # AUTO ROUTING
    # =========================================================================
    
    def auto_route(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Automatically route a user request to the appropriate tool.
        
        1. Detect which tool should handle this (using specialized router if available)
        2. If networking enabled, use intelligent routing
        3. Execute the tool
        4. Return result
        """
        # Detect tool using specialized model or keyword matching
        tool_name = self.classify_intent(user_input) if self.use_specialized else self.detect_tool(user_input)
        
        if not tool_name:
            tool_name = "chat"
        
        logger.info(f"Routing to tool: {tool_name}")
        
        # Build params
        params = {"prompt": user_input}
        if context:
            params.update(context)
        
        # Use intelligent routing if networking enabled
        if self._networking_enabled and self._remote_peers:
            return self.route_intelligently(tool_name, params)
            
        # Standard local execution
        return self.execute_tool(tool_name, params)
    
    def auto_route_with_context(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Route with conversation context for better tool selection.
        
        This is useful when:
        - Routing to specialized AIs that need context
        - Continuing conversations across different tools
        - Providing background for complex requests
        
        Args:
            user_input: Current user message
            conversation_history: List of previous messages
            include_summary: Whether to generate and include summary
            
        Returns:
            Tool execution result with context
        """
        context = {}
        
        if conversation_history and include_summary:
            try:
                from ..memory.conversation_summary import get_continuation_context
                context["conversation_context"] = get_continuation_context(
                    conversation_history,
                    max_tokens=300  # Keep context compact for routing
                )
            except Exception as e:
                logger.debug(f"Could not generate conversation context: {e}")
        
        # Include recent messages even without full summary
        if conversation_history and not context.get("conversation_context"):
            recent = conversation_history[-4:]
            context["recent_messages"] = [
                {"role": m.get("role", "user"), "text": m.get("text", "")[:200]}
                for m in recent
            ]
        
        return self.auto_route(user_input, context)
    
    # =========================================================================
    # SELF-TRAINING SYSTEM
    # =========================================================================
    
    def train_router_from_feedback(
        self,
        feedback_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Train the router model from routing feedback.
        
        📖 SELF-IMPROVEMENT:
        The router can learn from successful and failed routing decisions.
        When a user corrects a routing decision or a route succeeds,
        that feedback is used to improve future routing.
        
        Args:
            feedback_data: Optional list of feedback entries. If None, uses stored feedback.
            
        Returns:
            Training statistics
        """
        
        feedback_path = self.config_path.parent / "routing_feedback.json"
        
        # Load feedback data
        if feedback_data is None:
            if feedback_path.exists():
                try:
                    import json
                    feedback_data = json.loads(feedback_path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"Could not load feedback: {e}")
                    feedback_data = []
            else:
                return {"error": "No feedback data available"}
        
        if not feedback_data or len(feedback_data) < 10:
            return {"error": "Insufficient feedback data (need at least 10 examples)"}
        
        # Generate training data
        training_lines = []
        for entry in feedback_data:
            user_input = entry.get("input", "")
            correct_tool = entry.get("correct_tool", entry.get("tool"))
            success = entry.get("success", True)
            
            if user_input and correct_tool and success:
                # Format: Q: <input>\nA: [E:tool]<correct_tool>
                training_lines.append(f"Q: {user_input}\nA: [E:tool]{correct_tool}")
        
        if len(training_lines) < 10:
            return {"error": "Not enough successful examples for training"}
        
        # Save training data
        training_path = self.config_path.parent / "router_training_data.txt"
        training_path.write_text("\n\n".join(training_lines), encoding="utf-8")
        
        logger.info(f"Generated {len(training_lines)} training examples for router")
        
        # Attempt training if training module is available
        try:
            from .training import Trainer, TrainingConfig
            from .model import create_model
            
            # Create small router model if needed
            model = create_model("micro", vocab_size=10000)
            
            config = TrainingConfig(
                num_epochs=5,
                batch_size=4,
                learning_rate=0.0001,
                max_seq_len=128
            )
            
            trainer = Trainer(model, config)
            
            # This is a simplified training call
            # In practice, the user would use the full training tab
            stats = {
                "status": "training_data_ready",
                "training_file": str(training_path),
                "examples": len(training_lines),
                "message": "Training data generated. Use Training tab to train the router model."
            }
            
            return stats
            
        except Exception as e:
            logger.warning(f"Auto-training not available: {e}")
            return {
                "status": "training_data_ready",
                "training_file": str(training_path),
                "examples": len(training_lines),
                "message": f"Training data ready at {training_path}. Train manually in Training tab."
            }
    
    def record_routing_feedback(
        self,
        user_input: str,
        original_tool: str,
        correct_tool: Optional[str] = None,
        success: bool = True
    ):
        """
        Record feedback about a routing decision.
        
        📖 HOW THIS IMPROVES THE ROUTER:
        When routing succeeds or fails, call this to record the outcome.
        The router uses this feedback to improve future decisions.
        
        Args:
            user_input: The original user input
            original_tool: The tool the router selected
            correct_tool: The correct tool (if different from original)
            success: Whether the routing was successful
        """
        import json
        from datetime import datetime
        
        feedback_path = self.config_path.parent / "routing_feedback.json"
        
        # Load existing feedback
        feedback = []
        if feedback_path.exists():
            try:
                feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
            except Exception:
                feedback = []
        
        # Add new entry
        entry = {
            "input": user_input,
            "tool": original_tool,
            "correct_tool": correct_tool or original_tool,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        feedback.append(entry)
        
        # Keep last 1000 entries
        feedback = feedback[-1000:]
        
        # Save
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_path.write_text(json.dumps(feedback, indent=2), encoding="utf-8")
        
        logger.debug(f"Recorded routing feedback: {original_tool} -> {correct_tool or original_tool}")
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about routing performance.
        
        Returns:
            Dictionary with routing statistics
        """
        import json
        from collections import Counter
        
        feedback_path = self.config_path.parent / "routing_feedback.json"
        
        if not feedback_path.exists():
            return {"status": "no_data", "message": "No routing feedback recorded yet"}
        
        try:
            feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
        except Exception:
            return {"status": "error", "message": "Could not load feedback data"}
        
        if not feedback:
            return {"status": "no_data", "message": "Feedback file is empty"}
        
        # Calculate statistics
        total = len(feedback)
        successful = sum(1 for f in feedback if f.get("success", True))
        corrections = sum(1 for f in feedback if f.get("tool") != f.get("correct_tool"))
        
        tool_counts = Counter(f.get("correct_tool", f.get("tool")) for f in feedback)
        error_tools = Counter(
            f.get("tool") for f in feedback 
            if f.get("tool") != f.get("correct_tool")
        )
        
        return {
            "total_routes": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0,
            "corrections": corrections,
            "correction_rate": corrections / total if total > 0 else 0,
            "tool_usage": dict(tool_counts),
            "common_errors": dict(error_tools.most_common(5)),
            "ready_for_training": total >= 10
        }
    
    def train_sub_router(
        self,
        domain: str,
        training_data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train a sub-router for a specific domain.
        
        📖 DOMAIN-SPECIFIC ROUTING:
        For complex domains (games, code, creative), you can train
        specialized sub-routers that understand domain-specific intents.
        
        Args:
            domain: Domain name (e.g., "game", "code", "creative")
            training_data_path: Path to domain-specific training data
            
        Returns:
            Training status
        """
        valid_domains = ["game", "code", "creative", "productivity", "media"]
        
        if domain not in valid_domains:
            return {"error": f"Invalid domain. Must be one of: {valid_domains}"}
        
        # Generate domain-specific training data
        
        output_dir = self.config_path.parent / "sub_routers"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        domain_file = output_dir / f"{domain}_router_training.txt"
        
        # If no training data provided, generate from feedback
        if not training_data_path:
            import json
            feedback_path = self.config_path.parent / "routing_feedback.json"
            
            if feedback_path.exists():
                feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
                
                # Filter for domain-related entries
                domain_keywords = {
                    "game": ["game", "play", "level", "quest", "inventory", "enemy"],
                    "code": ["code", "program", "function", "debug", "script", "variable"],
                    "creative": ["draw", "paint", "create", "design", "art", "write"],
                    "productivity": ["schedule", "task", "email", "document", "organize"],
                    "media": ["video", "audio", "image", "music", "photo", "record"]
                }
                
                keywords = domain_keywords.get(domain, [])
                domain_feedback = [
                    f for f in feedback
                    if any(kw in f.get("input", "").lower() for kw in keywords)
                ]
                
                if len(domain_feedback) < 5:
                    return {
                        "error": f"Insufficient domain feedback for '{domain}'",
                        "found": len(domain_feedback),
                        "needed": 5
                    }
                
                # Generate training data
                lines = []
                for f in domain_feedback:
                    lines.append(f"Q: {f['input']}\nA: [E:tool]{f.get('correct_tool', f['tool'])}")
                
                domain_file.write_text("\n\n".join(lines), encoding="utf-8")
                
                return {
                    "status": "success",
                    "domain": domain,
                    "training_file": str(domain_file),
                    "examples": len(lines),
                    "message": f"Sub-router training data generated for '{domain}'"
                }
        
        return {
            "status": "not_implemented",
            "message": "Custom training data path not yet implemented"
        }


# Singleton instance
_router_instance: Optional[ToolRouter] = None


def get_router(use_specialized: bool = False, enable_networking: bool = False) -> ToolRouter:
    """
    Get the global ToolRouter instance.
    
    Args:
        use_specialized: Enable specialized models for routing
        enable_networking: Enable AI-to-AI collaboration
    
    Returns:
        ToolRouter instance
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = ToolRouter(
            use_specialized=use_specialized,
            enable_networking=enable_networking
        )
    return _router_instance
