"""
Tool Router - Connect AI models to tools and dispatch tool calls.

This system allows:
  - Assigning any model (Forge or HuggingFace) to any tool
  - Main AI automatically calling tools based on user requests
  - Multiple models assigned to same tool (fallback chain)
  - Tool execution and result handling

USAGE:
    from forge_ai.core.tool_router import ToolRouter, get_router
    
    router = get_router()
    
    # Assign models to tools
    router.assign_model("chat", "forge:small_forge_ai")
    router.assign_model("chat", "huggingface:mistralai/Mistral-7B-Instruct-v0.2")
    router.assign_model("image", "local:stable-diffusion")
    
    # Execute tool
    result = router.execute_tool("image", {"prompt": "a sunset"})
    
    # Let AI decide which tool to use
    result = router.auto_route("Draw me a cat")
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolDefinition:
    """Definition of an available tool."""
    name: str
    description: str
    keywords: List[str]  # Keywords that trigger this tool
    parameters: Dict[str, str]  # param_name -> description
    handler: Optional[Callable] = None
    

@dataclass 
class ModelAssignment:
    """A model assigned to a tool."""
    model_id: str  # "forge:name", "huggingface:repo/model", "local:name"
    model_type: str  # "forge_ai", "huggingface", "local", "api"
    priority: int = 0  # Higher = try first
    config: Dict[str, Any] = field(default_factory=dict)


# Built-in tool definitions
TOOL_DEFINITIONS = {
    "chat": ToolDefinition(
        name="chat",
        description="General conversation and reasoning",
        keywords=["chat", "talk", "explain", "help", "what", "why", "how"],
        parameters={"prompt": "The user message to respond to"}
    ),
    "image": ToolDefinition(
        name="image",
        description="Generate images from text descriptions",
        keywords=["draw", "paint", "create image", "generate image", "picture", "photo", "illustration", "artwork"],
        parameters={"prompt": "Description of image to generate", "width": "Image width", "height": "Image height"}
    ),
    "code": ToolDefinition(
        name="code",
        description="Generate or analyze code",
        keywords=["code", "program", "script", "function", "debug", "fix code", "write code"],
        parameters={"prompt": "Code task description", "language": "Programming language"}
    ),
    "video": ToolDefinition(
        name="video",
        description="Generate video clips",
        keywords=["video", "animate", "animation", "clip", "movie"],
        parameters={"prompt": "Video description", "duration": "Length in seconds"}
    ),
    "audio": ToolDefinition(
        name="audio",
        description="Generate audio or speech",
        keywords=["speak", "say", "voice", "audio", "sound", "music", "song", "read aloud"],
        parameters={"text": "Text to speak or audio description"}
    ),
    "3d": ToolDefinition(
        name="3d",
        description="Generate 3D models",
        keywords=["3d", "model", "mesh", "object", "sculpt"],
        parameters={"prompt": "3D model description"}
    ),
    "gif": ToolDefinition(
        name="gif",
        description="Create animated GIFs",
        keywords=["gif", "animated", "animation loop", "meme"],
        parameters={"prompt": "GIF description", "frames": "Number of frames"}
    ),
    "web": ToolDefinition(
        name="web",
        description="Search the web or fetch information",
        keywords=["search", "google", "look up", "find", "website", "browse"],
        parameters={"query": "Search query or URL"}
    ),
    "memory": ToolDefinition(
        name="memory",
        description="Remember or recall information",
        keywords=["remember", "recall", "forget", "memory", "save this"],
        parameters={"action": "save/recall/list", "content": "What to remember"}
    ),
    "embeddings": ToolDefinition(
        name="embeddings",
        description="Semantic search and similarity matching",
        keywords=["search", "find similar", "semantic", "embedding", "vector search"],
        parameters={"query": "Text to search for", "top_k": "Number of results"}
    ),
    "camera": ToolDefinition(
        name="camera",
        description="Capture from webcam or camera",
        keywords=["camera", "webcam", "photo", "capture", "take picture", "snap"],
        parameters={"action": "capture/record/analyze"}
    ),
    "vision": ToolDefinition(
        name="vision",
        description="Analyze images or screen captures",
        keywords=["see", "look at", "analyze image", "what's on screen", "screenshot", "describe image"],
        parameters={"image_path": "Path to image or 'screen' for screenshot"}
    ),
    "avatar": ToolDefinition(
        name="avatar",
        description="Control the AI avatar display",
        keywords=["avatar", "face", "expression", "show me", "look"],
        parameters={"expression": "Avatar expression to show"}
    ),
}


class ToolRouter:
    """Routes requests to appropriate tools and models."""
    
    # Maximum number of models to keep in cache (LRU eviction)
    MAX_CACHE_SIZE = 5
    MAX_SPECIALIZED_CACHE_SIZE = 3
    
    def __init__(self, config_path: Optional[Path] = None, use_specialized: bool = False):
        from ..config import CONFIG
        
        self.config_path = config_path or Path(CONFIG.get("data_dir", "data")) / "tool_routing.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Tool -> List of assigned models
        self.assignments: Dict[str, List[ModelAssignment]] = {}
        
        # Tool definitions
        self.tools = TOOL_DEFINITIONS.copy()
        
        # Loaded model instances (cached with LRU tracking)
        self._model_cache: Dict[str, Any] = {}
        self._model_cache_order: List[str] = []  # Track access order for LRU
        
        # Tool handlers
        self._handlers: Dict[str, Callable] = {}
        
        # Specialized models configuration
        self.use_specialized = use_specialized
        self._specialized_models: Dict[str, Any] = {}
        self._specialized_cache_order: List[str] = []
        self._shared_tokenizer = None
        
        self._load_config()
        
        # Load specialized models config if enabled
        if use_specialized:
            self._load_specialized_config()
    
    def _cache_model(self, key: str, model: Any, is_specialized: bool = False):
        """Add model to cache with LRU eviction."""
        if is_specialized:
            cache = self._specialized_models
            order = self._specialized_cache_order
            max_size = self.MAX_SPECIALIZED_CACHE_SIZE
        else:
            cache = self._model_cache
            order = self._model_cache_order
            max_size = self.MAX_CACHE_SIZE
        
        # Update access order
        if key in order:
            order.remove(key)
        order.append(key)
        
        cache[key] = model
        
        # Evict oldest if over limit
        while len(order) > max_size:
            oldest = order.pop(0)
            if oldest in cache:
                old_model = cache.pop(oldest)
                self._cleanup_model(old_model)
                logger.info(f"Evicted model from cache: {oldest}")
    
    def _cleanup_model(self, model: Any):
        """Clean up a model (free GPU memory)."""
        try:
            if model is None:
                return
            
            # Handle specialized model dict format
            if isinstance(model, dict) and 'model' in model:
                actual_model = model['model']
                del model['model']
                model.clear()
                model = actual_model
            
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
            
            # Free GPU memory
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
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                
                for tool_name, models in data.get("assignments", {}).items():
                    self.assignments[tool_name] = [
                        ModelAssignment(**m) for m in models
                    ]
            except Exception as e:
                print(f"Error loading tool routing config: {e}")
                self.assignments = {}
        else:
            # Default assignments
            self._set_defaults()
            
    def _set_defaults(self):
        """Set default model assignments."""
        self.assignments = {
            "chat": [ModelAssignment("forge:default", "forge_ai", priority=10)],
            "image": [ModelAssignment("local:stable-diffusion", "local", priority=10)],
            "code": [ModelAssignment("forge:default", "forge_ai", priority=10)],
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
            
            with open(config_path, 'r') as f:
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
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
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
        Classify user intent using specialized router model.
        
        Args:
            text: User input text
        
        Returns:
            Intent class (chat, vision, image, code, etc.)
        """
        # Try specialized router model first
        if self.use_specialized:
            router_model = self._specialized_models.get('router')
            if router_model is None and 'router' not in self._specialized_models:
                # Try to load it (only once)
                router_model = self._load_specialized_model('router')
                self._specialized_models['router'] = router_model
            
            if router_model:
                try:
                    import torch
                    
                    model = router_model['model']
                    tokenizer = router_model['tokenizer']
                    
                    # Prepare input
                    prompt = f"Q: {text}\nA: [E:tool]"
                    input_ids = tokenizer.encode(prompt)
                    input_tensor = torch.tensor([input_ids])
                    
                    # Generate
                    with torch.no_grad():
                        output = model.generate(
                            input_tensor,
                            max_new_tokens=10,
                            temperature=0.1,  # Low temperature for classification
                        )
                    
                    # Decode and extract intent
                    result = tokenizer.decode(output[0].tolist())
                    
                    # Extract intent from output using regex for robustness
                    # Expected format: "Q: ... A: [E:tool]intent"
                    import re
                    match = re.search(r'\[E:tool\](\w+)', result)
                    if match:
                        intent = match.group(1).lower()
                        if intent in self.tools:
                            logger.info(f"Router classified intent: {intent}")
                            return intent
                    
                    # Fallback: try simple split if regex fails
                    if '[E:tool]' in result:
                        parts = result.split('[E:tool]')
                        if len(parts) > 1:
                            # Get first word after [E:tool]
                            intent_text = parts[-1].strip()
                            # Extract first word (handle whitespace, newlines, etc.)
                            intent = intent_text.split()[0].lower() if intent_text else None
                            if intent and intent in self.tools:
                                logger.info(f"Router classified intent (fallback): {intent}")
                                return intent
                    
                except Exception as e:
                    logger.warning(f"Specialized router failed: {e}")
        
        # Fallback to keyword-based detection
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
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}")
        
        # Parse model type from ID
        if ":" in model_id:
            model_type = model_id.split(":")[0]
        else:
            model_type = "forge_ai"  # Default
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
        for tool_name, tool_def in self.tools.items():
            score = 0
            for keyword in tool_def.keywords:
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
            
        model_type, model_name = model_id.split(":", 1) if ":" in model_id else ("forge_ai", model_id)
        
        model = None
        
        if model_type == "forge_ai":
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
        """Load a ForgeAI model."""
        try:
            from .model_registry import ModelRegistry
            registry = ModelRegistry()
            
            if name == "default":
                # Get first available model - list_models returns Dict[str, Any]
                models_dict = registry.list_models()
                if models_dict:
                    # Get first key (model name)
                    name = next(iter(models_dict.keys()), "small_forge_ai")
                else:
                    return None
                    
            model, config = registry.load_model(name)
            return {"model": model, "config": config, "type": "forge_ai"}
        except Exception as e:
            print(f"Failed to load Forge model {name}: {e}")
            return None
            
    def _load_huggingface_model(self, repo_id: str) -> Any:
        """Load a HuggingFace model for text generation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
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
            print(f"Failed to load HuggingFace model {repo_id}: {e}")
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
        if tool_name not in self.tools:
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
                print(f"Model {assignment.model_id} failed: {e}")
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
        elif model_type == "forge_ai":
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
        
        if model_data["type"] == "forge_ai":
            # Use Forge inference
            try:
                from .inference import ForgeEngine
                engine = ForgeEngine(model_data["model"], model_data["config"])
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
        """Execute code generation using local Forge model."""
        try:
            from ..gui.tabs.code_tab import get_provider
            
            provider = get_provider('local')
            if provider is None:
                return {"success": False, "error": "Code provider not available"}
            
            if not provider.is_loaded:
                if not provider.load():
                    return {"success": False, "error": "Failed to load code model"}
            
            prompt = params.get("prompt", "")
            language = params.get("language", "python")
            
            result = provider.generate(prompt, language=language)
            return result
            
        except ImportError as e:
            return {"success": False, "error": f"Code generation not available: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_audio_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute audio/TTS generation using local provider."""
        try:
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
            from diffusers import StableDiffusionPipeline, DiffusionPipeline
            import torch
            from pathlib import Path
            import time
            
            model_id = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
            
            # Check cache
            if model_id not in self._model_cache:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"Loading HF image model: {model_id}")
                pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    safety_checker=None,
                )
                pipe = pipe.to(device)
                
                if device == "cuda":
                    try:
                        pipe.enable_attention_slicing()
                    except Exception:
                        pass
                
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
            import torch
            from pathlib import Path
            import time
            import scipy.io.wavfile as wavfile
            
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
            from diffusers import DiffusionPipeline
            import torch
            from pathlib import Path
            import time
            
            model_id = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
            
            if model_id not in self._model_cache:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"Loading HF video model: {model_id}")
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
            from diffusers import ShapEPipeline
            import torch
            from pathlib import Path
            import time
            
            model_id = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
            
            if model_id not in self._model_cache:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                
                print(f"Loading HF 3D model: {model_id}")
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
            from transformers import pipeline
            from PIL import Image
            
            model_id = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
            
            if model_id not in self._model_cache:
                print(f"Loading HF vision model: {model_id}")
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
                print(f"Loading HF embeddings model: {model_id}")
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
    
    def auto_route(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Automatically route a user request to the appropriate tool.
        
        1. Detect which tool should handle this (using specialized router if available)
        2. Execute the tool
        3. Return result
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
            
        # Execute
        return self.execute_tool(tool_name, params)


# Singleton instance
_router_instance: Optional[ToolRouter] = None


def get_router(use_specialized: bool = False) -> ToolRouter:
    """
    Get the global ToolRouter instance.
    
    Args:
        use_specialized: Enable specialized models for routing
    
    Returns:
        ToolRouter instance
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = ToolRouter(use_specialized=use_specialized)
    return _router_instance
