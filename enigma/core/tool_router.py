"""
Tool Router - Connect AI models to tools and dispatch tool calls.

This system allows:
  - Assigning any model (Enigma or HuggingFace) to any tool
  - Main AI automatically calling tools based on user requests
  - Multiple models assigned to same tool (fallback chain)
  - Tool execution and result handling

USAGE:
    from enigma.core.tool_router import ToolRouter, get_router
    
    router = get_router()
    
    # Assign models to tools
    router.assign_model("chat", "enigma:small_enigma")
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
    model_id: str  # "enigma:name", "huggingface:repo/model", "local:name"
    model_type: str  # "enigma", "huggingface", "local", "api"
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
            "chat": [ModelAssignment("enigma:default", "enigma", priority=10)],
            "image": [ModelAssignment("local:stable-diffusion", "local", priority=10)],
            "code": [ModelAssignment("enigma:default", "enigma", priority=10)],
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
            from .model import Enigma
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Get tokenizer
            tokenizer = self._get_shared_tokenizer()
            if tokenizer is None:
                logger.error("Cannot load specialized model without tokenizer")
                return None
            
            # Create model from config
            model = Enigma.from_config(checkpoint['config'])
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
                - "enigma:model_name" - Enigma model from registry
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
            model_type = "enigma"  # Default
            model_id = f"enigma:{model_id}"
            
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
            - model_type: Type (enigma, huggingface, api, local)
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
            
        model_type, model_name = model_id.split(":", 1) if ":" in model_id else ("enigma", model_id)
        
        model = None
        
        if model_type == "enigma":
            model = self._load_enigma_model(model_name)
        elif model_type == "huggingface":
            model = self._load_huggingface_model(model_name)
        elif model_type == "local":
            model = self._load_local_module(model_name)
        elif model_type == "api":
            model = self._load_api_provider(model_name)
            
        if model:
            self._model_cache[model_id] = model
            
        return model
    
    def _load_enigma_model(self, name: str) -> Any:
        """Load an Enigma model."""
        try:
            from .model_registry import ModelRegistry
            registry = ModelRegistry()
            
            if name == "default":
                # Get first available model - list_models returns Dict[str, Any]
                models_dict = registry.list_models()
                if models_dict:
                    # Get first key (model name)
                    name = next(iter(models_dict.keys()), "small_enigma")
                else:
                    return None
                    
            model, config = registry.load_model(name)
            return {"model": model, "config": config, "type": "enigma"}
        except Exception as e:
            print(f"Failed to load Enigma model {name}: {e}")
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
        
        if model_type in ("enigma", "huggingface"):
            # Text generation
            return self._execute_text_generation(assignment, params)
        elif model_type == "local":
            # Local module
            return self._execute_local_module(assignment, params)
        elif model_type == "api":
            # API call
            return self._execute_api_call(assignment, params)
            
        return {"success": False, "error": f"Unknown model type: {model_type}"}
    
    def _execute_text_generation(self, assignment: ModelAssignment, 
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text generation with Enigma or HuggingFace model."""
        model_data = self.load_model(assignment.model_id)
        if not model_data:
            return {"success": False, "error": "Failed to load model"}
            
        prompt = params.get("prompt", "")
        max_tokens = params.get("max_tokens", 256)
        
        if model_data["type"] == "enigma":
            # Use Enigma inference
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
        """Execute a local module (stable-diffusion, etc.)."""
        module_name = assignment.model_id.split(":", 1)[1] if ":" in assignment.model_id else assignment.model_id
        
        # Map module names to actual handlers
        # This integrates with the existing module system
        try:
            # Try to get the module from module manager
            # The actual execution happens in the GUI tabs
            return {
                "success": True, 
                "result": {"action": "execute_module", "module": module_name, "params": params},
                "model": assignment.model_id
            }
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
