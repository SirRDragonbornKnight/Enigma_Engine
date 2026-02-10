"""
AI Integration Helper - Connect AI models to all Enigma AI Engine features.

This module helps:
1. Check which features have AI models assigned
2. Setup AI for missing features
3. Test AI connectivity
4. Generate training data for specific tools

Example:
    from enigma_engine.core.ai_integration import AIIntegration
    
    ai = AIIntegration()
    
    # Check status
    status = ai.get_integration_status()
    
    # Setup chat
    ai.setup_feature("chat", model_id="huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Test all features
    results = ai.test_all()
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class FeatureStatus:
    """Status of an AI feature."""
    name: str
    has_model: bool
    model_id: Optional[str] = None
    model_type: Optional[str] = None  # "chat", "vision", "code", etc.
    is_working: bool = False
    error: Optional[str] = None
    recommended_models: list[str] = field(default_factory=list)


# Feature requirements - what type of model each feature needs
FEATURE_REQUIREMENTS = {
    # Core chat - needs text generation model
    "chat": {
        "type": "text",
        "min_params": "500M",  # Minimum recommended
        "description": "General conversation",
        "recommended": [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Qwen/Qwen2-1.5B-Instruct",
            "microsoft/phi-2",
            "mistralai/Mistral-7B-Instruct-v0.2",
        ],
        "training_file": "training.txt",
    },
    
    # Vision - needs vision-language model
    "vision": {
        "type": "vision",
        "description": "Image/screen analysis",
        "recommended": [
            "Qwen/Qwen2-VL-2B-Instruct",
            "llava-hf/llava-1.5-7b-hf",
            "microsoft/Florence-2-large",
        ],
        "note": "Cannot use regular text models - must be vision-language",
    },
    
    # Code generation - needs code-trained model
    "code": {
        "type": "code",
        "description": "Code generation and debugging",
        "recommended": [
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "codellama/CodeLlama-7b-Instruct-hf",
        ],
        "training_file": "tool_training_data.txt",
    },
    
    # Image generation - uses Stable Diffusion (not LLM)
    "image": {
        "type": "diffusion",
        "description": "Generate images from text",
        "recommended": [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ],
        "note": "Uses diffusion model, not LLM",
    },
    
    # Audio/TTS - separate system
    "audio": {
        "type": "tts",
        "description": "Text-to-speech",
        "recommended": [
            "pyttsx3 (local)",
            "ElevenLabs (API)",
        ],
        "note": "Uses TTS engine, not LLM",
    },
    
    # Embeddings - for semantic search
    "embeddings": {
        "type": "embedding",
        "description": "Semantic search and similarity",
        "recommended": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
        ],
    },
    
    # Tool use - main chat model trained on tools
    "tools": {
        "type": "text",
        "description": "AI calling tools (file, web, etc.)",
        "note": "Requires chat model trained on tool_training_data.txt",
        "training_file": "tool_training_data.txt",
    },
    
    # Avatar - expression control
    "avatar": {
        "type": "classifier",
        "description": "Emotion/expression detection for avatar",
        "note": "Can use simple classifier or chat model",
    },
}


class AIIntegration:
    """Helper for AI integration across Enigma AI Engine."""
    
    def __init__(self):
        self._router = None
        self._model_registry = None
    
    @property
    def router(self):
        """Get tool router (lazy load)."""
        if self._router is None:
            try:
                from .tool_router import get_router
                self._router = get_router()
            except Exception as e:
                logger.error(f"Could not load tool router: {e}")
        return self._router
    
    @property
    def model_registry(self):
        """Get model registry (lazy load)."""
        if self._model_registry is None:
            try:
                from .model_registry import ModelRegistry
                self._model_registry = ModelRegistry()
            except Exception as e:
                logger.error(f"Could not load model registry: {e}")
        return self._model_registry
    
    def get_integration_status(self) -> dict[str, FeatureStatus]:
        """
        Get status of all features - which have models, which don't.
        
        Returns:
            Dict mapping feature name to FeatureStatus
        """
        status = {}
        
        for feature, req in FEATURE_REQUIREMENTS.items():
            fs = FeatureStatus(
                name=feature,
                has_model=False,
                recommended_models=req.get("recommended", []),
            )
            
            # Check if model is assigned
            if self.router:
                models = self.router.get_assignments(feature)
                if models:
                    fs.has_model = True
                    fs.model_id = models[0].model_id
                    fs.model_type = models[0].model_type
            
            status[feature] = fs
        
        return status
    
    def print_status(self):
        """Print integration status nicely."""
        status = self.get_integration_status()
        
        print("\n" + "=" * 60)
        print("Enigma AI Engine - AI Integration Status")
        print("=" * 60)
        
        working = []
        missing = []
        
        for name, fs in status.items():
            if fs.has_model:
                working.append(f"✅ {name}: {fs.model_id}")
            else:
                req = FEATURE_REQUIREMENTS[name]
                missing.append(f"❌ {name}: {req['description']}")
        
        print("\n🟢 Working Features:")
        if working:
            for w in working:
                print(f"   {w}")
        else:
            print("   None - no models assigned yet!")
        
        print("\n🔴 Missing AI Models:")
        if missing:
            for m in missing:
                print(f"   {m}")
        else:
            print("   All features have models!")
        
        print("\n" + "-" * 60)
        print("To setup: Use Model Manager in GUI or call setup_feature()")
        print("=" * 60)
    
    def setup_feature(self, feature: str, model_id: str, priority: int = 0) -> bool:
        """
        Assign a model to a feature.
        
        Args:
            feature: Feature name (chat, vision, code, etc.)
            model_id: Model identifier (e.g., "huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            priority: Priority (higher = try first)
            
        Returns:
            True if successful
        """
        if feature not in FEATURE_REQUIREMENTS:
            logger.error(f"Unknown feature: {feature}")
            return False
        
        if not self.router:
            logger.error("Tool router not available")
            return False
        
        # Add prefix if needed
        if not model_id.startswith(("forge:", "huggingface:", "local:", "api:")):
            model_id = f"huggingface:{model_id}"
        
        try:
            self.router.assign_model(feature, model_id, priority=priority)
            logger.info(f"Assigned {model_id} to {feature}")
            return True
        except Exception as e:
            logger.error(f"Failed to assign model: {e}")
            return False
    
    def setup_basic_chat(self, model_size: str = "small") -> bool:
        """
        Quick setup for basic chat functionality.
        
        Args:
            model_size: "tiny" (1.1B), "small" (1.5B), "medium" (7B)
            
        Returns:
            True if successful
        """
        models_by_size = {
            "tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "small": "Qwen/Qwen2-1.5B-Instruct",
            "medium": "mistralai/Mistral-7B-Instruct-v0.2",
            "large": "meta-llama/Llama-2-13b-chat-hf",
        }
        
        model = models_by_size.get(model_size, models_by_size["small"])
        return self.setup_feature("chat", f"huggingface:{model}")
    
    def setup_vision(self, model_size: str = "small") -> bool:
        """
        Quick setup for vision functionality.
        
        Args:
            model_size: "small" (2B), "medium" (7B)
            
        Returns:
            True if successful
        """
        models_by_size = {
            "small": "Qwen/Qwen2-VL-2B-Instruct",
            "medium": "llava-hf/llava-1.5-7b-hf",
        }
        
        model = models_by_size.get(model_size, models_by_size["small"])
        return self.setup_feature("vision", f"huggingface:{model}")
    
    def test_feature(self, feature: str) -> dict[str, Any]:
        """
        Test if a feature is working.
        
        Args:
            feature: Feature to test
            
        Returns:
            Test result dict
        """
        result = {
            "feature": feature,
            "success": False,
            "message": "",
            "output": None,
        }
        
        if not self.router:
            result["message"] = "Tool router not available"
            return result
        
        # Test prompts for each feature
        test_prompts = {
            "chat": "Say hello in one sentence.",
            "vision": "Describe what you see.",  # Would need image
            "code": "Write a hello world in Python.",
            "tools": "What time is it?",  # Triggers tool use
        }
        
        prompt = test_prompts.get(feature, "Hello")
        
        try:
            output = self.router.execute_tool(feature, {"prompt": prompt})
            result["success"] = True
            result["output"] = output
            result["message"] = "Feature working!"
        except Exception as e:
            result["message"] = f"Error: {e}"
        
        return result
    
    def get_training_tips(self, feature: str) -> str:
        """
        Get tips for training AI for a specific feature.
        
        Args:
            feature: Feature name
            
        Returns:
            Training tips string
        """
        req = FEATURE_REQUIREMENTS.get(feature)
        if not req:
            return f"Unknown feature: {feature}"
        
        tips = [f"Training Tips for {feature.upper()}:", ""]
        
        # Feature-specific tips
        if feature == "chat":
            tips.extend([
                "✦ Use data/training.txt for personality and knowledge",
                "✦ Format: Q: question / A: answer",
                "✦ Include diverse topics and conversation styles",
                "✦ Minimum 500-1000 Q/A pairs recommended",
                "✦ Add your own Q&A to customize personality",
            ])
        
        elif feature == "tools":
            tips.extend([
                "✦ Use data/tool_training_data.txt",
                "✦ Format: Q: / A: with <tool_call> and <tool_result> tags",
                "✦ Example:",
                "  Q: Search for cats",
                "  A: I'll search for that.",
                "  <tool_call>{\"tool\": \"web_search\", \"params\": {\"query\": \"cats\"}}</tool_call>",
                "  <tool_result>{\"success\": true, \"results\": [...]}</tool_result>",
                "  I found some information about cats!",
                "",
                "✦ Train AI to recognize WHEN to use tools",
                "✦ Include examples of NOT using tools for simple questions",
            ])
        
        elif feature == "vision":
            tips.extend([
                "✦ Cannot train regular models for vision",
                "✦ Must use vision-language models (Qwen2-VL, LLaVA)",
                "✦ These have built-in image understanding",
                "✦ Fine-tuning vision models requires image datasets",
            ])
        
        elif feature == "code":
            tips.extend([
                "✦ Use code-specific training data",
                "✦ Include: code generation, debugging, explanation",
                "✦ Format: Q: task / A: code with explanation",
                "✦ Or use pre-trained code models (DeepSeek, CodeLlama)",
            ])
        
        # Common tips
        tips.extend([
            "",
            "Common Issues:",
            "• Model outputs raw tensors → Model not fully loaded",
            "• Repetition → Lower temperature, adjust repetition penalty",
            "• Ignores tools → Need more tool training examples",
            "• Wrong tool → Add disambiguation examples",
        ])
        
        return "\n".join(tips)
    
    def generate_tool_training_examples(self, tool_name: str, num_examples: int = 10) -> str:
        """
        Generate training examples for a specific tool.
        
        Args:
            tool_name: Name of tool to generate examples for
            num_examples: Number of examples to generate
            
        Returns:
            Training data string
        """
        from ..tools.tool_definitions import get_all_tools
        
        tools = get_all_tools()
        tool = None
        for t in tools:
            if t.name == tool_name:
                tool = t
                break
        
        if not tool:
            return f"# Tool not found: {tool_name}"
        
        examples = [f"# Training examples for {tool_name}", f"# {tool.description}", ""]
        
        # Generate example prompts based on tool
        prompts = {
            "generate_image": [
                "Draw a sunset over mountains",
                "Create an image of a cute robot",
                "Make me a picture of a forest",
            ],
            "read_file": [
                "Read the README file",
                "What's in config.json?",
                "Show me the contents of main.py",
            ],
            "web_search": [
                "Search for Python tutorials",
                "Look up the weather",
                "Find information about AI",
            ],
            "speak_text": [
                "Say hello out loud",
                "Read this message to me",
                "Speak: Welcome to Enigma AI Engine",
            ],
        }
        
        tool_prompts = prompts.get(tool_name, [
            f"Use {tool_name}",
            f"Run the {tool_name} tool",
            f"Execute {tool_name}",
        ])
        
        for i, prompt in enumerate(tool_prompts[:num_examples]):
            params = {}
            for p in tool.parameters:
                if p.required:
                    params[p.name] = f"<{p.name}>"
            
            examples.append(f"Q: {prompt}")
            examples.append(f"A: I'll do that for you.")
            examples.append(f'<tool_call>{{"tool": "{tool_name}", "params": {params}}}</tool_call>')
            examples.append(f'<tool_result>{{"tool": "{tool_name}", "success": true}}</tool_result>')
            examples.append(f"Done!")
            examples.append("")
        
        return "\n".join(examples)


def get_integration() -> AIIntegration:
    """Get the AI integration helper."""
    return AIIntegration()


# Quick access functions
def print_integration_status():
    """Print AI integration status."""
    get_integration().print_status()


def setup_chat(model_size: str = "small") -> bool:
    """Quick setup for chat."""
    return get_integration().setup_basic_chat(model_size)


def setup_vision(model_size: str = "small") -> bool:
    """Quick setup for vision."""
    return get_integration().setup_vision(model_size)


if __name__ == "__main__":
    # Print status when run directly
    print_integration_status()
