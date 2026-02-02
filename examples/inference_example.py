#!/usr/bin/env python3
"""
ForgeAI Inference Example
=========================

Complete example showing how to run model inference including:
- Loading trained models
- Text generation
- Tool routing (specialized models)
- Batch inference
- Streaming generation
- Configuration options

This is the core of ForgeAI - generating AI responses from trained models.

Dependencies:
    pip install torch  # PyTorch

Run: python examples/inference_example.py
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, field


# =============================================================================
# Inference Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """Inference configuration."""
    # Generation parameters
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Model settings
    model_path: str = "models/forge-small"
    device: str = "auto"  # auto, cuda, cpu
    
    # Advanced
    use_cache: bool = True
    batch_size: int = 1
    use_flash_attention: bool = True


# =============================================================================
# Simulated Inference Engine
# =============================================================================

class ForgeEngine:
    """
    ForgeAI inference engine.
    
    Handles model loading, text generation, and routing to
    specialized models when needed.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Specialized model routing
        self.use_routing = False
        self.router = None
        self.specialized_models: Dict[str, Any] = {}
    
    def _log(self, message: str):
        print(f"[ForgeEngine] {message}")
    
    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model for inference.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            True if loaded successfully
        """
        model_path = model_path or self.config.model_path
        self._log(f"Loading model from: {model_path}")
        
        try:
            # In real implementation:
            # from forge_ai.core.model import Forge
            # from forge_ai.core.tokenizer import get_tokenizer
            # 
            # self.model = Forge.from_pretrained(model_path)
            # self.tokenizer = get_tokenizer()
            
            # Simulated
            self.model = {"name": "forge-small", "params": "27M", "loaded": True}
            self.tokenizer = {"vocab_size": 50257}
            
            self.is_loaded = True
            self._log(f"Model loaded successfully")
            return True
            
        except Exception as e:
            self._log(f"Error loading model: {e}")
            return False
    
    def unload(self):
        """Unload model to free memory."""
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self._log("Model unloaded")
    
    def generate(self, prompt: str, 
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self.is_loaded:
            return "[Error: Model not loaded]"
        
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        self._log(f"Generating (max_tokens={max_tokens}, temp={temperature})")
        
        # In real implementation:
        # inputs = self.tokenizer.encode(prompt)
        # outputs = self.model.generate(
        #     inputs,
        #     max_new_tokens=max_tokens,
        #     temperature=temperature,
        #     top_p=self.config.top_p,
        #     top_k=self.config.top_k
        # )
        # return self.tokenizer.decode(outputs)
        
        # Simulated response
        return f"This is a simulated response to: '{prompt[:30]}...' [Generated with temp={temperature}]"
    
    def generate_stream(self, prompt: str,
                        max_tokens: Optional[int] = None,
                        **kwargs) -> Generator[str, None, None]:
        """
        Generate text with streaming (token by token).
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            
        Yields:
            Generated tokens one at a time
        """
        if not self.is_loaded:
            yield "[Error: Model not loaded]"
            return
        
        max_tokens = max_tokens or self.config.max_tokens
        self._log(f"Streaming generation (max_tokens={max_tokens})")
        
        # Simulated streaming
        response = f"This is a streaming response to: '{prompt[:30]}...'"
        words = response.split()
        
        for word in words:
            yield word + " "
            time.sleep(0.05)  # Simulate generation time
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of generated responses
        """
        self._log(f"Batch generation ({len(prompts)} prompts)")
        
        # In real implementation, this would batch process for efficiency
        return [self.generate(p, **kwargs) for p in prompts]


# =============================================================================
# Tool Router (Specialized Models)
# =============================================================================

class ToolRouter:
    """
    Routes requests to specialized models based on intent.
    
    Uses a small classifier model to detect what the user wants,
    then routes to the appropriate specialized model.
    """
    
    # Intent categories
    INTENTS = {
        "chat": "General conversation",
        "code": "Code generation/debugging",
        "image": "Image generation",
        "search": "Web search",
        "file": "File operations",
        "math": "Math/calculations",
    }
    
    def __init__(self):
        self.classifier = None
        self.is_loaded = False
    
    def _log(self, message: str):
        print(f"[ToolRouter] {message}")
    
    def load(self) -> bool:
        """Load the router classifier."""
        self._log("Loading router classifier...")
        
        # In real implementation:
        # from forge_ai.core.tool_router import get_router
        # self.classifier = get_router()
        
        self.is_loaded = True
        return True
    
    def classify_intent(self, text: str) -> str:
        """
        Classify user intent from text.
        
        Args:
            text: User input text
            
        Returns:
            Intent category (chat, code, image, etc.)
        """
        text_lower = text.lower()
        
        # Simple keyword-based classification
        if any(kw in text_lower for kw in ['code', 'function', 'debug', 'program', 'python', 'javascript']):
            return "code"
        elif any(kw in text_lower for kw in ['image', 'picture', 'draw', 'generate image', 'create image']):
            return "image"
        elif any(kw in text_lower for kw in ['search', 'find', 'lookup', 'what is']):
            return "search"
        elif any(kw in text_lower for kw in ['file', 'read', 'write', 'open', 'save']):
            return "file"
        elif any(kw in text_lower for kw in ['calculate', 'math', 'solve', '+', '-', '*', '/']):
            return "math"
        else:
            return "chat"
    
    def route(self, text: str, models: Dict[str, Any]) -> str:
        """
        Route text to appropriate model.
        
        Args:
            text: User input
            models: Dictionary of specialized models
            
        Returns:
            Model name to use
        """
        intent = self.classify_intent(text)
        self._log(f"Detected intent: {intent}")
        
        # Map intent to model
        model_map = {
            "chat": "forge-chat",
            "code": "forge-code",
            "image": "image-gen",
            "search": "web-search",
            "file": "file-ops",
            "math": "forge-math",
        }
        
        model_name = model_map.get(intent, "forge-chat")
        
        if model_name in models:
            return model_name
        else:
            self._log(f"Model '{model_name}' not available, using default")
            return "forge-chat"


# =============================================================================
# Multi-Model Manager
# =============================================================================

class ModelRegistry:
    """
    Manage multiple loaded models.
    
    Allows loading different models for different tasks and
    switching between them efficiently.
    """
    
    def __init__(self):
        self.models: Dict[str, ForgeEngine] = {}
        self.router = ToolRouter()
    
    def _log(self, message: str):
        print(f"[ModelRegistry] {message}")
    
    def load_model(self, name: str, model_path: str) -> bool:
        """Load a model with a given name."""
        self._log(f"Loading model '{name}' from {model_path}")
        
        engine = ForgeEngine()
        if engine.load(model_path):
            self.models[name] = engine
            return True
        return False
    
    def unload_model(self, name: str):
        """Unload a model by name."""
        if name in self.models:
            self.models[name].unload()
            del self.models[name]
            self._log(f"Unloaded model '{name}'")
    
    def list_models(self) -> List[str]:
        """List loaded models."""
        return list(self.models.keys())
    
    def generate(self, prompt: str, model_name: Optional[str] = None,
                 use_routing: bool = True, **kwargs) -> str:
        """
        Generate using appropriate model.
        
        Args:
            prompt: Input prompt
            model_name: Specific model to use (or None for auto-routing)
            use_routing: Whether to use intent-based routing
            
        Returns:
            Generated text
        """
        if not self.models:
            return "[Error: No models loaded]"
        
        if model_name:
            target_model = model_name
        elif use_routing and self.router.is_loaded:
            target_model = self.router.route(prompt, self.models)
        else:
            target_model = list(self.models.keys())[0]
        
        if target_model not in self.models:
            target_model = list(self.models.keys())[0]
        
        self._log(f"Using model: {target_model}")
        return self.models[target_model].generate(prompt, **kwargs)


# =============================================================================
# Example Usage
# =============================================================================

def example_basic_inference():
    """Basic text generation."""
    print("\n" + "="*60)
    print("Example 1: Basic Inference")
    print("="*60)
    
    config = InferenceConfig(
        max_tokens=256,
        temperature=0.7
    )
    
    engine = ForgeEngine(config)
    engine.load()
    
    # Generate text
    prompt = "What is the meaning of life?"
    response = engine.generate(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")


def example_generation_params():
    """Different generation parameters."""
    print("\n" + "="*60)
    print("Example 2: Generation Parameters")
    print("="*60)
    
    engine = ForgeEngine()
    engine.load()
    
    prompt = "Write a creative story about a robot"
    
    print("\nLow temperature (more focused):")
    response = engine.generate(prompt, temperature=0.3)
    print(f"  {response}")
    
    print("\nHigh temperature (more creative):")
    response = engine.generate(prompt, temperature=1.0)
    print(f"  {response}")
    
    print("\nShort generation:")
    response = engine.generate(prompt, max_tokens=50)
    print(f"  {response}")


def example_streaming():
    """Streaming generation."""
    print("\n" + "="*60)
    print("Example 3: Streaming Generation")
    print("="*60)
    
    engine = ForgeEngine()
    engine.load()
    
    prompt = "Explain how computers work"
    
    print(f"Prompt: {prompt}")
    print("Streaming response: ", end="")
    
    for token in engine.generate_stream(prompt, max_tokens=100):
        print(token, end="", flush=True)
    
    print()


def example_tool_routing():
    """Intent-based routing to specialized models."""
    print("\n" + "="*60)
    print("Example 4: Tool Routing")
    print("="*60)
    
    router = ToolRouter()
    router.load()
    
    test_inputs = [
        "Hello, how are you?",
        "Write a Python function to sort a list",
        "Generate an image of a sunset",
        "Search for the latest news",
        "Read the contents of config.json",
        "Calculate 15 * 23 + 7",
    ]
    
    print("Intent classification:\n")
    for text in test_inputs:
        intent = router.classify_intent(text)
        print(f"  '{text[:40]}...' -> {intent}")


def example_multi_model():
    """Using multiple models."""
    print("\n" + "="*60)
    print("Example 5: Multi-Model Registry")
    print("="*60)
    
    registry = ModelRegistry()
    registry.router.load()
    
    # Load multiple models
    registry.load_model("forge-chat", "models/forge-chat")
    registry.load_model("forge-code", "models/forge-code")
    
    print(f"Loaded models: {registry.list_models()}")
    
    # Generate with auto-routing
    prompts = [
        "Tell me a joke",
        "Write a Python function to reverse a string"
    ]
    
    print("\nAuto-routed generation:")
    for prompt in prompts:
        response = registry.generate(prompt, use_routing=True)
        print(f"  Prompt: '{prompt[:30]}...'")
        print(f"  Response: {response}")


def example_batch():
    """Batch inference."""
    print("\n" + "="*60)
    print("Example 6: Batch Inference")
    print("="*60)
    
    engine = ForgeEngine()
    engine.load()
    
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]
    
    print(f"Processing {len(prompts)} prompts in batch...")
    
    start = time.time()
    responses = engine.generate_batch(prompts)
    elapsed = time.time() - start
    
    print(f"Completed in {elapsed:.2f}s")
    
    for prompt, response in zip(prompts, responses):
        print(f"\n  Q: {prompt}")
        print(f"  A: {response[:50]}...")


def example_forge_integration():
    """Real ForgeAI integration."""
    print("\n" + "="*60)
    print("Example 7: ForgeAI Integration")
    print("="*60)
    
    print("For actual ForgeAI inference:")
    print("""
    from forge_ai.core.inference import ForgeEngine
    from forge_ai.core.tool_router import ToolRouter, get_router
    from forge_ai.core.model_registry import ModelRegistry
    
    # Basic inference
    engine = ForgeEngine()
    engine.load("models/forge-small")
    
    response = engine.generate(
        prompt="Hello!",
        max_tokens=256,
        temperature=0.7,
        top_p=0.9
    )
    
    # Streaming
    for token in engine.generate_stream("Tell me a story"):
        print(token, end="")
    
    # With tool routing
    engine = ForgeEngine(use_routing=True)
    engine.load("models/forge-small")
    engine.load_router("models/router")
    
    # Auto-routes to code model if detected
    response = engine.generate("Write a sorting algorithm")
    
    # Command line
    python run.py --run                    # Interactive chat
    python run.py --serve                  # API server
    python run.py --run --model medium     # Use medium model
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("ForgeAI Inference Examples")
    print("="*60)
    
    example_basic_inference()
    example_generation_params()
    example_streaming()
    example_tool_routing()
    example_multi_model()
    example_batch()
    example_forge_integration()
    
    print("\n" + "="*60)
    print("Inference Summary:")
    print("="*60)
    print("""
Inference Parameters:

1. temperature (0.0 - 2.0):
   - Lower = more focused, deterministic
   - Higher = more creative, random
   - Default: 0.7

2. top_p (0.0 - 1.0):
   - Nucleus sampling threshold
   - Lower = fewer token choices
   - Default: 0.9

3. top_k (1 - vocab_size):
   - Number of top tokens to consider
   - Lower = more focused
   - Default: 50

4. max_tokens:
   - Maximum tokens to generate
   - Affects response length
   - Default: 512

5. repetition_penalty (1.0+):
   - Penalizes repeated tokens
   - Higher = less repetition
   - Default: 1.1

Tool Routing:
   - Detects user intent (chat, code, image, etc.)
   - Routes to specialized models
   - Improves response quality

Multi-Model:
   - Load multiple models simultaneously
   - Switch based on task
   - Memory efficient with unloading

Command Line:
   python run.py --run                    # Chat mode
   python run.py --serve --port 5000      # API server
   python run.py --run --model large      # Larger model
   python run.py --run --temp 0.5         # Lower temperature
""")
