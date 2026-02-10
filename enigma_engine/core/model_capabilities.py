# type: ignore
"""
Model Capabilities Helper - Help users understand and set up AI capabilities.

This module provides:
1. Recommended models for different tasks
2. Capability checking (what a model can/can't do)
3. Automatic model suggestions based on user needs
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelRecommendation:
    """A recommended model for a specific capability."""
    model_id: str  # HuggingFace model ID
    name: str  # Display name
    size_str: str  # "1.5B", "7B", etc.
    capabilities: list[str]  # What it can do
    requirements: str  # "CPU", "GPU 8GB+", etc.
    description: str
    priority: int = 0  # Higher = better recommendation


# =============================================================================
# Model Recommendations by Capability
# =============================================================================

# Chat models - for general conversation
CHAT_MODELS = [
    ModelRecommendation(
        model_id="Qwen/Qwen2-0.5B-Instruct",
        name="Qwen2 0.5B",
        size_str="0.5B",
        capabilities=["chat", "reasoning"],
        requirements="CPU or any GPU",
        description="Smallest, fastest. Good for testing but limited quality.",
        priority=10
    ),
    ModelRecommendation(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        name="TinyLlama 1.1B",
        size_str="1.1B",
        capabilities=["chat", "reasoning"],
        requirements="CPU (slow) or 4GB+ GPU",
        description="Good balance of speed and quality for small systems.",
        priority=30
    ),
    ModelRecommendation(
        model_id="Qwen/Qwen2-1.5B-Instruct",
        name="Qwen2 1.5B",
        size_str="1.5B",
        capabilities=["chat", "reasoning", "multilingual"],
        requirements="CPU (slow) or 4GB+ GPU",
        description="Multilingual, good instruction following.",
        priority=40
    ),
    ModelRecommendation(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        name="Phi-3 Mini",
        size_str="3.8B",
        capabilities=["chat", "reasoning", "code"],
        requirements="8GB+ GPU recommended",
        description="Microsoft's efficient small model, great quality for size.",
        priority=60
    ),
    ModelRecommendation(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        name="Llama 3.2 3B",
        size_str="3B",
        capabilities=["chat", "reasoning", "code"],
        requirements="6GB+ GPU recommended",
        description="Meta's latest small model, excellent quality.",
        priority=70
    ),
    ModelRecommendation(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        name="Mistral 7B",
        size_str="7B",
        capabilities=["chat", "reasoning", "code", "creative"],
        requirements="16GB+ GPU",
        description="High quality, great for most tasks. Needs good GPU.",
        priority=90
    ),
]

# Vision models - can see and describe images
VISION_MODELS = [
    ModelRecommendation(
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        name="Qwen2-VL 2B",
        size_str="2B",
        capabilities=["vision", "chat", "image_understanding"],
        requirements="6GB+ GPU",
        description="Small but capable vision model. Can describe images and answer questions about them.",
        priority=50
    ),
    ModelRecommendation(
        model_id="llava-hf/llava-1.5-7b-hf",
        name="LLaVA 1.5 7B",
        size_str="7B",
        capabilities=["vision", "chat", "image_understanding", "reasoning"],
        requirements="16GB+ GPU",
        description="Popular vision model, good quality image understanding.",
        priority=70
    ),
    ModelRecommendation(
        model_id="llava-hf/llava-v1.6-mistral-7b-hf",
        name="LLaVA 1.6 Mistral",
        size_str="7B",
        capabilities=["vision", "chat", "image_understanding", "reasoning"],
        requirements="16GB+ GPU",
        description="Latest LLaVA with Mistral backbone, best open-source vision.",
        priority=90
    ),
]

# Code models - specialized for programming
CODE_MODELS = [
    ModelRecommendation(
        model_id="Salesforce/codegen-350M-mono",
        name="CodeGen 350M",
        size_str="350M",
        capabilities=["code", "python"],
        requirements="CPU or any GPU",
        description="Small Python code model, fast but basic.",
        priority=20
    ),
    ModelRecommendation(
        model_id="bigcode/starcoder2-3b",
        name="StarCoder2 3B",
        size_str="3B",
        capabilities=["code", "multi_language"],
        requirements="6GB+ GPU",
        description="Good multi-language code model.",
        priority=50
    ),
    ModelRecommendation(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        name="Qwen2.5 Coder 7B",
        size_str="7B",
        capabilities=["code", "multi_language", "explanation"],
        requirements="16GB+ GPU",
        description="Excellent code model with explanation abilities.",
        priority=80
    ),
]


# =============================================================================
# Capability Definitions
# =============================================================================

CAPABILITIES = {
    "chat": {
        "name": "Chat/Conversation",
        "description": "General conversation and Q&A",
        "models": CHAT_MODELS,
        "forge_tool": "chat",
    },
    "vision": {
        "name": "Vision/Image Understanding",
        "description": "See and describe images, answer questions about pictures",
        "models": VISION_MODELS,
        "forge_tool": "vision",
        "note": "Requires vision-capable model, not regular text models"
    },
    "code": {
        "name": "Code Generation",
        "description": "Write and explain code in various languages",
        "models": CODE_MODELS,
        "forge_tool": "code",
    },
    "voice_output": {
        "name": "Voice Output (TTS)",
        "description": "AI speaks responses aloud",
        "models": [],  # Uses pyttsx3 or ElevenLabs, not a model
        "forge_tool": "audio",
        "note": "Built into Enigma AI Engine! Enable in settings or click voice button.",
        "no_model_needed": True
    },
    "voice_input": {
        "name": "Voice Input (STT)",
        "description": "Speak to AI instead of typing",
        "models": [],  # Uses speech_recognition library
        "forge_tool": None,
        "note": "Built into Enigma AI Engine! Click the REC button in chat.",
        "no_model_needed": True
    },
    "image_generation": {
        "name": "Image Generation",
        "description": "Create images from text descriptions",
        "models": [],  # Uses Stable Diffusion locally or DALL-E API
        "forge_tool": "image",
        "note": "Use Image tab. Local: Stable Diffusion. Cloud: DALL-E API.",
        "separate_system": True
    },
    "avatar": {
        "name": "Avatar Control",
        "description": "Display and animate a 3D avatar",
        "models": [],
        "forge_tool": "avatar",
        "note": "Avatar display is visual only. For AI-controlled expressions, need to train model on avatar commands.",
        "separate_system": True
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_recommended_models(capability: str, max_vram_gb: float = None) -> list[ModelRecommendation]:
    """Get recommended models for a capability, filtered by VRAM if specified."""
    cap_info = CAPABILITIES.get(capability)
    if not cap_info:
        return []
    
    models = cap_info.get("models", [])
    
    if max_vram_gb is not None:
        # Filter by VRAM requirements
        filtered = []
        for m in models:
            req = m.requirements.lower()
            if "cpu" in req:
                filtered.append(m)
            elif "4gb" in req and max_vram_gb >= 4:
                filtered.append(m)
            elif "6gb" in req and max_vram_gb >= 6:
                filtered.append(m)
            elif "8gb" in req and max_vram_gb >= 8:
                filtered.append(m)
            elif "16gb" in req and max_vram_gb >= 16:
                filtered.append(m)
        models = filtered
    
    return sorted(models, key=lambda x: -x.priority)


def get_best_model_for_system(capability: str) -> Optional[ModelRecommendation]:
    """Get the best model for this capability based on system resources."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get VRAM
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            models = get_recommended_models(capability, max_vram_gb=total_vram)
        else:
            # CPU only - get smallest models
            models = get_recommended_models(capability, max_vram_gb=0)
            # Filter to CPU-capable
            models = [m for m in models if "cpu" in m.requirements.lower()]
    except ImportError:
        models = get_recommended_models(capability)
    
    return models[0] if models else None


def check_model_capabilities(model_id: str) -> dict[str, bool]:
    """Check what capabilities a model has based on its type."""
    model_lower = model_id.lower()
    
    caps = {
        "chat": True,  # All text models can chat
        "vision": False,
        "code": False,
        "reasoning": True,
    }
    
    # Check for vision models
    if any(x in model_lower for x in ["vl", "vision", "llava", "cogvlm", "idefics"]):
        caps["vision"] = True
    
    # Check for code models
    if any(x in model_lower for x in ["code", "coder", "codegen", "starcoder", "deepseek-coder"]):
        caps["code"] = True
    
    # All instruction-tuned models have basic reasoning
    if "instruct" in model_lower or "chat" in model_lower:
        caps["reasoning"] = True
    
    return caps


def explain_capability(capability: str) -> str:
    """Get a user-friendly explanation of a capability."""
    cap_info = CAPABILITIES.get(capability, {})
    
    if cap_info.get("no_model_needed"):
        return f"""
**{cap_info.get('name', capability)}**

{cap_info.get('description', '')}

✅ **Already built into Enigma AI Engine!**
{cap_info.get('note', '')}
"""
    
    if cap_info.get("separate_system"):
        return f"""
**{cap_info.get('name', capability)}**

{cap_info.get('description', '')}

ℹ️ {cap_info.get('note', '')}
"""
    
    best_model = get_best_model_for_system(capability)
    if best_model:
        return f"""
**{cap_info.get('name', capability)}**

{cap_info.get('description', '')}

🎯 **Recommended for your system:** {best_model.name}
- Model: `{best_model.model_id}`
- Size: {best_model.size_str}
- Requirements: {best_model.requirements}

To add this model:
1. Open Model Manager (Settings > Models)
2. In HuggingFace section, enter: `{best_model.model_id}`
3. Click "Add"
"""
    
    return f"""
**{cap_info.get('name', capability)}**

{cap_info.get('description', '')}

⚠️ No recommended model found for your system. You may need more GPU memory.
"""


def get_training_requirements() -> dict[str, Any]:
    """Get information about what Enigma AI Engine training needs."""
    return {
        "data_format": {
            "conversation": "Q: question\\nA: answer",
            "tool_call": '<tool_call>{"tool": "name", "params": {...}}</tool_call>',
            "tool_result": '<tool_result>{"success": true, "result": "..."}</tool_result>',
        },
        "minimum_data": {
            "lines": 1000,
            "note": "More data = better results. 10,000+ lines recommended."
        },
        "training_tips": [
            "Use consistent format (Q:/A: for all conversations)",
            "Include varied examples of each capability you want",
            "Train tool usage with many examples if you want AI to use tools",
            "Don't train during inference (close chat first)",
            "Save backups before training",
        ],
        "weird_issues": [
            {
                "issue": "Model outputs raw numbers/tensors",
                "cause": "Model is too small or not trained on conversation",
                "fix": "Use a larger pre-trained model or train with more data"
            },
            {
                "issue": "Model repeats itself endlessly",
                "cause": "Repetition penalty too low or temperature too high",
                "fix": "Increase repetition_penalty to 1.2+ or lower temperature to 0.7"
            },
            {
                "issue": "Model ignores tool calls",
                "cause": "Not enough tool training examples",
                "fix": "Add 500+ tool call examples to training data"
            },
            {
                "issue": "HuggingFace models can't be trained",
                "cause": "By design - HF models are read-only in Enigma AI Engine",
                "fix": "Use LoRA fine-tuning or train a Forge model instead"
            },
        ]
    }


# =============================================================================
# Quick Setup Functions
# =============================================================================

def setup_chat_ai(model_id: str = None) -> dict[str, Any]:
    """Set up a model as the main chat AI."""
    if model_id is None:
        best = get_best_model_for_system("chat")
        if best:
            model_id = best.model_id
        else:
            return {"success": False, "error": "No suitable model found"}
    
    try:
        from ..core.tool_router import get_router
        router = get_router()
        router.assign_model("chat", f"huggingface:{model_id}", priority=100)
        return {"success": True, "model": model_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


def setup_vision_ai(model_id: str = None) -> dict[str, Any]:
    """Set up a vision model."""
    if model_id is None:
        best = get_best_model_for_system("vision")
        if best:
            model_id = best.model_id
        else:
            return {"success": False, "error": "No suitable vision model found for your system"}
    
    try:
        from ..core.tool_router import get_router
        router = get_router()
        router.assign_model("vision", f"huggingface:{model_id}", priority=100)
        return {"success": True, "model": model_id}
    except Exception as e:
        return {"success": False, "error": str(e)}
