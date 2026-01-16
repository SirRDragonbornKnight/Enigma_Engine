#!/usr/bin/env python3
"""
Download recommended models for ForgeAI GUI.

This script downloads HuggingFace models that work well with the GUI
and fit in your GPU memory (RTX 2080 with 8GB VRAM).

Recommended models:
- TinyLlama-1.1B-Chat: Fast, small, great for chat (~1.1B params, ~2.5GB VRAM)
- Phi-2: Microsoft's small but powerful model (~2.7B params, ~6GB VRAM)  
- DialoGPT-medium: Good conversational model (~355M params, ~1.5GB VRAM)
- Qwen2-0.5B-Instruct: Excellent small instruct model (~0.5B params, ~1GB VRAM)
"""

import sys
import os
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path


def download_model(model_id: str, model_name: Optional[str] = None):
    """Download a HuggingFace model."""
    from forge_ai.core.huggingface_loader import HuggingFaceModel, get_huggingface_model_info
    from forge_ai.core.model_registry import ModelRegistry
    
    if model_name is None:
        model_name = model_id.replace("/", "_").lower()
    
    print(f"\n{'='*60}")
    print(f"Downloading: {model_id}")
    print(f"Local name: {model_name}")
    print("="*60)
    
    # Get model info first
    info = get_huggingface_model_info(model_id)
    if info.get("error"):
        print(f"Warning: Could not get model info: {info['error']}")
    else:
        print(f"  Size: {info['size_str']} parameters")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Hidden size: {info['hidden_size']}")
        print(f"  Layers: {info['num_layers']}")
    
    print("\nDownloading model files (this may take a while)...")
    
    try:
        # Create the model instance and load it (triggers download)
        model = HuggingFaceModel(
            model_id,
            device="cuda",
            torch_dtype="float16"
        )
        model.load()
        
        print(f"✓ Model loaded successfully!")
        
        # Register in the model registry
        registry = ModelRegistry()
        
        # Save model info to registry
        models_dir = Path("models") / model_name
        models_dir.mkdir(parents=True, exist_ok=True)
        
        registry.registry["models"][model_name] = {
            "path": str(models_dir.absolute()),
            "size": "huggingface",
            "created": __import__("datetime").datetime.now().isoformat(),
            "has_weights": False,  # Weights are in HF cache
            "source": "huggingface",
            "huggingface_id": model_id,
            "use_custom_tokenizer": False
        }
        registry._save_registry()
        
        print(f"✓ Registered as '{model_name}' in model registry")
        
        # Test generation
        print("\nTesting generation...")
        response = model.generate("Hello, I am", max_new_tokens=20)
        print(f"  Test output: {response[:100]}...")
        
        # Unload to free VRAM
        del model
        import torch
        torch.cuda.empty_cache()
        
        print(f"\n✓ {model_name} ready to use!")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {model_id}: {e}")
        return False


def main():
    """Download recommended models."""
    print("="*60)
    print("ForgeAI Model Downloader")
    print("="*60)
    print("\nThis will download models from HuggingFace Hub.")
    print("Models will be cached in your HuggingFace cache directory.")
    print("\nRecommended models for RTX 2080 (8GB VRAM):")
    print()
    
    # Models sorted by size (smallest first)
    recommended_models = [
        ("Qwen/Qwen2-0.5B-Instruct", "qwen2_0.5b", "~0.5B params, ~1GB VRAM - Fast & capable"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama_chat", "~1.1B params, ~2.5GB VRAM - Great for chat"),
        ("microsoft/DialoGPT-medium", "dialogpt_medium", "~355M params, ~1.5GB VRAM - Conversational"),
        ("microsoft/phi-2", "phi2", "~2.7B params, ~6GB VRAM - Very capable"),
    ]
    
    for i, (model_id, name, desc) in enumerate(recommended_models, 1):
        print(f"  {i}. {name}: {desc}")
    
    print(f"\n  A. Download ALL recommended models")
    print(f"  Q. Quit")
    
    choice = input("\nEnter your choice (1-4, A, or Q): ").strip().upper()
    
    if choice == 'Q':
        print("Exiting.")
        return
    
    if choice == 'A':
        print("\nDownloading all recommended models...")
        for model_id, name, _ in recommended_models:
            download_model(model_id, name)
    elif choice.isdigit() and 1 <= int(choice) <= len(recommended_models):
        idx = int(choice) - 1
        model_id, name, _ = recommended_models[idx]
        download_model(model_id, name)
    else:
        print("Invalid choice.")
        return
    
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print("\nYou can now use these models in the ForgeAI GUI:")
    print("  1. Open the GUI: python run.py --gui")
    print("  2. Go to Model Manager tab")
    print("  3. Select your downloaded model from the list")
    print("  4. Click 'Load Model'")


if __name__ == "__main__":
    main()
