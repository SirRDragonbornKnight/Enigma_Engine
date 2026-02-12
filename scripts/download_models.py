#!/usr/bin/env python3
"""
Download HuggingFace models for Enigma AI Engine.

Usage:
    python scripts/download_models.py                           # Interactive menu
    python scripts/download_models.py Qwen/Qwen2-0.5B-Instruct  # Download specific model
    python scripts/download_models.py --list                    # List recommended models
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Recommended models (id, local_name, description)
RECOMMENDED = [
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "deepseek_r1_32b", "~32B, ~20GB VRAM - Reasoning"),
    ("meta-llama/Llama-3.3-70B-Instruct", "llama3_3_70b", "~70B, ~40GB VRAM - Top quality"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen2_5_7b", "~7B, ~8GB VRAM - Great balance"),
    ("microsoft/Phi-3.5-mini-instruct", "phi3_5_mini", "~3.8B, ~4GB VRAM - Fast"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral_7b", "~7B, ~8GB VRAM - Efficient"),
    ("google/gemma-2-9b-it", "gemma2_9b", "~9B, ~10GB VRAM - Versatile"),
]


def download_model(model_id: str, local_name: str | None = None) -> bool:
    """Download and register a HuggingFace model."""
    try:
        from enigma_engine.core.huggingface_loader import HuggingFaceModel, get_huggingface_model_info
        from enigma_engine.core.model_registry import ModelRegistry
    except ImportError as e:
        print(f"Error: Missing dependencies: {e}")
        return False
    
    if local_name is None:
        local_name = model_id.replace("/", "_").lower()
    
    print(f"\n{'='*50}")
    print(f"Downloading: {model_id}")
    print(f"Local name: {local_name}")
    print("="*50)
    
    # Get info
    info = get_huggingface_model_info(model_id)
    if not info.get("error"):
        print(f"Size: {info['size_str']} | Arch: {info['architecture']}")
    
    print("\nDownloading (this may take a while)...")
    
    try:
        model = HuggingFaceModel(model_id, device="cuda", torch_dtype="float16")
        model.load()
        print("[OK] Model loaded!")
        
        # Register
        registry = ModelRegistry()
        models_dir = Path("models") / local_name
        models_dir.mkdir(parents=True, exist_ok=True)
        
        registry.registry["models"][local_name] = {
            "path": str(models_dir.absolute()),
            "size": "huggingface",
            "created": datetime.now().isoformat(),
            "has_weights": False,
            "source": "huggingface",
            "huggingface_id": model_id,
        }
        registry._save_registry()
        print(f"[OK] Registered as '{local_name}'")
        
        # Quick test
        print("\nTesting...")
        response = model.generate("Hello, I am", max_new_tokens=20)
        print(f"Test: {response[:80]}...")
        
        # Cleanup
        del model
        import torch
        torch.cuda.empty_cache()
        
        print(f"\n[OK] {local_name} ready!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Failed: {e}")
        return False


def main() -> int:
    """CLI entry point."""
    print("="*50)
    print("Enigma AI Engine Model Downloader")
    print("="*50)
    
    # Direct download
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--list":
            print("\nRecommended models:")
            for model_id, name, desc in RECOMMENDED:
                print(f"  {name}: {desc}")
                print(f"    {model_id}")
            return 0
        
        # Check if it's a known model
        for model_id, name, _ in RECOMMENDED:
            if arg == name or arg == model_id:
                return 0 if download_model(model_id, name) else 1
        
        # Assume it's a custom HuggingFace ID
        local_name = sys.argv[2] if len(sys.argv) > 2 else None
        return 0 if download_model(arg, local_name) else 1
    
    # Interactive menu
    print("\nRecommended models:\n")
    for i, (model_id, name, desc) in enumerate(RECOMMENDED, 1):
        print(f"  {i}. {name}: {desc}")
    print(f"\n  A. Download ALL")
    print(f"  Q. Quit")
    
    choice = input("\nChoice (1-4, A, Q): ").strip().upper()
    
    if choice == 'Q':
        return 0
    
    if choice == 'A':
        for model_id, name, _ in RECOMMENDED:
            download_model(model_id, name)
    elif choice.isdigit() and 1 <= int(choice) <= len(RECOMMENDED):
        model_id, name, _ = RECOMMENDED[int(choice) - 1]
        download_model(model_id, name)
    else:
        print("Invalid choice.")
        return 1
    
    print("\n" + "="*50)
    print("Done! Use models in GUI: python run.py --gui")
    print("="*50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
