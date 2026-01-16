"""Quick model download helper."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge_ai.core.huggingface_loader import HuggingFaceModel, get_huggingface_model_info
from forge_ai.core.model_registry import ModelRegistry
from datetime import datetime
from pathlib import Path

def download_and_register(model_id, model_name):
    print(f"Downloading {model_id}...")
    info = get_huggingface_model_info(model_id)
    print(f"Size: {info['size_str']}")
    
    model = HuggingFaceModel(model_id, device='cuda', torch_dtype='float16')
    model.load()
    print("Model loaded!")
    
    registry = ModelRegistry()
    models_dir = Path('models') / model_name
    models_dir.mkdir(parents=True, exist_ok=True)
    
    registry.registry['models'][model_name] = {
        'path': str(models_dir.absolute()),
        'size': 'huggingface',
        'created': datetime.now().isoformat(),
        'has_weights': False,
        'source': 'huggingface',
        'huggingface_id': model_id,
        'use_custom_tokenizer': False
    }
    registry._save_registry()
    print(f"Registered as {model_name}")
    
    # Test
    response = model.generate("Hello!", max_new_tokens=30)
    print(f"Test: {response[:100]}")
    return model

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python quick_download.py <model_id> <local_name>")
        sys.exit(1)
    download_and_register(sys.argv[1], sys.argv[2])
