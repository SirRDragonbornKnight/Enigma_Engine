"""Test script that mimics exactly what the GUI does."""
import sys
sys.path.insert(0, '.')

import torch
from enigma_engine.core.model_registry import ModelRegistry
from enigma_engine.core.inference import EnigmaEngine

# Load model through registry (like GUI)
r = ModelRegistry()
model, config = r.load_model('qwen2_0.5b')

print("=== Config ===")
print(f"source: {config.get('source')}")
print(f"use_custom_tokenizer: {config.get('use_custom_tokenizer')}")

# Create engine like GUI does
engine = EnigmaEngine.__new__(EnigmaEngine)
engine.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
engine.use_half = False
engine.enable_tools = False
engine.module_manager = None

# Set HuggingFace flags
is_huggingface = config.get("source") == "huggingface"
engine._is_huggingface = is_huggingface
engine.model = model

# Set tokenizer
use_custom_tokenizer = config.get("use_custom_tokenizer", False)
if use_custom_tokenizer:
    from enigma_engine.core.tokenizer import load_tokenizer
    engine.tokenizer = load_tokenizer()
    engine._using_custom_tokenizer = True
else:
    engine.tokenizer = model.tokenizer
    engine._using_custom_tokenizer = False

print(f"\n=== Engine Setup ===")
print(f"_is_huggingface: {engine._is_huggingface}")
print(f"_using_custom_tokenizer: {engine._using_custom_tokenizer}")
print(f"Engine model type: {type(engine.model)}")
print(f"Has chat method: {hasattr(engine.model, 'chat')}")

# Now simulate what AIGenerationWorker does
print(f"\n=== Testing Chat ===")
is_hf = getattr(engine, '_is_huggingface', False)
custom_tokenizer = engine.tokenizer if getattr(engine, '_using_custom_tokenizer', False) else None

print(f"is_hf: {is_hf}")
print(f"custom_tokenizer is None: {custom_tokenizer is None}")
print(f"Will use chat(): {hasattr(engine.model, 'chat') and not custom_tokenizer}")

if is_hf:
    if hasattr(engine.model, 'chat') and not custom_tokenizer:
        print("\n>>> Using engine.model.chat()")
        response = engine.model.chat(
            "hello",
            history=None,
            system_prompt="You are a helpful assistant.",
            max_new_tokens=50,
            temperature=0.7
        )
        print(f"Response: {response}")
    else:
        print("\n>>> Using engine.model.generate() - fallback path")
        response = engine.model.generate(
            "hello",
            max_new_tokens=50,
            temperature=0.7,
            custom_tokenizer=custom_tokenizer
        )
        print(f"Response: {response}")
else:
    print("Not HuggingFace model")
