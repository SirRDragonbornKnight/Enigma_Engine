"""Quick test: generate text from the latest trained model."""
import torch
from pathlib import Path
from enigma_engine.core.inference import EnigmaEngine

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model_path = str(Path("models/checkpoints/best_model.pt"))
print(f"Loading model...")
engine = EnigmaEngine(model_path=model_path, device=device)
print(f"Model loaded")
print()

prompts = [
    "The history of",
    "In order to understand",
    "Science tells us that",
    "The most important thing about",
    "Once upon a time",
]

for prompt in prompts:
    print(f'--- Prompt: "{prompt}" ---')
    try:
        output = engine.generate(
            prompt,
            max_gen=60, temperature=0.8,
            top_k=50,
        )
        print(output.strip())
    except Exception as e:
        import traceback
        traceback.print_exc()
    print()
