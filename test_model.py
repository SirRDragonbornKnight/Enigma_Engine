"""Quick test of the trained model."""
from enigma.core.inference import EnigmaEngine

engine = EnigmaEngine()
print("Model loaded successfully!")
print(f"Parameters: {engine.model.num_parameters:,}")
print()

prompts = ["Hello", "How are you?", "What is your name?", "Tell me a joke"]
for p in prompts:
    print(f"You: {p}")
    resp = engine.generate(p)
    print(f"AI: {resp}")
    print()
