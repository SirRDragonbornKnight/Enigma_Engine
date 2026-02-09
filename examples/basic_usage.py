#!/usr/bin/env python3
"""
Basic Usage Example

Demonstrates the simplest way to use Enigma for text generation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enigma_engine.core.inference import EnigmaEngine


def main():
    # Create inference engine (auto-loads model)
    print("Loading Enigma AI Engine...")
    engine = EnigmaEngine()
    
    # Simple generation
    prompt = "Once upon a time"
    print(f"\nPrompt: {prompt}")
    
    response = engine.generate(prompt, max_gen=50)
    print(f"Response: {response}")
    
    # With different parameters
    print("\n--- With higher temperature ---")
    response = engine.generate(
        "The meaning of life is",
        max_gen=30,
        temperature=1.2,  # More creative
    )
    print(f"Creative: {response}")
    
    print("\n--- With lower temperature ---")
    response = engine.generate(
        "The meaning of life is",
        max_gen=30,
        temperature=0.3,  # More focused
    )
    print(f"Focused: {response}")


if __name__ == "__main__":
    main()
