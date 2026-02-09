#!/usr/bin/env python3
"""
Streaming Generation Example

Shows how to stream tokens as they're generated.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enigma_engine.core.inference import EnigmaEngine


def main():
    print("Loading Enigma AI Engine...")
    engine = EnigmaEngine()
    
    prompt = "In a land far away, there lived a"
    
    print(f"\nPrompt: {prompt}")
    print("Streaming response:\n")
    
    # Print prompt first
    print(prompt, end="", flush=True)
    
    # Stream tokens
    for token in engine.stream_generate(
        prompt,
        max_gen=100,
        temperature=0.9,
    ):
        print(token, end="", flush=True)
    
    print("\n\n--- Stream complete ---")


if __name__ == "__main__":
    main()
