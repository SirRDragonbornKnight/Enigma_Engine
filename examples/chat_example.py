#!/usr/bin/env python3
"""
Chat Example

Demonstrates the chat interface with conversation history.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enigma_engine.core.inference import EnigmaEngine


def main():
    print("Loading Enigma AI Engine...")
    engine = EnigmaEngine()
    
    print("\n=== Enigma Chat ===")
    print("Type 'quit' to exit, 'clear' to reset history")
    print("-" * 40)
    
    history = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            
            if user_input.lower() == "clear":
                history = []
                print("History cleared.")
                continue
            
            # Generate response with history
            response = engine.chat(
                user_input,
                history=history,
                max_gen=100,
                temperature=0.8,
            )
            
            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
