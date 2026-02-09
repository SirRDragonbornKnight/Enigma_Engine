"""Example: Run inference with your trained AI model.

Replace 'Hello!' with any prompt to test your AI's responses.
"""
from enigma_engine.core.inference import EnigmaEngine

if __name__ == "__main__":
    engine = EnigmaEngine()
    # Change the prompt to whatever you want to ask your AI
    print(engine.generate("Hello!", max_gen=20))
