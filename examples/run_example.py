"""Example: Run inference with your trained AI model.

Replace 'Hello!' with any prompt to test your AI's responses.
"""
from ai_tester.core.inference import AITesterEngine

if __name__ == "__main__":
    engine = AITesterEngine()
    # Change the prompt to whatever you want to ask your AI
    print(engine.generate("Hello!", max_gen=20))
