# Enigma Core Package
"""
Core components of the Enigma AI Engine:
  - model.py: TinyEnigma transformer (scalable architecture)
  - ai_brain.py: Learning and memory system
  - inference.py: Text generation
  - training.py: Model training
  - tokenizer.py: Text tokenization
"""

from .model import TinyEnigma, EnigmaModel
from .ai_brain import AIBrain, get_brain, set_auto_learn

__all__ = [
    'TinyEnigma',
    'EnigmaModel', 
    'AIBrain',
    'get_brain',
    'set_auto_learn'
]
