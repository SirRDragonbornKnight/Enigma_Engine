"""Example: Train your AI model.

Training data should be in data/data.txt
You can add more training data files and reference them.

Parameters:
  force=True  - Retrain even if model already has weights
  num_epochs  - How many times to go through training data (more = better learning)
"""
from enigma_engine.core.training import train_model

if __name__ == "__main__":
    # Increase num_epochs for better training (takes longer)
    train_model(force=True, num_epochs=5)
