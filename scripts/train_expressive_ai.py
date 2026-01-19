"""
Train the AI to be expressive for avatar integration.

This trains the model to use emotional language that the 
AI-Avatar bridge can detect and display.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge_ai.core.training import train_model, TrainingConfig
from forge_ai.core.model import create_model
from forge_ai.config import CONFIG
from pathlib import Path


def train_expressive_ai(
    model_size: str = "small",
    epochs: int = 5,
    additional_data: str = None
):
    """
    Train the AI to be expressive.
    
    Args:
        model_size: "tiny", "small", "medium", "large"
        epochs: Number of training epochs (3-10 recommended)
        additional_data: Path to additional training data
    """
    
    # Training data files
    data_files = [
        "data/specialized/avatar_expression_training.txt",
        "data/training.txt",  # Your existing training data
    ]
    
    if additional_data:
        data_files.append(additional_data)
    
    # Filter to existing files
    data_files = [f for f in data_files if Path(f).exists()]
    
    print("=" * 50)
    print("EXPRESSIVE AI TRAINING")
    print("=" * 50)
    print(f"Model size: {model_size}")
    print(f"Epochs: {epochs}")
    print(f"Training files: {data_files}")
    print()
    
    # Count data
    total_lines = 0
    for f in data_files:
        with open(f) as file:
            lines = [l for l in file.readlines() if l.strip() and not l.startswith('#')]
            total_lines += len(lines)
    
    print(f"Total training examples: {total_lines}")
    
    # Estimate time
    time_estimates = {
        "nano": "2-5 min",
        "micro": "5-10 min", 
        "tiny": "10-20 min",
        "small": "15-30 min",
        "medium": "30 min - 2 hrs",
        "large": "1-4 hrs",
    }
    print(f"Estimated time: {time_estimates.get(model_size, '30+ min')}")
    print()
    
    input("Press Enter to start training (or Ctrl+C to cancel)...")
    print()
    
    # Create config
    config = TrainingConfig(
        model_size=model_size,
        epochs=epochs,
        learning_rate=0.0001,
        batch_size=4,
        save_every=1,
    )
    
    # Load or create model
    model_path = CONFIG.MODEL_DIR / f"forge_{model_size}"
    
    if model_path.exists():
        print(f"Continuing training from existing model: {model_path}")
    else:
        print(f"Creating new {model_size} model...")
    
    # Combine training data
    combined_data = ""
    for f in data_files:
        with open(f) as file:
            combined_data += file.read() + "\n"
    
    # Save combined data temporarily
    temp_data = Path("data/temp_expressive_training.txt")
    temp_data.write_text(combined_data)
    
    try:
        # Train
        train_model(
            data_path=str(temp_data),
            config=config,
            model_path=str(model_path),
        )
        
        print()
        print("=" * 50)
        print("TRAINING COMPLETE!")
        print("=" * 50)
        print()
        print("Your AI is now more expressive!")
        print("The avatar will react to emotions in the AI's responses.")
        print()
        print("Keywords the avatar detects:")
        print("  Happy: 'happy', 'glad', 'great', 'awesome', '!'")
        print("  Sad: 'sorry', 'unfortunately', 'can't'")
        print("  Surprised: 'wow', 'amazing', 'incredible'")
        print("  Thinking: 'hmm', 'let me think', 'perhaps'")
        print()
        print("To test, run: python examples/avatar_demo.py")
        
    finally:
        # Cleanup
        if temp_data.exists():
            temp_data.unlink()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train expressive AI for avatar")
    parser.add_argument("--size", default="small", help="Model size: tiny, small, medium, large")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--data", help="Additional training data file")
    
    args = parser.parse_args()
    
    train_expressive_ai(
        model_size=args.size,
        epochs=args.epochs,
        additional_data=args.data,
    )
