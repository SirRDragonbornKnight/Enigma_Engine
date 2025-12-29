#!/usr/bin/env python3
"""
Train an Enigma model from command line.

Usage:
    python -m scripts.train --model my_model --data data/training.txt --epochs 10
    python -m scripts.train --model my_model --size medium --epochs 50
"""
import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Train an Enigma language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model", "-m", type=str, required=True,
        help="Model name (creates new if doesn't exist)"
    )
    model_group.add_argument(
        "--size", "-s", type=str, default="small",
        choices=["tiny", "small", "medium", "large", "xl", "xxl", "xxxl"],
        help="Model size preset"
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data", "-d", type=str, default="data/data.txt",
        help="Path to training data file"
    )
    data_group.add_argument(
        "--val-split", type=float, default=0.1,
        help="Validation split ratio"
    )
    
    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs", "-e", type=int, default=10,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--batch-size", "-b", type=int, default=32,
        help="Batch size"
    )
    train_group.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    train_group.add_argument(
        "--warmup", type=int, default=100,
        help="Warmup steps"
    )
    train_group.add_argument(
        "--grad-accum", type=int, default=1,
        help="Gradient accumulation steps"
    )
    train_group.add_argument(
        "--no-amp", action="store_true",
        help="Disable automatic mixed precision"
    )
    
    # Other
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (auto-detected if not specified)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Import after parsing to avoid slow startup
    from enigma.core.model_registry import ModelRegistry
    from enigma.core.trainer import EnigmaTrainer
    
    print(f"Training model: {args.model}")
    print(f"Size: {args.size}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    
    # Create or load model
    registry = ModelRegistry()
    if args.model not in registry.list_models():
        print(f"Creating new model '{args.model}'...")
        registry.create_model(args.model, size=args.size)
    
    # Load the model
    print(f"Loading model '{args.model}'...")
    model, config = registry.load_model(args.model)
    
    # Initialize trainer
    trainer = EnigmaTrainer(
        model=model,
        model_name=args.model,
        registry=registry,
        data_path=args.data,
        use_amp=not args.no_amp,
    )
    
    # Train
    trainer.train(
        epochs=args.epochs,
        save_every=5,
    )
    
    print("Training complete!")


if __name__ == "__main__":
    main()
