#!/usr/bin/env python3
"""
Train Specialized Models for Task-Specific Routing

This script trains small, specialized Enigma models for specific tasks like:
- Intent classification (router)
- Vision captioning
- Code generation
- And more...

All models share the same tokenizer for interoperability.

Usage:
    python scripts/train_specialized_model.py --type router --data data/specialized/router_training.txt --model-size nano
    python scripts/train_specialized_model.py --type vision --data data/specialized/vision_training.txt --model-size tiny
    python scripts/train_specialized_model.py --type code --data data/specialized/code_training.txt --model-size small
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from enigma_engine.core.model import create_model, MODEL_PRESETS
    from enigma_engine.core.tokenizer import get_tokenizer, train_tokenizer
    from enigma_engine.core.training import Trainer, TrainingConfig, train_model
    from enigma_engine.config import CONFIG
except ImportError as e:
    print(f"Error: Required dependencies not installed: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model type configurations
MODEL_TYPE_CONFIG = {
    "router": {
        "description": "Intent classification model (routes to appropriate tool)",
        "default_size": "nano",
        "recommended_epochs": 50,
        "output_dir": "models/specialized",
        "output_name": "intent_router_{size}.pth",
    },
    "vision": {
        "description": "Vision captioning model (describes images from features)",
        "default_size": "tiny",
        "recommended_epochs": 40,
        "output_dir": "models/specialized",
        "output_name": "vision_caption_{size}.pth",
    },
    "code": {
        "description": "Code generation model (writes and explains code)",
        "default_size": "small",
        "recommended_epochs": 40,
        "output_dir": "models/specialized",
        "output_name": "code_gen_{size}.pth",
    },
    "math": {
        "description": "Mathematical reasoning model",
        "default_size": "small",
        "recommended_epochs": 40,
        "output_dir": "models/specialized",
        "output_name": "math_{size}.pth",
    },
    "trainer": {
        "description": "Meta-AI that generates training data for other models",
        "default_size": "small",
        "recommended_epochs": 60,
        "output_dir": "models/specialized",
        "output_name": "trainer_{size}.pth",
    },
    "avatar": {
        "description": "Avatar control model (converts commands to bone movements)",
        "default_size": "tiny",
        "recommended_epochs": 50,
        "output_dir": "models/specialized",
        "output_name": "avatar_{size}.pth",
    },
    "chat": {
        "description": "General conversation model",
        "default_size": "small",
        "recommended_epochs": 40,
        "output_dir": "models/specialized",
        "output_name": "chat_{size}.pth",
    },
}


def load_training_data(data_path: Path) -> str:
    """Load training data from file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        return f.read()


def validate_model_type(model_type: str) -> dict:
    """Validate and get model type configuration."""
    if model_type not in MODEL_TYPE_CONFIG:
        available = ", ".join(MODEL_TYPE_CONFIG.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    return MODEL_TYPE_CONFIG[model_type]


def get_shared_tokenizer():
    """Get or create the shared tokenizer."""
    # Load config to get shared tokenizer path
    try:
        config_path = Path(__file__).parent.parent / "information" / "specialized_models.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                tokenizer_path = config.get("shared_tokenizer", "enigma_engine/vocab_model/bpe_vocab.json")
        else:
            tokenizer_path = "enigma_engine/vocab_model/bpe_vocab.json"
    except Exception:
        tokenizer_path = "enigma_engine/vocab_model/bpe_vocab.json"
    
    vocab_file = Path(__file__).parent.parent / tokenizer_path
    
    logger.info("Loading shared tokenizer...")
    logger.info(f"Looking for tokenizer at: {vocab_file}")
    
    # Try to load existing tokenizer
    try:
        tokenizer = get_tokenizer("bpe", vocab_path=str(vocab_file))
        logger.info(f"Loaded existing tokenizer with vocab size: {tokenizer.vocab_size}")
        return tokenizer
    except Exception as e:
        logger.warning(f"Could not load BPE tokenizer: {e}")
        logger.info("Falling back to character tokenizer...")
        tokenizer = get_tokenizer("char")
        return tokenizer


def train_specialized_model(
    model_type: str,
    data_path: Path,
    model_size: str,
    epochs: int = None,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    device: str = "auto",
    output_path: Path = None,
    verbose: bool = True
):
    """
    Train a specialized model for a specific task.
    
    Args:
        model_type: Type of model (router, vision, code, etc.)
        data_path: Path to training data file
        model_size: Model size preset (nano, tiny, small, etc.)
        epochs: Number of training epochs (None = use recommended)
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on (auto, cuda, cpu)
        output_path: Custom output path (None = use default)
        verbose: Whether to print progress
    """
    # Validate model type
    type_config = validate_model_type(model_type)
    
    logger.info(f"Training {model_type} model - {type_config['description']}")
    logger.info(f"Model size: {model_size}")
    
    # Load training data
    logger.info(f"Loading training data from: {data_path}")
    training_text = load_training_data(data_path)
    
    # Count lines for validation
    lines = [l.strip() for l in training_text.split('\n') if l.strip()]
    question_lines = len([l for l in lines if l.startswith('Q:')])
    answer_lines = len([l for l in lines if l.startswith('A:')])
    logger.info(f"Loaded {question_lines} questions and {answer_lines} answers")
    
    if question_lines < 10:
        logger.warning(f"Very small dataset ({question_lines} Q/A pairs). Results may be poor.")
        logger.warning("Recommended: at least 50+ Q/A pairs for good results")
    
    # Get shared tokenizer
    tokenizer = get_shared_tokenizer()
    
    # Determine epochs
    if epochs is None:
        epochs = type_config["recommended_epochs"]
    logger.info(f"Training for {epochs} epochs")
    
    # Determine output path
    if output_path is None:
        output_dir = Path(type_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = type_config["output_name"].format(size=model_size)
        output_path = output_dir / output_name
    
    logger.info(f"Will save model to: {output_path}")
    
    # Create model
    logger.info(f"Creating {model_size} model...")
    model = create_model(size=model_size, vocab_size=tokenizer.vocab_size)
    
    # Select device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")
    
    # Create training config
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=verbose,
        save_every=max(1, epochs // 5),  # Save 5 checkpoints
        checkpoint_dir=str(output_path.parent / "checkpoints" / f"{model_type}_{model_size}"),
    )
    
    # Create trainer
    trainer = Trainer(model, tokenizer, device=device, config=config)
    
    # Train
    logger.info("Starting training...")
    logger.info("=" * 70)
    
    try:
        trainer.train(training_text)
        
        logger.info("=" * 70)
        logger.info("Training complete!")
        
        # Save final model
        logger.info(f"Saving model to: {output_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model.get_config(),
            'model_type': model_type,
            'model_size': model_size,
            'vocab_size': tokenizer.vocab_size,
            'training_questions': question_lines,
            'epochs': epochs,
        }, output_path)
        
        # Save config alongside
        config_path = output_path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump({
                'model_type': model_type,
                'model_size': model_size,
                'vocab_size': tokenizer.vocab_size,
                'config': model.get_config(),
                'training_questions': question_lines,
                'epochs': epochs,
            }, f, indent=2)
        
        logger.info(f"Saved config to: {config_path}")
        logger.info("✓ Success! Model is ready to use.")
        
        return output_path
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info(f"Checkpoints saved in: {config.checkpoint_dir}")
        return None
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Train specialized models for task-specific routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train intent router (nano model, fast)
  python scripts/train_specialized_model.py --type router --data data/specialized/router_training.txt --model-size nano
  
  # Train vision captioning (tiny model)
  python scripts/train_specialized_model.py --type vision --data data/specialized/vision_training.txt --model-size tiny
  
  # Train code generation (small model)
  python scripts/train_specialized_model.py --type code --data data/specialized/code_training.txt --model-size small
  
  # Custom epochs and learning rate
  python scripts/train_specialized_model.py --type router --data data/specialized/router_training.txt --model-size nano --epochs 100 --lr 5e-4

Available model types: router, vision, code, math
Available model sizes: nano, micro, tiny, mini, small, medium, large, xl, xxl
        """
    )
    
    parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=list(MODEL_TYPE_CONFIG.keys()),
        help='Type of specialized model to train'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to training data file'
    )
    
    parser.add_argument(
        '--model-size',
        type=str,
        default=None,
        choices=list(MODEL_PRESETS.keys()),
        help='Model size preset (default: optimal for model type)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: recommended for model type)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training (default: 8)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to train on (default: auto)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Custom output path for model (default: models/specialized/)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Use default size if not specified
    if args.model_size is None:
        args.model_size = MODEL_TYPE_CONFIG[args.type]['default_size']
        logger.info(f"Using default model size for {args.type}: {args.model_size}")
    
    # Convert paths
    data_path = Path(args.data)
    output_path = Path(args.output) if args.output else None
    
    # Train
    try:
        result = train_specialized_model(
            model_type=args.type,
            data_path=data_path,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            output_path=output_path,
            verbose=not args.quiet
        )
        
        if result:
            print(f"\n✓ Model saved to: {result}")
            print(f"\nTo use this model, update information/specialized_models.json")
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
