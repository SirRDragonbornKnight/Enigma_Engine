#!/usr/bin/env python3
"""
Train Base Release Model for Enigma AI Engine

This script trains a base model (~27M "small" preset) suitable for GitHub releases.
The model is trained on curated data covering:
- General conversation and helpfulness
- Self-awareness and identity
- Basic knowledge and reasoning
- Code understanding basics
- Multiple specialized task foundations

Users can then fine-tune this model for their specific needs.

Usage:
    python scripts/train_base_release_model.py
    python scripts/train_base_release_model.py --size small --epochs 50
    python scripts/train_base_release_model.py --output models/releases/enigma_base_v1.0.pth
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from enigma_engine.core.model import create_model, MODEL_PRESETS
    from enigma_engine.core.tokenizer import get_tokenizer
    from enigma_engine.core.training import Trainer, TrainingConfig
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


# =============================================================================
# RELEASE MODEL CONFIGURATION
# =============================================================================

RELEASE_CONFIG = {
    "name": "Enigma Base Model",
    "version": "1.0.0",
    "description": "Pre-trained base model for Enigma AI Engine. Ready for conversation and fine-tuning.",
    "default_size": "small",  # ~27M params
    "recommended_epochs": 50,
    "batch_size": 8,
    "learning_rate": 3e-4,
}

# Data sources to combine for base model
DATA_SOURCES = [
    {
        "path": "data/base_knowledge.txt",
        "weight": 2.0,  # Include twice - foundational knowledge is important
        "description": "Self-awareness, basic conversation, and knowledge fundamentals",
    },
    {
        "path": "data/self_awareness_training.txt",
        "weight": 1.0,
        "description": "AI identity and self-awareness training",
    },
    {
        "path": "data/personality_development.txt",
        "weight": 1.0,
        "description": "Personality and helpful behavior patterns",
    },
    {
        "path": "data/instructions.txt",
        "weight": 1.0,
        "description": "Following instructions and task completion",
    },
    {
        "path": "data/specialized/router_training.txt",
        "weight": 0.5,  # Just basics for routing awareness
        "description": "Basic intent classification understanding",
    },
    {
        "path": "data/specialized/code_training.txt",
        "weight": 0.5,  # Foundation for code understanding
        "description": "Basic code generation and explanation",
    },
    {
        "path": "data/specialized/avatar_training.txt",
        "weight": 0.3,  # Light exposure to avatar concepts
        "description": "Basic avatar command understanding",
    },
]


def load_and_weight_data(sources: List[Dict], base_dir: Path) -> tuple[str, Dict]:
    """
    Load training data from multiple sources and combine with weighting.
    
    Returns:
        Tuple of (combined_text, stats_dict)
    """
    combined = []
    stats = {
        "sources": [],
        "total_lines": 0,
        "total_qa_pairs": 0,
    }
    
    for source in sources:
        path = base_dir / source["path"]
        if not path.exists():
            logger.warning(f"Data source not found, skipping: {path}")
            continue
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count Q/A pairs
            lines = content.split('\n')
            q_count = sum(1 for l in lines if l.strip().startswith('Q:') or l.strip().startswith('USER:'))
            a_count = sum(1 for l in lines if l.strip().startswith('A:') or l.strip().startswith('ASSISTANT:'))
            pair_count = min(q_count, a_count)
            
            # Apply weighting (repeat content based on weight)
            weight = source.get("weight", 1.0)
            for _ in range(int(weight)):
                combined.append(content)
            # Handle fractional weights
            if weight % 1 > 0:
                # Include partial content for fractional weights
                fraction = int(len(lines) * (weight % 1))
                combined.append('\n'.join(lines[:fraction]))
            
            stats["sources"].append({
                "path": source["path"],
                "description": source.get("description", ""),
                "qa_pairs": pair_count,
                "weight": weight,
                "weighted_contribution": int(pair_count * weight),
            })
            stats["total_lines"] += len(lines) * int(weight)
            stats["total_qa_pairs"] += int(pair_count * weight)
            
            logger.info(f"Loaded {pair_count} QA pairs from {source['path']} (weight: {weight})")
            
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
    
    return '\n\n'.join(combined), stats


def get_tokenizer_for_release():
    """Get or create tokenizer for release model."""
    vocab_file = Path(__file__).parent.parent / "enigma_engine/vocab_model/bpe_vocab.json"
    
    try:
        tokenizer = get_tokenizer("bpe", vocab_path=str(vocab_file))
        logger.info(f"Loaded BPE tokenizer with vocab size: {tokenizer.vocab_size}")
        return tokenizer
    except Exception as e:
        logger.warning(f"Could not load BPE tokenizer: {e}")
        logger.info("Falling back to character tokenizer...")
        return get_tokenizer("char")


def train_base_model(
    model_size: str = "small",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    output_path: Optional[Path] = None,
    device: str = "auto",
) -> Optional[Path]:
    """
    Train the base release model.
    
    Args:
        model_size: Model size preset (default: small ~27M params)
        epochs: Training epochs (default from config)
        batch_size: Batch size (default from config)
        learning_rate: Learning rate (default from config)
        output_path: Where to save (default: models/releases/)
        device: Training device (auto, cuda, cpu)
    
    Returns:
        Path to saved model, or None if failed
    """
    # Use defaults from config
    epochs = epochs or RELEASE_CONFIG["recommended_epochs"]
    batch_size = batch_size or RELEASE_CONFIG["batch_size"]
    learning_rate = learning_rate or RELEASE_CONFIG["learning_rate"]
    
    # Base directory
    base_dir = Path(__file__).parent.parent
    
    # Load tokenizer
    tokenizer = get_tokenizer_for_release()
    
    # Load and combine training data
    logger.info("=" * 70)
    logger.info("Loading training data from all sources...")
    logger.info("=" * 70)
    
    training_text, data_stats = load_and_weight_data(DATA_SOURCES, base_dir)
    
    logger.info("")
    logger.info("Data Summary:")
    logger.info(f"  Total QA pairs (weighted): {data_stats['total_qa_pairs']}")
    logger.info(f"  Sources used: {len(data_stats['sources'])}")
    logger.info("")
    
    if data_stats['total_qa_pairs'] < 50:
        logger.warning("Very small training set. Consider adding more data.")
    
    # Determine output path
    if output_path is None:
        output_dir = base_dir / "models" / "releases"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        output_path = output_dir / f"enigma_base_{model_size}_v{RELEASE_CONFIG['version']}_{timestamp}.pth"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create model
    logger.info("=" * 70)
    logger.info(f"Creating {model_size} model...")
    
    model = create_model(size=model_size, vocab_size=tokenizer.vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    model_config = model.get_config()
    model_max_seq_len = model_config.get('max_seq_len', 512)
    logger.info(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    logger.info(f"Model max seq length: {model_max_seq_len}")
    logger.info("=" * 70)
    
    # Select device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")
    
    # Create training config - match model's max_seq_len
    checkpoint_dir = output_path.parent / "checkpoints" / f"base_{model_size}"
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_len=model_max_seq_len,  # Match model's sequence length
        verbose=True,
        save_every=max(1, epochs // 5),
        checkpoint_dir=str(checkpoint_dir),
    )
    
    # Create trainer
    trainer = Trainer(model, tokenizer, device=device, config=config)
    
    # Train
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        trainer.train([training_text])
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("Training complete!")
        logger.info("=" * 70)
        
        # Save final model with metadata
        model_data = {
            'model_state_dict': model.state_dict(),
            'config': model.get_config(),
            'vocab_size': tokenizer.vocab_size,
            'model_size': model_size,
            'param_count': param_count,
            'training_stats': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'qa_pairs_trained': data_stats['total_qa_pairs'],
            },
            'data_sources': data_stats['sources'],
            'release_info': {
                'name': RELEASE_CONFIG['name'],
                'version': RELEASE_CONFIG['version'],
                'description': RELEASE_CONFIG['description'],
                'trained_date': datetime.now().isoformat(),
            },
        }
        
        logger.info(f"Saving model to: {output_path}")
        torch.save(model_data, output_path)
        
        # Save human-readable config/info
        info_path = output_path.with_suffix('.json')
        info_data = {
            'name': RELEASE_CONFIG['name'],
            'version': RELEASE_CONFIG['version'],
            'description': RELEASE_CONFIG['description'],
            'model_size': model_size,
            'param_count': param_count,
            'param_count_human': f"{param_count/1e6:.1f}M",
            'vocab_size': tokenizer.vocab_size,
            'training': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'total_qa_pairs': data_stats['total_qa_pairs'],
            },
            'data_sources': data_stats['sources'],
            'trained_date': datetime.now().isoformat(),
            'usage': {
                'load_command': f"from enigma_engine.core.model import create_model; model = create_model(size='{model_size}')",
                'load_weights': f"model.load_state_dict(torch.load('{output_path.name}')['model_state_dict'])",
            },
        }
        
        with open(info_path, 'w') as f:
            json.dump(info_data, f, indent=2)
        
        logger.info(f"Saved model info to: {info_path}")
        logger.info("")
        logger.info("=" * 70)
        logger.info("SUCCESS! Base model is ready for release.")
        logger.info("=" * 70)
        logger.info("")
        logger.info("To use this model:")
        logger.info(f"  1. Copy {output_path.name} to models/ directory")
        logger.info(f"  2. Load with: model.load_state_dict(torch.load('{output_path.name}')['model_state_dict'])")
        logger.info("")
        
        return output_path
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info(f"Checkpoints saved in: {checkpoint_dir}")
        return None
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Train base release model for Enigma AI Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with defaults (small model, 50 epochs)
    python scripts/train_base_release_model.py
    
    # Train smaller model for testing
    python scripts/train_base_release_model.py --size tiny --epochs 10
    
    # Train larger model
    python scripts/train_base_release_model.py --size medium --epochs 80
    
    # Custom output path
    python scripts/train_base_release_model.py --output models/my_release.pth

The trained model will be saved with full metadata for easy loading and fine-tuning.
        """
    )
    
    parser.add_argument(
        "--size",
        type=str,
        default=RELEASE_CONFIG["default_size"],
        choices=list(MODEL_PRESETS.keys()),
        help=f"Model size preset (default: {RELEASE_CONFIG['default_size']})"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Training epochs (default: {RELEASE_CONFIG['recommended_epochs']})"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size (default: {RELEASE_CONFIG['batch_size']})"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"Learning rate (default: {RELEASE_CONFIG['learning_rate']})"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for model file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to train on (default: auto)"
    )
    
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List data sources and exit"
    )
    
    args = parser.parse_args()
    
    # List sources mode
    if args.list_sources:
        print("\nData sources for base model training:")
        print("=" * 60)
        for source in DATA_SOURCES:
            print(f"\n  {source['path']}")
            print(f"    Weight: {source.get('weight', 1.0)}")
            print(f"    Description: {source.get('description', 'N/A')}")
        print()
        return
    
    # Print header
    print()
    print("=" * 70)
    print("  ENIGMA AI ENGINE - BASE RELEASE MODEL TRAINING")
    print("=" * 70)
    print()
    print(f"  Model Size: {args.size}")
    print(f"  Epochs: {args.epochs or RELEASE_CONFIG['recommended_epochs']}")
    print(f"  Version: {RELEASE_CONFIG['version']}")
    print()
    print("=" * 70)
    print()
    
    # Train
    output_path = args.output
    if output_path:
        output_path = Path(output_path)
    
    train_base_model(
        model_size=args.size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_path=output_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
