"""
Build Sacrifice Model
=====================

Build and train the sacrifice model from scratch using:
- Advanced byte-level BPE tokenizer
- State-of-the-art transformer architecture (RoPE, RMSNorm, SwiGLU, GQA)
- Production-grade training system (AMP, gradient accumulation, cosine warmup)

Usage:
    python scripts/build_sacrifice.py
    python scripts/build_sacrifice.py --size medium --epochs 50
"""
import sys
import argparse
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from enigma.core.model import Enigma, create_model
from enigma.core.training import Trainer, TrainingConfig
from enigma.core.advanced_tokenizer import AdvancedBPETokenizer

# Constants
DEFAULT_VOCAB_SIZE = 8000
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_EPOCHS = 30
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_WARMUP_STEPS = 100
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_GRAD_ACCUMULATION = 4
MIN_FREQUENCY = 2

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
VOCAB_DIR = BASE_DIR / 'enigma' / 'vocab_model'

# Training data files to load
TRAINING_DATA_FILES = [
    'default_training_data.txt',
    'starter_training.txt',
    'training_data.txt',
    'data.txt',
    'user_training.txt',
]


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def gather_training_data() -> str:
    """Gather all training data from data directory.
    
    Returns:
        str: Combined training text
    """
    all_text = []
    total_chars = 0
    files_loaded = 0
    
    for filename in TRAINING_DATA_FILES:
        filepath = DATA_DIR / filename
        if filepath.exists():
            try:
                text = filepath.read_text(encoding='utf-8', errors='ignore')
                if text.strip():  # Only add non-empty files
                    all_text.append(text)
                    total_chars += len(text)
                    files_loaded += 1
                    print(f"  ✓ {filename}: {len(text):,} chars")
            except Exception as e:
                print(f"  ✗ {filename}: Error reading file - {e}")
        else:
            print(f"  - {filename}: Not found (skipping)")
    
    if not all_text:
        raise ValueError("No training data found! Please add data files to the data/ directory.")
    
    combined = '\n'.join(all_text)
    print(f"\n  Total: {files_loaded} files, {total_chars:,} characters")
    
    return combined


def train_bpe_tokenizer(
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    training_text: str = None
) -> AdvancedBPETokenizer:
    """Train the advanced BPE tokenizer.
    
    Args:
        vocab_size: Target vocabulary size
        training_text: Pre-loaded training text (optional, will load if not provided)
    
    Returns:
        AdvancedBPETokenizer: Trained tokenizer
    """
    print_header("Training BPE Tokenizer")
    
    # Check if tokenizer already exists
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = VOCAB_DIR / 'bpe_vocab.json'
    
    if tokenizer_path.exists():
        print(f"  Found existing tokenizer at {tokenizer_path}")
        tokenizer = AdvancedBPETokenizer(vocab_file=tokenizer_path)
        print(f"  ✓ Loaded tokenizer: {tokenizer.vocab_size:,} tokens")
        return tokenizer
    
    # Gather training data if not provided
    if training_text is None:
        print("  Gathering training data...")
        training_text = gather_training_data()
    else:
        print("  Using pre-loaded training data")
    
    # Create and train tokenizer
    print(f"\n  Training tokenizer (vocab_size={vocab_size})...")
    tokenizer = AdvancedBPETokenizer()
    tokenizer.train(
        texts=[training_text],
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        verbose=True,
    )
    
    # Save tokenizer
    tokenizer.save(tokenizer_path)
    print(f"\n  ✓ Saved tokenizer to {tokenizer_path}")
    
    # Test tokenizer
    print("\n  Testing tokenizer:")
    test_strings = [
        "Q: Hello, how are you?",
        "A: I'm doing great, thank you!",
        "The quick brown fox jumps over the lazy dog.",
    ]
    for s in test_strings:
        tokens = tokenizer.encode(s)
        decoded = tokenizer.decode(tokens)
        match = "✓" if decoded == s else "✗"
        print(f"    {match} '{s}' -> {len(tokens)} tokens")
    
    return tokenizer


def create_sacrifice_model(
    tokenizer: AdvancedBPETokenizer, 
    size: str = 'small'
) -> Enigma:
    """Create the sacrifice model with proper configuration."""
    print_header(f"Creating {size.upper()} Model")
    
    # Create model with tokenizer vocab size
    model = create_model(size, vocab_size=tokenizer.vocab_size)
    
    # Print architecture info
    config = model.config
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"  Architecture: Enigma (Advanced Transformer)")
    print(f"  Parameters: {num_params:,}")
    print(f"  Vocab size: {config.vocab_size:,}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Attention heads: {config.num_heads} (KV: {config.num_kv_heads})")
    print(f"  FFN hidden: {config.intermediate_size}")
    print(f"  Max seq len: {config.max_seq_len}")
    print(f"  Features: RoPE, RMSNorm, SwiGLU, GQA")
    
    return model


def train_sacrifice_model(
    model: Enigma,
    tokenizer: AdvancedBPETokenizer,
    training_text: str = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> dict:
    """Train the model using the production trainer.
    
    Args:
        model: Enigma model to train
        tokenizer: Tokenizer for text encoding
        training_text: Pre-loaded training text (optional, will load if not provided)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    
    Returns:
        dict: Training results
    """
    print_header("Training Model")
    
    # Gather training data if not provided
    if training_text is None:
        print("  Loading training data...")
        training_text = gather_training_data()
    else:
        print("  Using pre-loaded training data")
    
    # Training configuration
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        warmup_steps=DEFAULT_WARMUP_STEPS,
        grad_clip=DEFAULT_GRAD_CLIP,
        grad_accumulation_steps=DEFAULT_GRAD_ACCUMULATION,
        use_amp=torch.cuda.is_available(),
        max_seq_len=model.config.max_seq_len,
        save_every=5,
        log_every=10,
        verbose=True,
    )
    
    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    trainer = Trainer(model, tokenizer, config, device=device)
    
    # Train
    print("\n  Starting training...")
    results = trainer.train(texts=[training_text], epochs=epochs)
    
    return results


def save_sacrifice_model(
    model: Enigma,
    tokenizer: AdvancedBPETokenizer,
    results: dict,
    model_name: str = 'sacrifice'
):
    """Save the trained model and tokenizer."""
    print_header("Saving Model")
    
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = model_dir / 'model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"  Model: {model_path}")
    
    # Save tokenizer
    tokenizer_path = model_dir / 'tokenizer.json'
    tokenizer.save(tokenizer_path)
    print(f"  Tokenizer: {tokenizer_path}")
    
    # Save config
    import json
    config_dict = {
        'model_config': model.config.__dict__,
        'training_results': {
            'final_loss': results.get('final_loss'),
            'best_loss': results.get('best_loss'),
            'total_steps': results.get('total_steps'),
        },
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'vocab_size': tokenizer.vocab_size,
    }
    
    config_path = model_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"  Config: {config_path}")
    
    print(f"\n  Total saved to: {model_dir}")


def test_model(model: Enigma, tokenizer: AdvancedBPETokenizer):
    """Test the trained model with sample prompts."""
    print_header("Testing Model")
    
    model.eval()
    device = next(model.parameters()).device
    
    test_prompts = [
        "Q: What is the capital of France?",
        "Q: Hello, how are you?",
        "The meaning of life is",
    ]
    
    for prompt in test_prompts:
        print(f"\n  Prompt: {prompt}")
        
        # Encode
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)], 
            dtype=torch.long, 
            device=device
        )
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
            )
        
        # Decode
        response = tokenizer.decode(output_ids[0].tolist())
        print(f"  Response: {response}")


def main():
    parser = argparse.ArgumentParser(description="Build Sacrifice Model")
    parser.add_argument("--size", type=str, default="small",
                       choices=["tiny", "small", "medium", "large", "xl", "xxl"],
                       help="Model size")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Vocab size")
    parser.add_argument("--name", type=str, default="sacrifice", help="Model name")
    parser.add_argument("--skip-tokenizer", action="store_true", 
                       help="Skip tokenizer training (use existing)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training (create only)")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  ENIGMA ENGINE - BUILD SACRIFICE MODEL")
    print("=" * 60)
    print(f"\n  Model size: {args.size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Vocab size: {args.vocab_size}")
    
    start_time = time.time()
    
    # Load training data once (if we're going to use it)
    training_data = None
    if not args.skip_tokenizer or not args.skip_training:
        print_header("Loading Training Data")
        training_data = gather_training_data()
    
    # Step 1: Train tokenizer
    if args.skip_tokenizer:
        tokenizer_path = VOCAB_DIR / 'bpe_vocab.json'
        if tokenizer_path.exists():
            print_header("Loading Existing Tokenizer")
            tokenizer = AdvancedBPETokenizer(vocab_file=tokenizer_path)
            print(f"  ✓ Loaded tokenizer: {tokenizer.vocab_size:,} tokens")
        else:
            tokenizer = train_bpe_tokenizer(
                vocab_size=args.vocab_size,
                training_text=training_data
            )
    else:
        tokenizer = train_bpe_tokenizer(
            vocab_size=args.vocab_size,
            training_text=training_data
        )
    
    # Step 2: Create model
    model = create_sacrifice_model(tokenizer, size=args.size)
    
    # Step 3: Train model
    if not args.skip_training:
        results = train_sacrifice_model(
            model, 
            tokenizer,
            training_text=training_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        
        # Step 4: Save model
        save_sacrifice_model(model, tokenizer, results, args.name)
        
        # Step 5: Test model
        test_model(model, tokenizer)
    else:
        print_header("Skipping Training (--skip-training)")
        results = {}
    
    # Done
    elapsed = time.time() - start_time
    print_header("Build Complete")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    
    if results:
        print(f"  Final loss: {results.get('final_loss', 'N/A')}")
        print(f"  Best loss: {results.get('best_loss', 'N/A')}")
    
    print(f"\n  Model ready at: {MODELS_DIR / args.name}")


if __name__ == "__main__":
    main()
