"""
Sacrifice Model - Build, Train, Test
=====================================

This script creates the ultimate sacrifice model using:
- Advanced byte-level BPE tokenizer
- State-of-the-art transformer architecture
- Production-grade training system

Run: python scripts/build_sacrifice.py
"""
import sys
import json
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from enigma.core.advanced_tokenizer import AdvancedBPETokenizer
from enigma.core.advanced_model import EnigmaModel, EnigmaConfig, create_model, MODEL_CONFIGS
from enigma.core.advanced_trainer import (
    Trainer, TrainingConfig, 
    TextDataset, QADataset,
    load_training_data, load_qa_data
)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
VOCAB_DIR = BASE_DIR / 'enigma' / 'vocab_model'


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def train_tokenizer(vocab_size: int = 8000) -> AdvancedBPETokenizer:
    """Train the advanced BPE tokenizer on all available data."""
    print_header("Training BPE Tokenizer")
    
    # Collect all training data
    data_files = [
        'default_training_data.txt',
        'starter_training.txt',
        'training_data.txt',
        'enigma_training.txt',
        'data.txt',
    ]
    
    all_text = []
    for filename in data_files:
        filepath = DATA_DIR / filename
        if filepath.exists():
            text = filepath.read_text(encoding='utf-8', errors='ignore')
            all_text.append(text)
            print(f"  Loaded {filename}: {len(text):,} chars")
    
    combined_text = '\n'.join(all_text)
    print(f"\n  Total: {len(combined_text):,} characters")
    
    # Create and train tokenizer
    tokenizer = AdvancedBPETokenizer()
    tokenizer.train(
        texts=[combined_text],  # Pass as list
        vocab_size=vocab_size,
        min_frequency=2,
        verbose=True,
    )
    
    # Save tokenizer
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(VOCAB_DIR / 'advanced_bpe')
    print(f"\n  Saved tokenizer to {VOCAB_DIR / 'advanced_bpe'}")
    
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
        print(f"    '{s}' -> {len(tokens)} tokens -> '{decoded}'")
    
    return tokenizer


def create_sacrifice_model(tokenizer: AdvancedBPETokenizer, size: str = 'small') -> EnigmaModel:
    """Create the sacrifice model with proper vocab size."""
    print_header(f"Creating {size.upper()} Model")
    
    # Get config for size
    config = MODEL_CONFIGS[size]
    config.vocab_size = tokenizer.vocab_size
    
    # Create model
    model = EnigmaModel(config)
    
    print(f"  Architecture: Enigma v2 (Advanced Transformer)")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Vocab size: {config.vocab_size:,}")
    print(f"  Dimensions: {config.dim}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads} (KV: {config.n_kv_heads})")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Max seq len: {config.max_seq_len}")
    
    return model


def train_model(
    model: EnigmaModel,
    tokenizer: AdvancedBPETokenizer,
    model_name: str = 'sacrifice',
    max_epochs: int = 50,
    learning_rate: float = 3e-4,
) -> dict:
    """Train the model on available data."""
    print_header("Training Model")
    
    # Collect training texts
    data_files = [
        'default_training_data.txt',
        'starter_training.txt',
        'training_data.txt',
        'enigma_training.txt',
    ]
    
    texts = []
    for filename in data_files:
        filepath = DATA_DIR / filename
        if filepath.exists():
            texts.append(filepath.read_text(encoding='utf-8', errors='ignore'))
    
    combined = '\n'.join(texts)
    print(f"  Training on {len(combined):,} characters")
    
    # Create dataset
    dataset = TextDataset(
        texts=[combined],
        tokenizer=tokenizer,
        max_length=model.config.max_seq_len,
        stride=model.config.max_seq_len // 2,
    )
    
    # Training config
    train_config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=4,  # Adjust based on GPU memory
        gradient_accumulation_steps=4,
        max_epochs=max_epochs,
        warmup_steps=100,
        checkpoint_interval=500,
        log_interval=10,
        use_amp=torch.cuda.is_available(),
    )
    
    # Create model directory
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(model, train_config)
    
    # Train!
    history = trainer.train(
        train_dataset=dataset,
        checkpoint_dir=model_dir,
    )
    
    # Save final model and metadata
    save_model(model, tokenizer, model_name, history)
    
    return history


def save_model(
    model: EnigmaModel,
    tokenizer: AdvancedBPETokenizer,
    model_name: str,
    history: dict = None,
):
    """Save model, tokenizer reference, and metadata."""
    print_header("Saving Model")
    
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = model_dir / 'model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': model.config.vocab_size,
            'dim': model.config.dim,
            'n_layers': model.config.n_layers,
            'n_heads': model.config.n_heads,
            'n_kv_heads': model.config.n_kv_heads,
            'max_seq_len': model.config.max_seq_len,
            'hidden_dim': model.config.hidden_dim,
            'dropout': model.config.dropout,
            'bias': model.config.bias,
            'rope_theta': model.config.rope_theta,
        },
    }, model_path)
    print(f"  Saved model to {model_path}")
    
    # Save metadata
    metadata = {
        'name': model_name,
        'type': 'advanced_enigma_v2',
        'tokenizer': 'advanced_bpe',
        'tokenizer_path': str(VOCAB_DIR / 'advanced_bpe'),
        'vocab_size': model.config.vocab_size,
        'parameters': model.count_parameters(),
        'config': {
            'dim': model.config.dim,
            'n_layers': model.config.n_layers,
            'n_heads': model.config.n_heads,
            'n_kv_heads': model.config.n_kv_heads,
            'max_seq_len': model.config.max_seq_len,
        },
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    if history:
        metadata['training'] = {
            'final_loss': history.get('final_loss'),
            'best_loss': history.get('best_loss'),
            'total_steps': history.get('total_steps'),
            'epochs': history.get('epochs'),
        }
    
    metadata_path = model_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")
    
    # Update registry
    update_registry(model_name, metadata)


def update_registry(model_name: str, metadata: dict):
    """Update the model registry."""
    registry_path = MODELS_DIR / 'registry.json'
    
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {'models': {}, 'created': time.strftime('%Y-%m-%dT%H:%M:%S')}
    
    registry['models'][model_name] = {
        'path': str(MODELS_DIR / model_name),
        'type': metadata.get('type', 'unknown'),
        'parameters': metadata.get('parameters', 0),
        'vocab_size': metadata.get('vocab_size', 0),
        'created': metadata.get('created', ''),
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"  Updated registry")


def load_model(model_name: str) -> tuple:
    """Load model and tokenizer."""
    model_dir = MODELS_DIR / model_name
    
    # Load metadata
    metadata_path = model_dir / 'metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Load tokenizer
    tokenizer = AdvancedBPETokenizer()
    tokenizer.load(metadata['tokenizer_path'])
    
    # Load model
    model_path = model_dir / 'model.pt'
    checkpoint = torch.load(model_path, map_location='cpu')
    
    config = EnigmaConfig(**checkpoint['config'])
    model = EnigmaModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer


def test_model(model: EnigmaModel, tokenizer: AdvancedBPETokenizer):
    """Test model generation."""
    print_header("Testing Model")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Test prompts
    prompts = [
        "Q: Hello!",
        "Q: What is your name?",
        "Q: How are you?",
        "Q: Tell me about yourself.",
    ]
    
    for prompt in prompts:
        print(f"\n  Prompt: {prompt}")
        
        # Encode prompt
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        
        # Generate
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
    """Main entry point."""
    print("\n" + "="*60)
    print("  SACRIFICE MODEL BUILDER")
    print("  Building the Ultimate AI from Scratch")
    print("="*60)
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
    else:
        print("\n  WARNING: No GPU detected, training will be slow!")
    
    # Step 1: Train tokenizer
    tokenizer = train_tokenizer(vocab_size=8000)
    
    # Step 2: Create model
    # Choose size based on available GPU memory
    # tiny: ~2M params, small: ~15M params, medium: ~50M params
    model = create_sacrifice_model(tokenizer, size='small')
    
    # Step 3: Train model
    history = train_model(
        model=model,
        tokenizer=tokenizer,
        model_name='sacrifice',
        max_epochs=30,
        learning_rate=3e-4,
    )
    
    # Step 4: Test model
    test_model(model, tokenizer)
    
    print_header("Complete!")
    print(f"  Model saved to: {MODELS_DIR / 'sacrifice'}")
    print(f"  Final loss: {history.get('final_loss', 'N/A'):.4f}")
    print(f"  Best loss: {history.get('best_loss', 'N/A'):.4f}")


if __name__ == '__main__':
    main()
