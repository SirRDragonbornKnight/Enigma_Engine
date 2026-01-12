#!/usr/bin/env python3
"""
Train the BPE tokenizer on your data.

This creates a tokenizer that learns patterns specific to YOUR training data.
Run this BEFORE training your model.

Usage:
    python scripts/train_tokenizer.py --data data/enigma_training.txt --vocab-size 8000
    python scripts/train_tokenizer.py --data data/*.txt --vocab-size 10000
"""
import argparse
import sys
from pathlib import Path
from glob import glob

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--data", "-d", type=str, nargs="+", required=True,
                        help="Path(s) to training data files (supports glob patterns)")
    parser.add_argument("--vocab-size", "-v", type=int, default=8000,
                        help="Target vocabulary size (default: 8000)")
    parser.add_argument("--min-freq", type=int, default=2,
                        help="Minimum pair frequency for merging (default: 2)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path (default: ai_tester/vocab_model/bpe_vocab.json)")
    
    args = parser.parse_args()
    
    # Expand glob patterns
    data_files = []
    for pattern in args.data:
        matches = glob(pattern)
        if matches:
            data_files.extend(matches)
        elif Path(pattern).exists():
            data_files.append(pattern)
    
    if not data_files:
        print(f"Error: No data files found matching: {args.data}")
        sys.exit(1)
    
    print(f"Found {len(data_files)} data file(s):")
    for f in data_files:
        print(f"  - {f}")
    
    # Load all texts
    texts = []
    total_chars = 0
    for path in data_files:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            texts.append(text)
            total_chars += len(text)
            print(f"  Loaded {path}: {len(text):,} chars")
    
    print(f"\nTotal: {total_chars:,} characters from {len(texts)} file(s)")
    
    # Train tokenizer
    from ai_tester.core.bpe_tokenizer import BPETokenizer
    
    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=args.vocab_size, min_frequency=args.min_freq, verbose=True)
    
    # Save
    output_path = args.output or "ai_tester/vocab_model/bpe_vocab.json"
    tokenizer.save(Path(output_path))
    
    # Test it
    print("\n" + "=" * 50)
    print("TESTING TOKENIZER")
    print("=" * 50)
    
    test_sentences = [
        "Q: Hello, how are you?",
        "A: I am Enigma, an AI assistant.",
        "Q: What is machine learning?",
        "A: Machine learning is a type of AI that learns from data.",
    ]
    
    for sentence in test_sentences:
        ids = tokenizer.encode(sentence)
        decoded = tokenizer.decode(ids)
        print(f"\nOriginal: {sentence}")
        print(f"Tokens:   {len(ids)} IDs")
        print(f"Decoded:  {decoded}")
        
        # Show actual tokens
        tokens = [tokenizer.id_to_token.get(i, '?') for i in ids]
        print(f"Token breakdown: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
    
    print("\n" + "=" * 50)
    print(f"Tokenizer saved to: {output_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Merges: {len(tokenizer.merges)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
