#!/usr/bin/env python3
"""
Enigma AI Engine - Main Entry Point

Commands:
    python run.py               Show info and test imports
    python run.py --chat        Simple CLI chat (requires model)
    python run.py --train       Train a model on data
    python run.py --train-tokenizer  Train BPE tokenizer on data
"""

import argparse
import sys
from pathlib import Path

# Import torch early to avoid DLL conflicts on Windows
try:
    import torch
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Enigma AI Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                                   Test imports and show info
  python run.py --gui                             Launch desktop GUI
  python run.py --gui --model models/my.pth       Desktop GUI with model pre-loaded
  python run.py --serve                           Start web UI on port 8080
  python run.py --serve --model models/my.pth     Web UI with model pre-loaded
  python run.py --chat                            CLI chat (requires trained model)
  python run.py --chat --model models/my.pth      Chat with specific model
  python run.py --train data/training.txt         Train model on text data
  python run.py --train data/qa.jsonl --epochs 20 Train with custom epochs
  python run.py --train-tokenizer data/training.txt  Train BPE tokenizer first
        """
    )
    parser.add_argument("--gui", action="store_true", help="Launch desktop GUI")
    parser.add_argument("--serve", action="store_true", help="Start web UI server")
    parser.add_argument("--port", type=int, default=8080, help="Web UI port (default: 8080)")
    parser.add_argument("--chat", action="store_true", help="Simple CLI chat")
    parser.add_argument("--train", type=str, nargs="?", const="auto", default=None,
                        metavar="DATA_PATH", help="Train model (path to .txt or .jsonl)")
    parser.add_argument("--train-tokenizer", type=str, nargs="?", const="auto", default=None,
                        metavar="DATA_PATH", help="Train BPE tokenizer on data")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["pi_zero", "nano", "tiny", "small", "medium", "large"],
                        help="Model size preset for training (default: small)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--vocab-size", type=int, default=8000,
                        help="Vocabulary size for tokenizer training (default: 8000)")
    
    args = parser.parse_args()

    if args.train_tokenizer is not None:
        run_train_tokenizer(args.train_tokenizer, args.vocab_size)
    elif args.train is not None:
        run_train(args.train, args.model, args.model_size, args.epochs, args.batch_size, args.lr)
    elif args.serve:
        run_serve(args.model, args.port)
    elif args.gui:
        run_gui_app(args.model)
    elif args.chat:
        run_chat(args.model)
    else:
        show_info()


def run_gui_app(model_path: str = None):
    """Launch the desktop GUI."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine - Desktop GUI")
    print("=" * 50 + "\n")

    try:
        from enigma_engine.gui.desktop import run_gui
        run_gui(model_path=model_path)
    except ImportError as e:
        print(f"  [ERROR] Missing GUI dependencies: {e}")
        print(f"  Install them:  pip install customtkinter")
        sys.exit(1)
    except Exception as e:
        print(f"  [ERROR] GUI failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_serve(model_path: str = None, port: int = 8080):
    """Start the web UI server."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine - Web UI")
    print("=" * 50 + "\n")

    try:
        from enigma_engine.api.server import run_server
        run_server(host="0.0.0.0", port=port, model_path=model_path)
    except ImportError as e:
        print(f"  [ERROR] Missing web dependencies: {e}")
        print(f"  Install them:  pip install fastapi uvicorn jinja2")
        sys.exit(1)
    except Exception as e:
        print(f"  [ERROR] Server failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def show_info():
    """Show system info and test imports."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine")
    print("=" * 50)
    
    # Test imports
    print("\nTesting imports...")
    
    try:
        from enigma_engine import CONFIG
        print(f"  [OK] CONFIG loaded")
        print(f"       Models dir: {CONFIG.get('models_dir', 'models')}")
    except Exception as e:
        print(f"  [FAIL] CONFIG: {e}")
    
    try:
        from enigma_engine.core import get_hardware
        if get_hardware:
            hw = get_hardware()
            print(f"  [OK] Hardware detection")
            print(f"       Device: {hw.device if hw else 'unknown'}")
        else:
            print(f"  [OK] Hardware module (not initialized)")
    except Exception as e:
        print(f"  [FAIL] Hardware: {e}")
    
    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")
        print(f"       CUDA: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"  [FAIL] PyTorch: {e}")
    
    try:
        from enigma_engine.core import Enigma, create_model
        print(f"  [OK] Model classes loaded")
    except Exception as e:
        print(f"  [FAIL] Model: {e}")
    
    try:
        from enigma_engine.core import EnigmaEngine
        print(f"  [OK] EnigmaEngine loaded")
    except Exception as e:
        print(f"  [FAIL] EnigmaEngine: {e}")
    
    print("\n" + "=" * 50)
    print("  Commands:")
    print("    python run.py --gui                 Launch desktop GUI")
    print("    python run.py --serve              Start web UI")
    print("    python run.py --chat               Start CLI chat")
    print("    python run.py --train <data>        Train a model")
    print("    python run.py --train-tokenizer <data>  Train tokenizer")
    print("=" * 50 + "\n")


def _find_training_data(data_path: str) -> list:
    """Find training data files. Returns list of Path objects."""
    if data_path != "auto":
        p = Path(data_path)
        if not p.exists():
            print(f"  [ERROR] File not found: {data_path}")
            sys.exit(1)
        return [p]
    
    # Auto-discover data files
    data_dir = Path("data")
    candidates = []
    for pattern in ["*.txt", "*.jsonl", "*.json"]:
        candidates.extend(data_dir.glob(pattern))
    
    # Also check root
    for pattern in ["training.txt", "training_data.txt", "train.txt", "data.txt"]:
        p = Path(pattern)
        if p.exists():
            candidates.append(p)
    
    if not candidates:
        print("  [ERROR] No training data found.")
        print("  Put a .txt or .jsonl file in data/ or specify a path:")
        print("    python run.py --train path/to/data.txt")
        sys.exit(1)
    
    return candidates


def run_train_tokenizer(data_path: str, vocab_size: int):
    """Train a BPE tokenizer on data files."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine - Train Tokenizer")
    print("=" * 50 + "\n")
    
    data_files = _find_training_data(data_path)
    print(f"Training data: {[str(f) for f in data_files]}")
    print(f"Target vocab size: {vocab_size}\n")
    
    try:
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        
        # Load all text
        texts = []
        total_chars = 0
        for f in data_files:
            text = f.read_text(encoding="utf-8")
            texts.append(text)
            total_chars += len(text)
            print(f"  Loaded {f.name} ({len(text):,} chars)")
        
        print(f"\n  Total: {total_chars:,} chars across {len(texts)} files")
        print(f"  Training BPE tokenizer...\n")
        
        tokenizer = BPETokenizer()
        tokenizer.train(texts, vocab_size=vocab_size, verbose=True)
        
        # Save next to model files
        output_path = Path("models") / "tokenizer.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(output_path)
        
        # Also save to vocab_model for auto-discovery
        vocab_path = Path("enigma_engine") / "vocab_model" / "bpe_vocab.json"
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(vocab_path)
        
        # Test round-trip
        test_text = "Hello, how are you today?"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"\n  Saved to: {output_path}")
        print(f"  Also saved to: {vocab_path}")
        print(f"  Final vocab size: {tokenizer.vocab_size}")
        print(f"  Total merges: {len(tokenizer.merges)}")
        print(f"\n  Round-trip test:")
        print(f"    Input:   '{test_text}'")
        print(f"    Encoded: {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
        print(f"    Decoded: '{decoded}'")
        print(f"\n  Tokenizer ready!")
        
    except Exception as e:
        print(f"\n  [ERROR] Tokenizer training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_train(data_path: str, model_path: str, model_size: str,
              epochs: int, batch_size: int, lr: float):
    """Train a model on data."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine - Train Model")
    print("=" * 50 + "\n")
    
    data_files = _find_training_data(data_path)
    
    try:
        import torch
        from enigma_engine.core.model import create_model, MODEL_PRESETS
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import get_tokenizer
        
        # Show hardware info
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  VRAM: {vram:.1f} GB")
        
        # Load tokenizer
        print(f"\n  Loading tokenizer...")
        tokenizer = get_tokenizer("auto")
        print(f"  Tokenizer: {type(tokenizer).__name__} (vocab: {tokenizer.vocab_size})")
        
        # Create or load model
        if model_path and Path(model_path).exists():
            print(f"\n  Loading existing model from {model_path}...")
            from enigma_engine.core.model_registry import safe_load_weights
            
            # Load model config from checkpoint
            checkpoint = safe_load_weights(model_path, device=device)
            if "config" in checkpoint:
                from enigma_engine.core.model import ForgeConfig
                config_dict = checkpoint["config"]
                config = ForgeConfig(**{k: v for k, v in config_dict.items() 
                                       if k in ForgeConfig.__dataclass_fields__})
            else:
                config = MODEL_PRESETS.get(model_size, MODEL_PRESETS["small"])
            
            from enigma_engine.core.model import Enigma
            model = Enigma(config=config)
            
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded model: {sum(p.numel() for p in model.parameters()):,} params")
        else:
            # Create new model from preset
            print(f"\n  Creating new '{model_size}' model...")
            
            # Override vocab_size to match tokenizer
            preset = MODEL_PRESETS.get(model_size, MODEL_PRESETS["small"])
            preset.vocab_size = tokenizer.vocab_size
            
            from enigma_engine.core.model import Enigma
            model = Enigma(config=preset)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Model: {model_size} ({param_count:,} params)")
            print(f"  Config: dim={preset.dim}, layers={preset.n_layers}, heads={preset.n_heads}")
        
        model = model.to(device)
        
        # Load training data
        print(f"\n  Loading training data...")
        all_text = []
        for f in data_files:
            text = f.read_text(encoding="utf-8")
            all_text.append(text)
            print(f"    {f.name}: {len(text):,} chars")
        
        training_data = "\n\n".join(all_text)
        
        # Configure training
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            save_every=max(1, epochs // 5),  # Save ~5 checkpoints
            checkpoint_dir="models/checkpoints",
            use_amp=torch.cuda.is_available(),
        )
        
        print(f"\n  Training config:")
        print(f"    Epochs: {epochs}")
        print(f"    Batch size: {batch_size}")
        print(f"    Learning rate: {lr}")
        print(f"    Mixed precision: {config.use_amp}")
        print(f"    Checkpoint dir: {config.checkpoint_dir}")
        
        # Train
        print(f"\n  Starting training...\n")
        
        trainer = Trainer(model, tokenizer, config, device=device)
        
        # Progress callback for CLI
        def on_epoch(epoch, loss):
            print(f"  Epoch {epoch}: loss = {loss:.4f}")
        
        trainer.on_epoch_complete = on_epoch
        state = trainer.train(training_data)
        
        # Save final model
        output_path = Path("models") / f"enigma_{model_size}.pth"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": model.config.vocab_size,
                "dim": model.config.dim,
                "n_layers": model.config.n_layers,
                "n_heads": model.config.n_heads,
                "n_kv_heads": model.config.n_kv_heads,
                "hidden_dim": model.config.hidden_dim,
                "max_seq_len": model.config.max_seq_len,
                "dropout": model.config.dropout,
                "use_rope": model.config.use_rope,
                "use_rms_norm": model.config.use_rms_norm,
                "use_swiglu": model.config.use_swiglu,
            },
            "training_state": {
                "epochs": state.epoch,
                "best_loss": state.best_loss,
                "total_tokens": state.total_tokens,
                "losses": state.training_losses,
            },
        }
        torch.save(save_data, output_path)
        
        # Save tokenizer alongside model
        tok_path = output_path.parent / f"enigma_{model_size}_tokenizer.json"
        if hasattr(tokenizer, 'save'):
            tokenizer.save(tok_path)
        elif hasattr(tokenizer, 'save_vocab'):
            tokenizer.save_vocab(tok_path)
        
        print(f"\n  Training complete!")
        print(f"  Best loss: {state.best_loss:.4f}")
        print(f"  Total tokens: {state.total_tokens:,}")
        print(f"  Model saved: {output_path}")
        print(f"  Tokenizer saved: {tok_path}")
        print(f"\n  To chat: python run.py --chat --model {output_path}")
        
    except Exception as e:
        print(f"\n  [ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_chat(model_path: str = None):
    """Simple CLI chat interface."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine - Chat")
    print("  Type 'quit' to exit")
    print("=" * 50 + "\n")
    
    try:
        from enigma_engine.core import EnigmaEngine
        
        # Try to load engine
        print("Loading model...")
        engine = EnigmaEngine(model_path=model_path) if model_path else EnigmaEngine()
        print("Model loaded!\n")
        
        # Chat loop
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ('quit', 'exit', 'q'):
                    print("Goodbye!")
                    break
                if not user_input:
                    continue
                
                response = engine.chat(user_input)
                print(f"AI: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
                
    except FileNotFoundError as e:
        print(f"\nNo model found. Train a model first or specify --model path.")
        print(f"Error: {e}")
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Make sure you have trained a model or specified a valid model path.")


if __name__ == "__main__":
    main()
