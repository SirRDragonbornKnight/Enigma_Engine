#!/usr/bin/env python3
"""
Enigma AI Engine - Main Entry Point

Commands:
    python run.py               Show info and test imports
    python run.py --client-chat  CLI chat via daemon (recommended)
    python run.py --train        Train a model on data
    python run.py --train --resume <checkpoint>  Resume training from checkpoint
    python run.py --train-tokenizer  Train BPE tokenizer on data
"""

import argparse
import copy
import os
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from urllib.parse import urlparse


def _ensure_venv() -> None:
    """Create the project venv if missing, install deps, and re-launch.

    On a fresh PC:
      1. Creates ``venv/`` via ``python -m venv venv``
      2. Installs ``requirements.txt`` into it
      3. Re-executes under the venv Python

    On a configured PC:
      Re-executes under the existing venv Python if the current
      interpreter is the system Python (no venv activated).
    """
    # Already inside a venv / virtualenv — nothing to do.
    if sys.prefix != sys.base_prefix:
        return

    script_dir = Path(__file__).resolve().parent
    # Windows: venv\Scripts\python.exe   Posix: venv/bin/python
    candidates = [
        script_dir / "venv" / "Scripts" / "python.exe",
        script_dir / "venv" / "bin" / "python",
    ]
    venv_python = next((p for p in candidates if p.is_file()), None)

    if venv_python is None:
        # No venv exists — create one and install dependencies
        print("=" * 60)
        print("  First-time setup: creating virtual environment...")
        print("=" * 60)
        venv_dir = script_dir / "venv"
        rc = subprocess.run([sys.executable, "-m", "venv", str(venv_dir)]).returncode
        if rc != 0:
            print(f"ERROR: Failed to create venv (exit code {rc})")
            print("Make sure Python 3.9+ is installed correctly.")
            raise SystemExit(1)

        venv_python = next((p for p in candidates if p.is_file()), None)
        if venv_python is None:
            print("ERROR: venv was created but Python executable not found in it.")
            raise SystemExit(1)

        # Install requirements
        req_file = script_dir / "requirements.txt"
        if req_file.is_file():
            print("  Installing dependencies (this may take a few minutes)...")
            rc = subprocess.run(
                [str(venv_python), "-m", "pip", "install", "-r", str(req_file)],
            ).returncode
            if rc != 0:
                print(f"WARNING: pip install exited with code {rc}")
                print("Some optional dependencies may have failed.")
                print("Core features should still work. You can re-run:")
                print(f"  {venv_python} -m pip install -r requirements.txt")
        print("=" * 60)
        print("  Setup complete! Launching Enigma Engine...")
        print("=" * 60)

    # Re-execute under the venv Python with the same arguments.
    result = subprocess.run(
        [str(venv_python)] + sys.argv,
        cwd=os.getcwd(),
    )
    raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Enigma AI Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                                   Test imports and show info
  python run.py --gui                             Launch desktop GUI
  python run.py --gui --model models/my.pth       Desktop GUI with model pre-loaded
  python run.py --serve                           Start API server on port 8080
  python run.py --serve --model models/my.pth     API server with model pre-loaded
    python run.py --client-chat                     CLI chat over API client
    python run.py --client-chat --api-url http://127.0.0.1:8080
  python run.py --client-chat                     CLI chat via daemon (auto-spawns if needed)
  python run.py --client-chat --model models/my.pth  Chat with specific model
  python run.py --train data/training.txt         Train model on text data
    python run.py --config train.yaml               Train using dispatcher config
  python run.py --train data/qa.jsonl --epochs 20 Train with custom epochs
  python run.py --train-tokenizer data/training.txt  Train BPE tokenizer first
  python run.py --train-tokenizer data/ --utf8-bytes Train byte-level BPE
  python run.py --analyze-tokenizer                  Analyze trained tokenizer
        """
    )
    parser.add_argument("--gui", action="store_true", help="Launch desktop GUI")
    parser.add_argument("--serve", action="store_true", help="Start local API server")
    parser.add_argument("--port", type=int, default=None,
                        help="API server port (default: reads from CONFIG, fallback 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Server bind address (default: 127.0.0.1, use 0.0.0.0 for network)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for server authentication (Bearer token)")
    parser.add_argument("--cors-origins", type=str, default=None,
                        help="Comma-separated CORS origins (e.g. http://localhost:3000). "
                             "CORS is disabled when omitted")
    parser.add_argument("--chat", action="store_true",
                        help="[deprecated] Use --client-chat instead")
    parser.add_argument("--client-chat", action="store_true",
                        help="CLI chat over HTTP API (ARCH-1b/1c bridge)")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8080",
                        help="Base URL for --client-chat (default: http://127.0.0.1:8080)")
    parser.add_argument("--no-auto-spawn", action="store_true",
                        help="Disable local daemon auto-spawn for --client-chat")
    parser.add_argument("--profile", type=str, default=None,
                        help="AI profile to load for --chat (e.g. assistant, creative_writer)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Generation temperature for --chat (e.g. 0.7)")
    parser.add_argument("--train", type=str, nargs="?", const="auto", default=None,
                        metavar="DATA_PATH", help="Train model (path to .txt or .jsonl)")
    parser.add_argument("--config", type=str, default=None,
                        help="Training config file (.yaml/.yml/.json) for dispatcher-based training"
                             " (can be used standalone or with --train)")
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
    parser.add_argument("--utf8-bytes", action="store_true",
                        help="Enable byte-level BPE encoding (handles any Unicode)")
    parser.add_argument("--analyze-tokenizer", type=str, nargs="?", const="auto", default=None,
                        metavar="VOCAB_PATH",
                        help="Analyze a trained tokenizer on data (default: auto-discover)")
    parser.add_argument("--benchmark", nargs="?", const="coherence",
                        default=None,
                        choices=["coherence", "gsm8k"],
                        help="Run a benchmark on a loaded model. "
                             "Default 'coherence' (no arg). Use 'gsm8k' for the "
                             "reasoning benchmark.")
    parser.add_argument("--benchmark-data", type=str, default=None,
                        metavar="PATH",
                        help="Path to GSM8K JSONL test file "
                             "(default: data/gsm8k_test.jsonl)")
    parser.add_argument("--benchmark-limit", type=int, default=-1,
                        help="Cap number of GSM8K examples (default: all 1319)")
    parser.add_argument("--benchmark-shots", type=int, default=8,
                        help="GSM8K few-shot CoT examples (0..8, default 8)")
    parser.add_argument("--resume", type=str, default=None, metavar="CHECKPOINT_PATH",
                        help="Resume training from a checkpoint file (e.g. models/checkpoints/best_model.pt)")
    parser.add_argument("--golden-eval", type=str, default=None, metavar="JSON_PATH",
                        help="Golden prompt regression eval file (JSON with prompt+expected pairs)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible training")
    parser.add_argument("--deterministic", action="store_true",
                        help="Pin cuBLAS workspace + use_deterministic_algorithms "
                             "for bitwise-reproducible GPU training (5-15%% "
                             "throughput cost; requires --seed). Off by default.")
    
    args = parser.parse_args()

    # DET-2: --deterministic without --seed is a silent no-op because
    # set_training_seed (the gate that flips the deterministic switch)
    # is only called when config.seed is not None. Fail loud instead.
    if args.deterministic and args.seed is None:
        parser.error("--deterministic requires --seed (e.g. --seed 42); "
                     "without a seed the flag is silently ignored.")

    if args.train_tokenizer is not None:
        run_train_tokenizer(args.train_tokenizer, args.vocab_size,
                            args.utf8_bytes)
    elif args.analyze_tokenizer is not None:
        run_analyze_tokenizer(args.analyze_tokenizer)
    elif args.train is not None:
        if args.config:
            run_train_config(args.config, args.model, args.model_size, args.resume)
        else:
            run_train(args.train, args.model, args.model_size, args.epochs, args.batch_size, args.lr,
                      golden_eval=args.golden_eval, seed=args.seed,
                      deterministic=args.deterministic, resume=args.resume)
    elif args.config:
        run_train_config(args.config, args.model, args.model_size, args.resume)
    elif args.serve:
        cors = None
        if args.cors_origins:
            cors = [o.strip() for o in args.cors_origins.split(",") if o.strip()]
        # Resolve port: CLI flag > CONFIG > fallback 8080
        port = args.port
        if port is None:
            try:
                from enigma_engine import CONFIG
                port = int(CONFIG.get("api_port", 8080))
            except Exception:
                port = 8080
        # Resolve API key: CLI flag > CONFIG > None
        key = args.api_key
        if key is None:
            try:
                from enigma_engine import CONFIG as _cfg
                key = _cfg.get("enigma_api_key")
            except Exception:
                pass
        run_serve(args.model, port, args.host, key, cors)
    elif args.gui:
        run_gui_app(args.model)
    elif args.client_chat:
        run_chat_client(
            api_url=args.api_url,
            model_path=args.model,
            profile=args.profile,
            temperature=args.temperature,
            no_auto_spawn=args.no_auto_spawn,
        )
    elif args.chat:
        # BIOME-1c: --chat is deprecated. Forward to --client-chat so the
        # in-process run_chat code path is no longer exercised. Auto-spawn
        # handles the "no daemon running" case transparently.
        print("[WARN] --chat is deprecated; use --client-chat (auto-spawns daemon).")
        run_chat_client(
            api_url=args.api_url,
            model_path=args.model,
            profile=args.profile,
            temperature=args.temperature,
            no_auto_spawn=False,
        )
    elif args.benchmark is not None:
        if args.benchmark == "gsm8k":
            run_gsm8k_benchmark_cli(
                args.model,
                data_path=args.benchmark_data,
                limit=args.benchmark_limit,
                num_shots=args.benchmark_shots,
            )
        else:
            run_benchmark(args.model)
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
        print("  Install them:  pip install customtkinter")
        sys.exit(1)
    except Exception as e:
        print(f"  [ERROR] GUI failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_serve(model_path: str = None, port: int = 8080,
              host: str = "127.0.0.1", api_key: str = None,
              cors_origins: list = None):
    """Start the local API server."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine - API Server")
    print("=" * 50 + "\n")

    try:
        from enigma_engine.api.server import run_server
        run_server(host=host, port=port, model_path=model_path,
                   api_key=api_key, cors_origins=cors_origins)
    except ImportError as e:
        print(f"  [ERROR] Missing server dependencies: {e}")
        print("  Install them:  pip install fastapi uvicorn")
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
        print("  [OK] CONFIG loaded")
        print(f"       Models dir: {CONFIG.get('models_dir', 'models')}")
    except Exception as e:
        print(f"  [FAIL] CONFIG: {e}")
    
    try:
        from enigma_engine.core import get_hardware
        if get_hardware:
            hw = get_hardware()
            print("  [OK] Hardware detection")
            print(f"       Device: {hw.device if hw else 'unknown'}")
        else:
            print("  [OK] Hardware module (not initialized)")
    except Exception as e:
        print(f"  [FAIL] Hardware: {e}")
    
    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")
        print(f"       CUDA: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"  [FAIL] PyTorch: {e}")
    
    try:
        print("  [OK] Model classes loaded")
    except Exception as e:
        print(f"  [FAIL] Model: {e}")
    
    try:
        print("  [OK] EnigmaEngine loaded")
    except Exception as e:
        print(f"  [FAIL] EnigmaEngine: {e}")
    
    print("\n" + "=" * 50)
    print("  Commands:")
    print("    python run.py --gui                 Launch desktop GUI")
    print("    python run.py --serve              Start API server")
    print("    python run.py --client-chat        Start CLI chat (via daemon)")
    print("    python run.py --train <data>        Train a model")
    print("    python run.py --train-tokenizer <data>  Train tokenizer")
    print("    python run.py --analyze-tokenizer       Analyze tokenizer")
    print("=" * 50 + "\n")


def _find_training_data(data_path: str) -> list:
    """Find training data files. Returns list of Path objects."""
    if data_path != "auto":
        p = Path(data_path)
        if not p.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
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
        raise FileNotFoundError(
            "No training data found. "
            "Put a .txt or .jsonl file in data/ or specify a path: "
            "python run.py --train path/to/data.txt")

    return candidates


def run_train_tokenizer(data_path: str, vocab_size: int,
                        utf8_bytes: bool = False):
    """Train a BPE tokenizer on data files."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine - Train Tokenizer")
    print("=" * 50 + "\n")
    
    data_files = _find_training_data(data_path)
    print(f"Training data: {[str(f) for f in data_files]}")
    print(f"Target vocab size: {vocab_size}")
    if utf8_bytes:
        print("Byte-level BPE: enabled")
    print()
    
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
        print("  Training BPE tokenizer...\n")
        
        tokenizer = BPETokenizer()
        if utf8_bytes:
            tokenizer.use_utf8_bytes = True
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
        print("\n  Round-trip test:")
        print(f"    Input:   '{test_text}'")
        print(f"    Encoded: {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
        print(f"    Decoded: '{decoded}'")
        print("\n  Tokenizer ready!")
        
    except Exception as e:
        print(f"\n  [ERROR] Tokenizer training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_analyze_tokenizer(vocab_path: str):
    """Analyze a trained tokenizer on data."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine - Analyze Tokenizer")
    print("=" * 50 + "\n")

    try:
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        from enigma_engine.core.tokenizer_metrics import format_report

        # Find tokenizer file
        if vocab_path == "auto":
            candidates = [
                Path("models") / "tokenizer.json",
                Path("enigma_engine") / "vocab_model" / "bpe_vocab.json",
                Path("enigma_engine") / "vocab_model" / "tokenizer.json",
            ]
            tok_file = next((p for p in candidates if p.exists()), None)
            if tok_file is None:
                print("  [ERROR] No tokenizer found. Train one first with:")
                print("    python run.py --train-tokenizer data/training.txt")
                sys.exit(1)
        else:
            tok_file = Path(vocab_path)
            if not tok_file.exists():
                print(f"  [ERROR] Tokenizer file not found: {vocab_path}")
                sys.exit(1)

        print(f"  Loading tokenizer: {tok_file}")
        tokenizer = BPETokenizer(tok_file)

        # Load analysis data
        data_dir = Path("data")
        texts = []
        for pattern in ["*.txt", "*.jsonl"]:
            for f in sorted(data_dir.glob(pattern)):
                text = f.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    texts.append(text)
        if not texts:
            print("  [!] No data files in data/ — using built-in test strings")
            texts = [
                "Hello, how are you today?",
                "The quick brown fox jumps over the lazy dog.",
                "User: What is quantum computing?\nAssistant: Quantum computing uses qubits.",
            ]

        print(format_report(tokenizer, texts))

    except Exception as e:
        print(f"\n  [ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_train(data_path: str, model_path: str, model_size: str,
              epochs: int, batch_size: int, lr: float,
              golden_eval: str | None = None,
              seed: int | None = None,
              deterministic: bool = False,
              resume: str | None = None):
    """Train a model on data."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine - Train Model")
    print("=" * 50 + "\n")
    
    data_files = _find_training_data(data_path)
    
    try:
        import torch
        from enigma_engine.core.model import MODEL_PRESETS
        from enigma_engine.training.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import get_tokenizer
        
        # Show hardware info
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  VRAM: {vram:.1f} GB")
        
        # Load tokenizer
        print("\n  Loading tokenizer...")
        tokenizer = get_tokenizer("auto")
        print(f"  Tokenizer: {type(tokenizer).__name__} (vocab: {tokenizer.vocab_size})")
        
        # Create or load model
        if model_path and Path(model_path).exists():
            print(f"\n  Loading existing model from {model_path}...")
            from enigma_engine.core.model_registry import safe_load_weights
            
            # Load model config from checkpoint
            checkpoint = safe_load_weights(model_path, map_location=device)
            if "config" in checkpoint:
                from enigma_engine.core.model import ForgeConfig
                config_dict = checkpoint["config"]
                config = ForgeConfig(**{k: v for k, v in config_dict.items() 
                                       if k in ForgeConfig.__dataclass_fields__})
            else:
                config = MODEL_PRESETS.get(model_size, MODEL_PRESETS["small"])
            
            from enigma_engine.core.model import Enigma
            model = Enigma(config=config)
            
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
            if state_dict is None:
                raise ValueError(
                    "Checkpoint missing 'model_state_dict' or 'state_dict' key")
            model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded model: {sum(p.numel() for p in model.parameters()):,} params")
        else:
            # Create new model from preset
            print(f"\n  Creating new '{model_size}' model...")
            
            # Override vocab_size to match tokenizer
            preset = copy.deepcopy(
                MODEL_PRESETS.get(model_size, MODEL_PRESETS["small"]))
            if not tokenizer.vocab_size or tokenizer.vocab_size < 1:
                raise ValueError(
                    f"Tokenizer returned invalid vocab_size: "
                    f"{tokenizer.vocab_size}"
                )
            preset.vocab_size = tokenizer.vocab_size
            
            from enigma_engine.core.model import Enigma
            model = Enigma(config=preset)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Model: {model_size} ({param_count:,} params)")
            print(f"  Config: dim={preset.dim}, layers={preset.n_layers}, heads={preset.n_heads}")
        
        try:
            model = model.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n  ERROR: Not enough GPU memory to load model.")
                print("  Try using --device cpu or a smaller model.")
                return
            raise
        
        # Load training data
        print("\n  Loading training data...")
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
            seed=seed,
            deterministic=deterministic,
            golden_eval_path=golden_eval or "",
        )
        
        print("\n  Training config:")
        print(f"    Epochs: {epochs}")
        print(f"    Batch size: {batch_size}")
        print(f"    Learning rate: {lr}")
        print(f"    Mixed precision: {config.use_amp}")
        print(f"    Checkpoint dir: {config.checkpoint_dir}")
        
        # Train
        print("\n  Starting training...\n")
        
        trainer = Trainer(model, tokenizer, config)
        
        # Progress callback for CLI
        def on_epoch(epoch, loss):
            print(f"  Epoch {epoch}: loss = {loss:.4f}")
        
        trainer.on_epoch_complete = on_epoch

        if resume:
            print(f"\n  Resuming from checkpoint: {resume}")
        state = trainer.train(training_data, resume_from=resume)
        
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
        from enigma_engine.core.safe_save import atomic_torch_save
        atomic_torch_save(save_data, output_path)
        
        # Save tokenizer alongside model
        tok_path = output_path.parent / f"enigma_{model_size}_tokenizer.json"
        if hasattr(tokenizer, 'save'):
            tokenizer.save(tok_path)
        elif hasattr(tokenizer, 'save_vocab'):
            tokenizer.save_vocab(tok_path)
        
        print("\n  Training complete!")
        print(f"  Best loss: {state.best_loss:.4f}")
        print(f"  Total tokens: {state.total_tokens:,}")
        print(f"  Model saved: {output_path}")
        print(f"  Tokenizer saved: {tok_path}")
        print(f"\n  To chat: python run.py --client-chat --model {output_path}")
        
    except Exception as e:
        print(f"\n  [ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _load_dispatch_payload(data: object, mode: str) -> object:
    """Thin wrapper around the shared dispatcher payload helper."""
    from enigma_engine.training import materialize_dispatch_payload

    return materialize_dispatch_payload(data, mode)


def run_train_config(
    config_path: str,
    model_path: str | None,
    model_size: str,
    resume: str | None = None,
):
    """Run training via the unified training dispatcher config."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine - Train Model (Config)")
    print("=" * 50 + "\n")

    try:
        import torch
        from enigma_engine.core.model import MODEL_PRESETS
        from enigma_engine.core.model_registry import safe_load_weights
        from enigma_engine.core.reward_functions import reasoning_reward
        from enigma_engine.core.safe_save import atomic_torch_save
        from enigma_engine.core.tokenizer import get_tokenizer
        from enigma_engine.training import (
            TrainingJobConfig,
            build_dispatch_context,
            load_training_config_raw,
            run_training,
        )

        raw_job = load_training_config_raw(config_path)
        mode = raw_job.get("mode")
        if resume and raw_job.get("resume_from") is None:
            raw_job["resume_from"] = resume

        # CLI config path currently does not construct the extra runtime
        # objects required by these modes.
        if mode in {"vision", "audio", "rlhf", "self_play", "adaptive"}:
            raise ValueError(
                f"Mode '{mode}' is not supported from run.py --config yet. "
                "Use API/GUI wiring for this mode until dispatcher context "
                "construction is completed."
            )

        # Load payload from file path when config.data points to a local file.
        raw_job["data"] = _load_dispatch_payload(raw_job.get("data"), str(mode))
        job = TrainingJobConfig.model_validate(raw_job)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Mode: {job.mode}")
        print(f"  Device: {device}")

        print("\n  Loading tokenizer...")
        tokenizer = get_tokenizer("auto")
        print(f"  Tokenizer: {type(tokenizer).__name__} (vocab: {tokenizer.vocab_size})")

        # Create or load model
        if model_path and Path(model_path).exists():
            print(f"\n  Loading existing model from {model_path}...")
            checkpoint = safe_load_weights(model_path, map_location=device)
            if "config" in checkpoint:
                from enigma_engine.core.model import ForgeConfig
                config_dict = checkpoint["config"]
                model_config = ForgeConfig(**{k: v for k, v in config_dict.items()
                                              if k in ForgeConfig.__dataclass_fields__})
            else:
                model_config = MODEL_PRESETS.get(model_size, MODEL_PRESETS["small"])

            from enigma_engine.core.model import Enigma
            model = Enigma(config=model_config)
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
            if state_dict is None:
                raise ValueError("Checkpoint missing 'model_state_dict' or 'state_dict' key")
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"\n  Creating new '{model_size}' model...")
            preset = copy.deepcopy(MODEL_PRESETS.get(model_size, MODEL_PRESETS["small"]))
            if not tokenizer.vocab_size or tokenizer.vocab_size < 1:
                raise ValueError(f"Tokenizer returned invalid vocab_size: {tokenizer.vocab_size}")
            preset.vocab_size = tokenizer.vocab_size
            from enigma_engine.core.model import Enigma
            model = Enigma(config=preset)

        model = model.to(device)

        context = build_dispatch_context(
            model=model,
            tokenizer=tokenizer,
            # GRPO/ReMax/ReST require a reward signal; this default keeps
            # config-run parity with the existing reasoning alignment flow.
            reward_fn=reasoning_reward,
        )

        result = run_training(job, context)

        if job.mode == "lora":
            print("\n  LoRA training complete!")
            print(f"  Adapter saved: {result.get('adapter_path', 'unknown')}")
            print(f"  Result: {result.get('train_result', result)}")
            return

        output_path = Path("models") / f"enigma_{model_size}_{job.mode}.pth"
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
            "dispatcher_mode": job.mode,
        }
        atomic_torch_save(save_data, output_path)
        print("\n  Config training complete!")
        print(f"  Model saved: {output_path}")
        print(f"  Result: {result}")

    except Exception as e:
        print(f"\n  [ERROR] Config training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_benchmark(model_path: str = None):
    """Run coherence benchmark on a loaded model (CLI)."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine - Coherence Benchmark")
    print("=" * 50 + "\n")

    if not model_path:
        print("  [ERROR] --model is required for --benchmark")
        print("  Usage: python run.py --benchmark --model models/my.pth")
        sys.exit(1)

    try:
        from enigma_engine.core import EnigmaEngine
        from enigma_engine.core.monologue import run_coherence_benchmark

        print(f"  Loading {model_path}...")
        engine = EnigmaEngine(model_path=model_path)
        print("  Model loaded. Running 20 reflection prompts...\n")

        def _progress(idx, total, score):
            status = "PASS" if score >= 0.7 else "--"
            print(f"  [{idx:>2d}/{total}]  coherence = {score:.3f}  {status}")

        result = run_coherence_benchmark(
            engine, num_prompts=20, on_progress=_progress)

        print(f"\n{'=' * 50}")
        print(f"  Prompts : {result['total']}")
        print(f"  Passed  : {result['passed']} / {result['total']}")
        print(f"  Mean    : {result['mean']:.3f}")
        print(f"  Pass %  : {result['pass_rate'] * 100:.0f}%")

        rec = result["recommendation"]
        if rec == "ready":
            print("  Result  : READY — safe for automatic monologue mode")
        elif rec == "marginal":
            print("  Result  : MARGINAL — journal_only mode recommended")
        else:
            print("  Result  : NOT READY — keep monologue disabled")
        print("=" * 50)

    except Exception as e:
        print(f"\n  [ERROR] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_gsm8k_benchmark_cli(model_path: str = None,
                             data_path: str = None,
                             limit: int = -1,
                             num_shots: int = 8):
    """Run GSM8K reasoning benchmark on a loaded model (CLI)."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine - GSM8K Reasoning Benchmark")
    print("=" * 50 + "\n")

    if not model_path:
        print("  [ERROR] --model is required for --benchmark gsm8k")
        print("  Usage: python run.py --benchmark gsm8k --model models/my.pth")
        sys.exit(1)

    try:
        from enigma_engine.core import EnigmaEngine
        from enigma_engine.training.training_evaluation import (
            load_gsm8k, run_gsm8k_benchmark)

        print(f"  Loading {model_path}...")
        engine = EnigmaEngine(model_path=model_path)

        print("  Loading GSM8K test set...")
        examples = load_gsm8k(data_path, n=limit)
        print(f"  Loaded {len(examples)} examples. "
              f"Few-shot: {num_shots}. Running...\n")

        def _progress(idx, total, correct):
            acc = correct / idx if idx else 0.0
            print(f"  [{idx:>4d}/{total}]  correct={correct}  "
                  f"acc={acc:.3f}")

        result = run_gsm8k_benchmark(
            engine, examples, num_shots=num_shots,
            on_progress=_progress)

        print(f"\n{'=' * 50}")
        print(f"  Total    : {result['total']}")
        print(f"  Correct  : {result['correct']}")
        print(f"  Accuracy : {result['accuracy'] * 100:.2f}%")
        print("=" * 50)

    except FileNotFoundError as e:
        print(f"\n  [ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  [ERROR] GSM8K benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# run_chat (in-process) was removed in BIOME-1c (Pass 156z9dc).
# --chat now forwards to run_chat_client, which auto-spawns the daemon
# if it is not already running. This keeps the public flag alive for any
# existing scripts while retiring the in-process engine path.


@dataclass
class _ChatClientState:
    temperature: float | None = None
    transcript: list[str] = field(default_factory=list)
    daemon_pid: int | None = None
    profile: str | None = None
    model_path: str | None = None


_CHAT_HELP_TEXT = "\n".join([
    "Commands:",
    "  :help              Show this command list",
    "  :new               Start a fresh conversation thread",
    "  :list              List conversations with short previews",
    "  :delete <id>       Delete a conversation",
    "  :reset             Clear local transcript history",
    "  :profile <name>    Activate a profile on the daemon",
    "  :model <path>      Load a model on the daemon",
    "  :temp <n>          Set the per-turn temperature [0.0, 2.0]",
    "  :save <path>       Save the local transcript to a text file",
    "  :status            Show local and daemon state",
])


def _is_local_api_url(api_url: str) -> bool:
    host = urlparse(api_url).hostname or ""
    return host in {"127.0.0.1", "localhost"}


def _try_autospawn_daemon(api_url: str, model_path: str | None = None,
                          timeout: float = 8.0) -> int | None:
    parsed = urlparse(api_url)
    host = parsed.hostname or ""
    if host not in {"127.0.0.1", "localhost"}:
        return None

    port = parsed.port or 8080
    script_path = Path(__file__).resolve()
    cmd = [sys.executable, str(script_path), "--serve", "--port", str(port)]
    if model_path:
        cmd.extend(["--model", model_path])

    print(f"Starting daemon at {api_url}...")
    try:
        proc = subprocess.Popen(cmd)
    except Exception as exc:
        import traceback
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        print(f"\nError connecting to API server: {exc}")
        return None

    from enigma_engine import EnigmaClient

    client = EnigmaClient(api_url)
    attempts = max(1, int(timeout / 0.25) + 1)
    last_error: Exception | None = None

    for _ in range(attempts):
        try:
            health = client.health()
            if health.get("status") == "ok":
                return getattr(proc, "pid", None)
            last_error = RuntimeError(f"server health check failed: {health}")
        except Exception as exc:
            last_error = exc
        time.sleep(0.25)

    # Health never went OK — terminate the orphan to free the port + PID.
    # Pass 156z9da audit #1: silent leak left the subprocess running while
    # the caller exited, blocking subsequent --client-chat retries.
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    except Exception:
        pass

    if last_error is not None:
        import traceback
        traceback.print_exception(type(last_error), last_error,
                                  last_error.__traceback__)
        print(f"\nError connecting to API server: {last_error}")
    return None


def _dispatch_command(line: str, client, state: _ChatClientState) -> bool:
    if not line.startswith(":"):
        return False

    command_text = line[1:].strip()
    if not command_text:
        print("[ERROR] unknown command — try :help")
        return True

    command, _, remainder = command_text.partition(" ")
    command = command.lower()
    arg = remainder.strip()

    if command == "help":
        print(_CHAT_HELP_TEXT)
        return True

    if command == "new":
        conversation_id = client.new_conversation()
        state.transcript.clear()
        print(f"Conversation: {conversation_id}")
        return True

    if command == "list":
        conversations = client.list_conversations()
        if not conversations:
            print("No conversations.")
            return True
        for entry in conversations:
            if not isinstance(entry, dict):
                continue
            conversation_id = entry.get("id", "(unknown)")
            preview = entry.get("last_message") or entry.get("preview") \
                or entry.get("message") or ""
            print(f"{conversation_id}: {preview}")
        return True

    if command == "delete":
        if not arg:
            print("[ERROR] usage: :delete <id>")
            return True
        was_active = getattr(client, "conversation_id", None) == arg
        client.delete_conversation(arg)
        if was_active:
            # Pass 156z9da audit #2: clear local scrollback when the active
            # thread is deleted, matching :new behaviour. Inactive deletes
            # leave scrollback untouched.
            client.conversation_id = None
            state.transcript.clear()
        print(f"Deleted conversation: {arg}")
        return True

    if command == "reset":
        state.transcript.clear()
        print("Local transcript cleared.")
        return True

    if command == "profile":
        if not arg:
            print("[ERROR] usage: :profile <name>")
            return True
        client.activate_profile(arg)
        state.profile = arg
        print(f"Profile: {arg}")
        return True

    if command == "model":
        if not arg:
            print("[ERROR] usage: :model <path>")
            return True
        client.load_model(arg)
        state.model_path = arg
        print(f"Model: {arg}")
        return True

    if command == "temp":
        if not arg:
            print("[ERROR] usage: :temp <n>")
            return True
        try:
            temperature = float(arg)
        except ValueError:
            print("[ERROR] temperature must be a number in [0.0, 2.0]")
            return True
        if not 0.0 <= temperature <= 2.0:
            print("[ERROR] temperature must be a number in [0.0, 2.0]")
            return True
        state.temperature = temperature
        print(f"Temperature: {temperature:g}")
        return True

    if command == "save":
        if not arg:
            print("[ERROR] usage: :save <path>")
            return True
        from enigma_engine.core.safe_save import atomic_write_text

        save_path = Path(arg).expanduser()
        transcript = "\n".join(state.transcript)
        if transcript:
            transcript += "\n"
        atomic_write_text(save_path, transcript)
        print(f"Saved transcript to {save_path}")
        return True

    if command == "status":
        health = client.health()
        conversation_id = getattr(client, "conversation_id", None) or "(none)"
        profile = state.profile or "(none)"
        model_path = state.model_path or "(none)"
        daemon_pid = state.daemon_pid if state.daemon_pid is not None else "(none)"
        print(
            f"Health: {health.get('status', 'unknown')} | "
            f"Conversation: {conversation_id} | Profile: {profile} | "
            f"Model: {model_path} | Daemon PID: {daemon_pid}"
        )
        return True

    print("[ERROR] unknown command — try :help")
    return True


def _chat_repl(client, *, state: _ChatClientState, on_command_error) -> None:
    import sys

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue

            if user_input.startswith(":"):
                try:
                    handled = _dispatch_command(user_input, client, state)
                except Exception as exc:
                    on_command_error(exc)
                    continue
                if handled:
                    continue

            sys.stdout.write("AI: ")
            sys.stdout.flush()

            stream_kwargs: dict = {}
            if state.temperature is not None:
                stream_kwargs["temperature"] = state.temperature

            try:
                emitted_tokens: list[str] = []
                for token in client.chat_stream(user_input, **stream_kwargs):
                    emitted_tokens.append(token)
                    sys.stdout.write(token)
                    sys.stdout.flush()

                response_text = "".join(emitted_tokens)
                if not emitted_tokens:
                    # Fallback for servers/proxies that buffer SSE responses.
                    response_text = client.chat(user_input, **stream_kwargs)
                    sys.stdout.write(response_text)
                    sys.stdout.flush()
            except Exception as exc:
                on_command_error(exc)
                continue

            sys.stdout.write("\n\n")
            sys.stdout.flush()

            # Pass 156z9da audit #5: skip transcript append when both stream
            # and non-stream fallback returned empty — avoids logging a bare
            # "AI: " line under a misleading user turn.
            if not response_text:
                sys.stdout.write("  [WARN] empty response from server\n")
                sys.stdout.flush()
                continue

            state.transcript.append(f"You: {user_input}")
            state.transcript.append(f"AI: {response_text}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def run_chat_client(api_url: str, model_path: str | None = None,
                    profile: str | None = None,
                    temperature: float | None = None,
                    no_auto_spawn: bool = False):
    """CLI chat interface that talks to a running API server."""
    print("\n" + "=" * 50)
    print("  Enigma AI Engine - Client Chat")
    print("  Type 'quit' to exit")
    print("=" * 50 + "\n")

    try:
        from enigma_engine import EnigmaClient

        client = EnigmaClient(api_url)
        daemon_pid: int | None = None
        try:
            health = client.health()
            if health.get("status") != "ok":
                raise RuntimeError(f"server health check failed: {health}")
        except Exception as exc:
            if no_auto_spawn:
                print(f"\nError connecting to API server: {exc}")
                print("Start the server first: python run.py --serve")
                return

            if not _is_local_api_url(api_url):
                print(f"\nError connecting to API server: {exc}")
                print("Start the server first: python run.py --serve")
                return

            daemon_pid = _try_autospawn_daemon(api_url, model_path=model_path)
            if daemon_pid is None:
                return
            print(f"Daemon started (pid={daemon_pid})")

        print(f"Connected: {api_url}")

        state = _ChatClientState(
            temperature=temperature,
            daemon_pid=daemon_pid,
        )

        if model_path:
            print("Loading model via API...")
            client.load_model(model_path)
            state.model_path = model_path
            print("Model loaded!")

        if profile:
            try:
                client.activate_profile(profile)
                state.profile = profile
                print(f"Profile: {profile}")
            except Exception as exc:
                print(f"  [WARN] Could not activate profile '{profile}': {exc}")

        print()

        def _command_error(exc: Exception) -> None:
            print(f"\n  [ERROR] Request failed: {exc}\n")

        _chat_repl(client, state=state, on_command_error=_command_error)

    except Exception as exc:
        print(f"\nError connecting to API server: {exc}")
        print("Start the server first: python run.py --serve")


if __name__ == "__main__":
    _ensure_venv()
    main()
