"""
Enigma Engine - Main Entry Point

Run your AI with one of these commands:
    python run.py --gui     # GUI interface (recommended)
    python run.py --train   # Train from command line
    python run.py --run     # CLI chat interface
    python run.py --serve   # Start API server
    python run.py --build   # Build a new model from scratch

For first-time users, start with --gui
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Enigma Engine - Build and run your own AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --gui                    Launch the GUI (recommended for beginners)
  python run.py --train                  Train with default settings
  python run.py --train --model small    Train a small model
  python run.py --train --epochs 50      Train for 50 epochs
  python run.py --build                  Build new model from scratch
  python run.py --run                    Simple CLI chat
  python run.py --serve                  Start API server on localhost:5000
        """
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--build", action="store_true", help="Build a new model from scratch")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--run", action="store_true", help="Run CLI chat interface")
    parser.add_argument("--gui", action="store_true", help="Start the GUI (recommended)")
    
    # Training options
    parser.add_argument("--model", type=str, default="small",
                        choices=["tiny", "small", "medium", "large", "xl", "xxl"],
                        help="Model size (default: small)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--data", type=str, default=None, help="Path to training data")
    parser.add_argument("--output", type=str, default=None, help="Output model path")
    parser.add_argument("--force", action="store_true", help="Force retrain even if model exists")
    
    args = parser.parse_args()

    # If no arguments, show help and suggest GUI
    if not any([args.train, args.build, args.serve, args.run, args.gui]):
        print("\nWelcome to Enigma Engine!")
        print("=" * 50)
        print("\nTo get started, run one of:")
        print("  python run.py --gui                    # GUI (recommended)")
        print("  python run.py --train                  # Train model")
        print("  python run.py --train --model medium   # Train medium model")
        print("  python run.py --build                  # Build from scratch")
        print("  python run.py --run                    # CLI chat")
        print("  python run.py --serve                  # API server")
        print("\nFor more options: python run.py --help")
        print()
        return

    if args.build:
        # Full build: train tokenizer + model from scratch
        print("\n" + "=" * 60)
        print("ENIGMA ENGINE - FULL BUILD")
        print("=" * 60)
        
        from enigma.core.training import train_model
        from enigma.config import CONFIG
        
        data_path = args.data or Path(CONFIG["data_dir"]) / "data.txt"
        output_path = args.output or Path(CONFIG["models_dir"]) / f"{args.model}_enigma.pth"
        
        print(f"\nBuilding {args.model} model from {data_path}")
        print(f"Output: {output_path}")
        print()
        
        results = train_model(
            data_path=data_path,
            epochs=args.epochs,
            model_size=args.model,
            output_path=output_path,
            train_tokenizer_first=True,
            force=args.force
        )
        
        print(f"\nBuild complete!")
        print(f"  Model saved to: {results.get('model_path', output_path)}")
        print(f"  Final loss: {results.get('final_loss', 'N/A')}")

    if args.train:
        from enigma.core.training import train_model
        from enigma.config import CONFIG
        
        data_path = args.data or Path(CONFIG["data_dir"]) / "data.txt"
        output_path = args.output
        
        print(f"\nTraining {args.model} model...")
        print(f"Data: {data_path}")
        print(f"Epochs: {args.epochs}")
        print()
        
        results = train_model(
            data_path=data_path,
            epochs=args.epochs,
            model_size=args.model,
            output_path=output_path,
            train_tokenizer_first=True,
            force=args.force
        )
        
        if results.get('status') != 'skipped':
            print(f"\nTraining complete!")
            print(f"  Final loss: {results.get('final_loss', 'N/A')}")

    if args.serve:
        from enigma.comms.api_server import create_app
        app = create_app()
        print("\nStarting API server at http://127.0.0.1:5000")
        print("Press Ctrl+C to stop\n")
        app.run(host="127.0.0.1", port=5000, debug=True)

    if args.run:
        from enigma.core.inference import EnigmaEngine
        print("\n" + "=" * 50)
        print("Enigma CLI Chat")
        print("=" * 50)
        print("Type your message and press Enter.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        try:
            engine = EnigmaEngine()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Have you trained the model yet?")
            print("Try: python run.py --train")
            print("Or:  python run.py --gui")
            return
            
        while True:
            try:
                prompt = input("You: ")
                if prompt.strip().lower() in ("quit", "exit", "q"):
                    print("\n[SYSTEM] Goodbye!")
                    break
                if not prompt.strip():
                    continue
                    
                # Generate with streaming
                print("AI: ", end="", flush=True)
                for token in engine.stream_generate(prompt, max_gen=200):
                    print(token, end="", flush=True)
                print("\n")
                
            except KeyboardInterrupt:
                print("\n\n[SYSTEM] Goodbye!")
                break

    if args.gui:
        try:
            from enigma.gui.enhanced_window import run_app
        except ImportError as e:
            print(f"GUI requires PyQt5. Install with: pip install PyQt5")
            print(f"Error: {e}")
            sys.exit(1)
        run_app()


if __name__ == "__main__":
    main()
