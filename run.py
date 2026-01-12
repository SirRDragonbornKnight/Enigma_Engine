"""
AI Tester - Main Entry Point

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
import os
from pathlib import Path


def _print_startup_banner():
    """Print a clean startup message."""
    print("=" * 50)
    print("  AI Tester - Starting...")
    print("  (Audio warnings below are normal on headless Pi)")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="AI Tester - Build and run your own AI",
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
  python run.py --background             Run in system tray only (background mode)
        """
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--build", action="store_true", help="Build a new model from scratch")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--run", action="store_true", help="Run CLI chat interface")
    parser.add_argument("--gui", action="store_true", help="Start the GUI (recommended)")
    parser.add_argument("--web", action="store_true", help="Start web dashboard")
    parser.add_argument("--background", action="store_true", help="Run in system tray (background mode)")
    
    # Multi-instance options
    parser.add_argument("--instance", type=str, default=None, help="Instance ID (for multi-instance)")
    parser.add_argument("--new-instance", action="store_true", help="Force new instance")
    
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
    if not any([args.train, args.build, args.serve, args.run, args.gui, args.web, args.background]):
        print("\n" + "=" * 60)
        print("  AI TESTER - Build Your Own AI")
        print("=" * 60)
        print("\nQuick Start Options:\n")
        print("  python run.py --gui")
        print("    -> Launch GUI (recommended for beginners)")
        print("\n  python run.py --background")
        print("    -> Run in system tray (always available, lightweight)")
        print("\n  python run.py --train")
        print("    -> Train a model with default settings")
        print("\n  python run.py --train --model medium")
        print("    -> Train a medium-sized model")
        print("\n  python run.py --run")
        print("    -> Start CLI chat interface")
        print("\n  python run.py --serve")
        print("    -> Start API server on localhost:5000")
        print("\n  python run.py --web")
        print("    -> Start web dashboard on localhost:8080")
        print("\nFor detailed options: python run.py --help")
        print("=" * 60 + "\n")
        return

    if args.build:
        # Full build: train tokenizer + model from scratch
        print("\n" + "=" * 60)
        print("AI TESTER - FULL BUILD")
        print("=" * 60)
        
        from ai_tester.core.training import train_model
        from ai_tester.config import CONFIG
        
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
        from ai_tester.core.training import train_model
        from ai_tester.config import CONFIG
        
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
        from ai_tester.comms.api_server import create_app
        app = create_app()
        print("\nStarting API server at http://127.0.0.1:5000")
        print("Press Ctrl+C to stop\n")
        app.run(host="127.0.0.1", port=5000, debug=True)

    if args.run:
        from ai_tester.core.inference import AITesterEngine
        print("\n" + "=" * 50)
        print("AI Tester CLI Chat")
        print("=" * 50)
        print("Type your message and press Enter.")
        print("Type 'quit' or 'exit' to stop.\n")

        try:
            engine = AITesterEngine()
        except FileNotFoundError as e:
            print(f"\n[ERROR] Model not found")
            print(f"   {e}")
            print("\nTo fix this:")
            print("   1. Train a model first:")
            print("      python run.py --train")
            print("   2. Or use the GUI to train:")
            print("      python run.py --gui")
            return
        except ImportError as e:
            print(f"\n[ERROR] Missing dependency")
            print(f"   {e}")
            print("\nTo fix this:")
            print("   Install required packages:")
            print("      pip install -r requirements.txt")
            return
        except Exception as e:
            print(f"\n[ERROR] Error loading model: {e}")
            print("\nTroubleshooting:")
            print("   - Check if the model file exists in the models/ directory")
            print("   - Try retraining: python run.py --train --force")
            print("   - Check logs for more details")
            return

        print("[OK] Model loaded successfully!\n")

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
                try:
                    for token in engine.stream_generate(prompt, max_gen=200):
                        print(token, end="", flush=True)
                except Exception as e:
                    print(f"\n\n[WARNING] Generation error: {e}")
                    print("Try a different prompt or check the model.")
                print("\n")

            except KeyboardInterrupt:
                print("\n\n[SYSTEM] Goodbye!")
                break

    if args.gui:
        _print_startup_banner()
        try:
            from ai_tester.gui.enhanced_window import run_app
        except ImportError as e:
            print(f"\n[ERROR] GUI requires PyQt5")
            print(f"   Error: {e}")
            print("\nTo fix this:")
            print("   Install PyQt5:")
            print("      pip install PyQt5")
            print("\n   On Raspberry Pi, use the system package:")
            print("      sudo apt install python3-pyqt5")
            sys.exit(1)
        run_app()
    
    if args.background:
        try:
            from ai_tester.background import main as run_background
        except ImportError as e:
            print(f"\n[ERROR] Background mode requires PyQt5")
            print(f"   Error: {e}")
            print("\nTo fix this:")
            print("   Install PyQt5:")
            print("      pip install PyQt5")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("AI TESTER - BACKGROUND MODE")
        print("=" * 60)
        print("\nRunning in system tray...")
        print("Click tray icon or press Ctrl+Space for quick commands")
        print("Press Ctrl+C to exit\n")
        run_background()
    
    if args.web:
        try:
            from ai_tester.web.app import run_web
        except ImportError as e:
            print(f"\n[ERROR] Web dashboard requires flask-socketio")
            print(f"   Error: {e}")
            print("\nTo fix this:")
            print("   Install required packages:")
            print("      pip install flask-socketio")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("AI TESTER - WEB DASHBOARD")
        print("=" * 60)
        print(f"\nStarting web server...")
        
        # Setup instance manager if needed
        if args.instance or args.new_instance:
            from ai_tester.core.instance_manager import InstanceManager
            instance_manager = InstanceManager(instance_id=args.instance)
            print(f"Instance ID: {instance_manager.instance_id}")
        
        run_web(host='0.0.0.0', port=8080)


if __name__ == "__main__":
    main()
