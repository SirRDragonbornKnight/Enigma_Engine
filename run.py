"""
Enigma Engine - Main Entry Point

Run your AI with one of these commands:
    python run.py --gui     # GUI interface (recommended)
    python run.py --train   # Train from command line
    python run.py --run     # CLI chat interface
    python run.py --serve   # Start API server

For first-time users, start with --gui
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Enigma Engine - Build and run your own AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --gui     Launch the GUI (recommended for beginners)
  python run.py --train   Train from command line
  python run.py --run     Simple CLI chat
  python run.py --serve   Start API server on localhost:5000
        """
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--run", action="store_true", help="Run CLI chat interface")
    parser.add_argument("--gui", action="store_true", help="Start the GUI (recommended)")
    args = parser.parse_args()

    # If no arguments, show help and suggest GUI
    if not any([args.train, args.serve, args.run, args.gui]):
        print("\nWelcome to Enigma Engine!")
        print("=" * 40)
        print("\nTo get started, run one of:")
        print("  python run.py --gui     # GUI (recommended)")
        print("  python run.py --train   # Train model")
        print("  python run.py --run     # CLI chat")
        print("  python run.py --serve   # API server")
        print("\nFor more options: python run.py --help")
        print()
        return

    if args.train:
        from enigma.core.training import train_model
        train_model(force=False)

    if args.serve:
        from enigma.comms.api_server import create_app
        app = create_app()
        print("\nStarting API server at http://127.0.0.1:5000")
        print("Press Ctrl+C to stop\n")
        app.run(host="127.0.0.1", port=5000, debug=True)

    if args.run:
        from enigma.core.inference import EnigmaEngine
        print("\n" + "=" * 40)
        print("Enigma CLI Chat")
        print("=" * 40)
        print("Type your message and press Enter.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        try:
            engine = EnigmaEngine()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Have you trained the model yet? Try: python run.py --gui")
            return
            
        while True:
            try:
                prompt = input("You: ")
                if prompt.strip().lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                if not prompt.strip():
                    continue
                resp = engine.generate(prompt)
                print(f"AI: {resp}\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
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
