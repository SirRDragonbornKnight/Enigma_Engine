#!/usr/bin/env python3
"""
================================================================================
🚀 ForgeAI - MAIN ENTRY POINT
================================================================================

This is where EVERYTHING begins! Like the front door to a castle.
Your journey through ForgeAI starts here.

📍 FILE: run.py
🏷️ TYPE: Application Entry Point

┌─────────────────────────────────────────────────────────────────────────────┐
│  COMMAND OPTIONS:                                                           │
│                                                                             │
│    python run.py --gui     → Opens graphical interface (RECOMMENDED!)      │
│    python run.py --train   → Train your AI model                           │
│    python run.py --run     → Chat in terminal (CLI)                        │
│    python run.py --serve   → Start REST API server                         │
│    python run.py --tunnel  → Expose server to internet (public access)     │
│    python run.py --build   → Build new model from scratch                  │
└─────────────────────────────────────────────────────────────────────────────┘

🔗 CONNECTED FILES:
    --train  → forge_ai/core/training.py      (Trainer, train_model)
    --run    → forge_ai/core/inference.py     (ForgeEngine)
    --gui    → forge_ai/gui/enhanced_window.py (EnhancedMainWindow)
    --serve  → forge_ai/comms/api_server.py   (Flask REST API)

📖 SEE ALSO:
    • CODE_ADVENTURE_TOUR.txt  - Full guided tour of the codebase
    • QUICK_FILE_LOCATOR.txt   - Fast file finder
    • docs/CODE_TOUR.md        - Detailed documentation

For first-time users, start with: python run.py --gui
"""

# === EARLY ENVIRONMENT SETUP ===
# These must be set BEFORE any imports that might load GTK/Qt
import os
os.environ["NO_AT_BRIDGE"] = "1"
os.environ["GTK_A11Y"] = "none"
os.environ["GTK_MODULES"] = ""  # Disable gail and atk-bridge modules
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import argparse
import sys
from pathlib import Path


def _suppress_noise():
    """Suppress noisy warnings from Qt, pygame, ALSA, and other libs.
    
    MUST be called BEFORE any other imports that might load audio libs.
    """
    import os
    import warnings
    import logging
    import ctypes
    
    # ===== ALSA ERROR SUPPRESSION =====
    # Suppress ALSA error messages at the C level
    # This MUST happen before any audio library is loaded
    try:
        # Try to load libasound and redirect errors to null
        ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                               ctypes.c_char_p, ctypes.c_int,
                                               ctypes.c_char_p)
        def py_error_handler(filename, line, function, err, fmt):
            pass  # Swallow all ALSA errors
        
        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        
        try:
            asound = ctypes.cdll.LoadLibrary('libasound.so.2')
            asound.snd_lib_error_set_handler(c_error_handler)
        except OSError:
            pass  # libasound not available, that's fine
    except Exception:
        pass  # If this fails, continue anyway
    
    # ===== QT NOISE SUPPRESSION =====
    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
    
    # ===== GTK/ATK ACCESSIBILITY SUPPRESSION =====
    # Suppress GTK gail module and ATK bridge warnings
    os.environ["NO_AT_BRIDGE"] = "1"
    os.environ["GTK_A11Y"] = "none"
    
    # ===== AUDIO DRIVER SETTINGS =====
    # Use dummy audio driver if no real audio needed
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    
    # Suppress JACK server warnings
    os.environ["JACK_NO_START_SERVER"] = "1"
    
    # ===== PYTHON LOGGING =====
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("alsa").setLevel(logging.CRITICAL)
    logging.getLogger("jack").setLevel(logging.CRITICAL)
    
    # ===== PYTHON WARNINGS =====
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*ALSA.*")
    warnings.filterwarnings("ignore", message=".*audio.*")
    warnings.filterwarnings("ignore", message=".*jack.*")
    warnings.filterwarnings("ignore", message=".*torch_dtype.*")
    warnings.filterwarnings("ignore", message=".*deprecated.*")


# MUST suppress noise BEFORE any other imports
_suppress_noise()


def _print_startup_banner():
    """Print a clean startup message."""
    print("=" * 50)
    print("  ForgeAI - Starting...")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="ForgeAI - Build and run your own AI",
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
  python run.py --tunnel                 Expose server to internet via ngrok
  python run.py --tunnel --tunnel-provider localtunnel  Use localtunnel instead
  python run.py --tunnel --tunnel-token YOUR_TOKEN      Use ngrok with auth token
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
    parser.add_argument("--tunnel", action="store_true", help="Start tunnel for public access")
    
    # API server options
    parser.add_argument("--api-type", type=str, default="openai",
                        choices=["openai", "simple"],
                        help="API type: openai (compatible) or simple (default: openai)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port for API server (default: 8000 for openai, 5000 for simple)")
    
    # Multi-instance options
    parser.add_argument("--instance", type=str, default=None, help="Instance ID (for multi-instance)")
    parser.add_argument("--new-instance", action="store_true", help="Force new instance")
    
    # Tunnel options
    parser.add_argument("--tunnel-provider", type=str, default="ngrok",
                        choices=["ngrok", "localtunnel", "bore"],
                        help="Tunnel provider (default: ngrok)")
    parser.add_argument("--tunnel-port", type=int, default=5000,
                        help="Port to tunnel (default: 5000)")
    parser.add_argument("--tunnel-token", type=str, default=None,
                        help="Tunnel auth token (required for ngrok)")
    parser.add_argument("--tunnel-region", type=str, default=None,
                        help="Tunnel region (ngrok only): us, eu, ap, au, sa, jp, in")
    parser.add_argument("--tunnel-subdomain", type=str, default=None,
                        help="Custom subdomain (requires paid plan)")
    
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
    if not any([args.train, args.build, args.serve, args.run, args.gui, args.web, args.background, args.tunnel]):
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
        
        from forge_ai.core.training import train_model
        from forge_ai.config import CONFIG
        
        data_path = args.data or Path(CONFIG["data_dir"]) / "data.txt"
        output_path = args.output or Path(CONFIG["models_dir"]) / f"{args.model}_forge.pth"
        
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
        from forge_ai.core.training import train_model
        from forge_ai.config import CONFIG
        
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

    if args.tunnel:
        from forge_ai.comms.tunnel_manager import TunnelManager
        
        print("\n" + "=" * 60)
        print("  ForgeAI Tunnel Manager")
        print("=" * 60)
        
        # Create tunnel manager
        manager = TunnelManager(
            provider=args.tunnel_provider,
            auth_token=args.tunnel_token,
            region=args.tunnel_region,
            subdomain=args.tunnel_subdomain
        )
        
        print(f"\nStarting {args.tunnel_provider} tunnel on port {args.tunnel_port}...")
        print("This will expose your local server to the internet.\n")
        
        if args.tunnel_provider == "ngrok" and not args.tunnel_token:
            print("⚠️  Warning: ngrok requires an auth token for best results.")
            print("   Sign up at https://ngrok.com and get your token.")
            print("   Then use: --tunnel-token YOUR_TOKEN\n")
        
        # Start tunnel
        tunnel_url = manager.start_tunnel(args.tunnel_port)
        
        if tunnel_url:
            print("\n✓ Tunnel started successfully!")
            print(f"\n  Public URL: {tunnel_url}")
            print(f"  Local Port: {args.tunnel_port}")
            print(f"\n  Share this URL to give others access to your ForgeAI server.")
            print(f"  Press Ctrl+C to stop the tunnel.\n")
            
            # Keep running
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\nStopping tunnel...")
                manager.stop_tunnel()
                print("✓ Tunnel stopped.\n")
        else:
            print("\n✗ Failed to start tunnel.")
            print(f"\nTroubleshooting:")
            print(f"  1. Make sure {args.tunnel_provider} is installed")
            if args.tunnel_provider == "ngrok":
                print(f"     Download from: https://ngrok.com/download")
                print(f"     Or install with: snap install ngrok")
            elif args.tunnel_provider == "localtunnel":
                print(f"     Install with: npm install -g localtunnel")
            elif args.tunnel_provider == "bore":
                print(f"     Install from: https://github.com/ekzhang/bore")
            print(f"  2. Check that port {args.tunnel_port} is available")
            print(f"  3. Verify your internet connection\n")
            return

    if args.serve:
        api_type = getattr(args, 'api_type', 'openai')
        port = getattr(args, 'port', None)
        
        if api_type == "openai":
            # OpenAI-compatible API (recommended)
            from forge_ai.comms.openai_api import create_openai_server
            port = port or 8000
            print("\n" + "=" * 60)
            print("  ForgeAI OpenAI-Compatible API Server")
            print("=" * 60)
            print("\nThis server is compatible with:")
            print("  - OpenAI Python SDK")
            print("  - LangChain")
            print("  - LlamaIndex")
            print("  - Any OpenAI-compatible tool")
            print(f"\nBase URL: http://localhost:{port}/v1")
            print("\nPress Ctrl+C to stop\n")
            create_openai_server(host="0.0.0.0", port=port)
        else:
            # Simple API (original)
            from forge_ai.comms.api_server import create_app
            port = port or 5000
            app = create_app()
            print(f"\nStarting API server at http://127.0.0.1:{port}")
            print("Press Ctrl+C to stop\n")
            app.run(host="127.0.0.1", port=port, debug=True)

    if args.run:
        from forge_ai.core.inference import ForgeEngine
        print("\n" + "=" * 50)
        print("ForgeAI CLI Chat")
        print("=" * 50)
        print("Type your message and press Enter.")
        print("Type 'quit' or 'exit' to stop.\n")

        try:
            engine = ForgeEngine()
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
        # Suppress GTK/ATK stderr noise during Qt initialization
        # Redirect stderr at the file descriptor level to catch C-level output
        import sys
        stderr_fd = sys.stderr.fileno()
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            old_stderr_fd = os.dup(stderr_fd)
            os.dup2(devnull_fd, stderr_fd)
            os.close(devnull_fd)
        except (OSError, AttributeError):
            old_stderr_fd = None
        
        _print_startup_banner()
        
        try:
            from forge_ai.gui.enhanced_window import run_app
        except ImportError as e:
            # Restore stderr for error messages
            if old_stderr_fd is not None:
                os.dup2(old_stderr_fd, stderr_fd)
                os.close(old_stderr_fd)
            print(f"\n[ERROR] GUI requires PyQt5")
            print(f"   Error: {e}")
            print("\nTo fix this:")
            print("   Install PyQt5:")
            print("      pip install PyQt5")
            print("\n   On Raspberry Pi, use the system package:")
            print("      sudo apt install python3-pyqt5")
            sys.exit(1)
        
        # Restore stderr before running app (so we can see real errors)
        if old_stderr_fd is not None:
            os.dup2(old_stderr_fd, stderr_fd)
            os.close(old_stderr_fd)
        
        run_app()
    
    if args.background:
        try:
            from forge_ai.background import main as run_background
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
            from forge_ai.web.app import run_web
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
            from forge_ai.core.instance_manager import InstanceManager
            instance_manager = InstanceManager(instance_id=args.instance)
            print(f"Instance ID: {instance_manager.instance_id}")
        
        run_web(host='0.0.0.0', port=8080)


if __name__ == "__main__":
    main()
