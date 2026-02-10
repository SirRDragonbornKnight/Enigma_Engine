#!/usr/bin/env python3
"""
Enigma AI Engine - Main Entry Point

Commands:
    python run.py --gui     Launch GUI (recommended)
    python run.py --train   Train model
    python run.py --run     CLI chat
    python run.py --serve   Start API server
    python run.py --tunnel  Expose server to internet
    python run.py --build   Build new model from scratch

See --help for all options.
"""

# === EARLY ENVIRONMENT SETUP ===
# These must be set BEFORE any imports that might load GTK/Qt
import os
os.environ["NO_AT_BRIDGE"] = "1"
os.environ["GTK_A11Y"] = "none"
os.environ["GTK_MODULES"] = ""  # Disable gail and atk-bridge modules
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import argparse
import logging
import sys
from pathlib import Path

# Module logger
logger = logging.getLogger(__name__)


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
        except OSError as e:
            logger.debug(f"libasound not available: {e}")
    except Exception as e:
        logger.debug(f"ALSA error suppression setup failed: {e}")
    
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
    print("  Enigma AI Engine - Starting...")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Enigma AI Engine - Build and run your own AI",
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

    # Import command handlers (lazy import to avoid circular deps)
    from enigma_engine.cli.commands import (
        cmd_build, cmd_train, cmd_tunnel, cmd_serve, 
        cmd_run_cli, cmd_gui, cmd_background, cmd_web
    )
    
    if args.build:
        cmd_build(
            model_size=args.model,
            epochs=args.epochs,
            data_path=args.data,
            output_path=args.output,
            force=args.force
        )

    if args.train:
        cmd_train(
            model_size=args.model,
            epochs=args.epochs,
            data_path=args.data,
            output_path=args.output,
            force=args.force
        )

    if args.tunnel:
        cmd_tunnel(
            provider=args.tunnel_provider,
            port=args.tunnel_port,
            auth_token=args.tunnel_token,
            region=args.tunnel_region,
            subdomain=args.tunnel_subdomain
        )

    if args.serve:
        cmd_serve(
            api_type=getattr(args, 'api_type', 'openai'),
            port=getattr(args, 'port', None)
        )

    if args.run:
        cmd_run_cli()

    if args.gui:
        cmd_gui()
    
    if args.background:
        cmd_background()
    
    if args.web:
        cmd_web(
            instance_id=args.instance,
            new_instance=args.new_instance
        )


if __name__ == "__main__":
    main()
