"""
CLI Command Handlers

Extracted command handlers for run.py and CLI interfaces.
These can be called programmatically or from command line.

FILE: enigma_engine/cli/commands.py
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def cmd_train(
    model_size: str = "small",
    epochs: int = 30,
    data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    force: bool = False,
    train_tokenizer: bool = True
) -> Dict[str, Any]:
    """
    Train a model.
    
    Args:
        model_size: Model size (tiny, small, medium, large, xl, xxl)
        epochs: Number of training epochs
        data_path: Path to training data (default: data/data.txt)
        output_path: Output model path (default: models/{size}_forge.pth)
        force: Force retrain even if model exists
        train_tokenizer: Train tokenizer first
        
    Returns:
        Dict with training results (status, final_loss, model_path)
    """
    from ..core.training import train_model
    from ..config import CONFIG
    
    data_path = data_path or Path(CONFIG["data_dir"]) / "training.txt"
    
    logger.info(f"Training {model_size} model, data={data_path}, epochs={epochs}")
    print(f"\nTraining {model_size} model...")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}")
    print()
    
    results = train_model(
        data_path=data_path,
        epochs=epochs,
        model_size=model_size,
        output_path=output_path,
        train_tokenizer_first=train_tokenizer,
        force=force
    )
    
    if results.get('status') != 'skipped':
        print(f"\nTraining complete!")
        print(f"  Final loss: {results.get('final_loss', 'N/A')}")
    
    return results


def cmd_build(
    model_size: str = "small",
    epochs: int = 30,
    data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Build a new model from scratch (tokenizer + model).
    
    Args:
        model_size: Model size
        epochs: Training epochs
        data_path: Training data path
        output_path: Output path
        force: Force rebuild
        
    Returns:
        Dict with build results
    """
    from ..core.training import train_model
    from ..config import CONFIG
    
    data_path = data_path or Path(CONFIG["data_dir"]) / "training.txt"
    output_path = output_path or Path(CONFIG["models_dir"]) / f"{model_size}_forge.pth"
    
    print("\n" + "=" * 60)
    print("ENIGMA AI ENGINE - FULL BUILD")
    print("=" * 60)
    
    logger.info(f"Building {model_size} model from {data_path}")
    logger.info(f"Output: {output_path}")
    print(f"\nBuilding {model_size} model from {data_path}")
    print(f"Output: {output_path}")
    print()
    
    results = train_model(
        data_path=data_path,
        epochs=epochs,
        model_size=model_size,
        output_path=output_path,
        train_tokenizer_first=True,
        force=force
    )
    
    print(f"\nBuild complete!")
    print(f"  Model saved to: {results.get('model_path', output_path)}")
    print(f"  Final loss: {results.get('final_loss', 'N/A')}")
    
    return results


def cmd_serve(
    api_type: str = "openai",
    port: Optional[int] = None,
    host: str = "127.0.0.1"
) -> None:
    """
    Start the API server.
    
    Args:
        api_type: API type ("openai" or "simple")
        port: Port number (default: 8000 for openai, 5000 for simple)
        host: Host to bind to
    """
    if api_type == "openai":
        from ..comms.openai_api import create_openai_server
        port = port or 8000
        logger.info(f"Starting OpenAI-compatible API server on port {port}")
        print("\n" + "=" * 60)
        print("  Enigma AI Engine OpenAI-Compatible API Server")
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
        from ..comms.api_server import create_app
        port = port or 5000
        app = create_app()
        logger.info(f"Starting simple API server on port {port}")
        print(f"\nStarting API server at http://{host}:{port}")
        print("Press Ctrl+C to stop\n")
        app.run(host=host, port=port, debug=True)


def cmd_tunnel(
    provider: str = "ngrok",
    port: int = 5000,
    auth_token: Optional[str] = None,
    region: Optional[str] = None,
    subdomain: Optional[str] = None
) -> Optional[str]:
    """
    Start a tunnel to expose local server.
    
    Args:
        provider: Tunnel provider (ngrok, localtunnel, bore)
        port: Local port to tunnel
        auth_token: Authentication token (required for ngrok)
        region: Region (ngrok: us, eu, ap, au, sa, jp, in)
        subdomain: Custom subdomain (paid plans)
        
    Returns:
        Tunnel URL if successful, None otherwise
    """
    from ..comms.tunnel_manager import TunnelManager
    
    print("\n" + "=" * 60)
    print("  Enigma AI Engine Tunnel Manager")
    print("=" * 60)
    
    manager = TunnelManager(
        provider=provider,
        auth_token=auth_token,
        region=region,
        subdomain=subdomain
    )
    
    logger.info(f"Starting {provider} tunnel on port {port}")
    print(f"\nStarting {provider} tunnel on port {port}...")
    print("This will expose your local server to the internet.\n")
    
    if provider == "ngrok" and not auth_token:
        print("Warning: ngrok requires an auth token for best results.")
        print("   Sign up at https://ngrok.com and get your token.")
        print("   Then use: --tunnel-token YOUR_TOKEN\n")
    
    tunnel_url = manager.start_tunnel(port)
    
    if tunnel_url:
        logger.info(f"Tunnel started successfully: {tunnel_url}")
        print("\nTunnel started successfully!")
        print(f"\n  Public URL: {tunnel_url}")
        print(f"  Local Port: {port}")
        print(f"\n  Share this URL to give others access to your server.")
        print(f"  Press Ctrl+C to stop the tunnel.\n")
        
        # Keep running until interrupted
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping tunnel...")
            manager.stop_tunnel()
            print("Tunnel stopped.\n")
        
        return tunnel_url
    else:
        logger.error(f"Failed to start {provider} tunnel on port {port}")
        print("\nFailed to start tunnel.")
        print(f"\nTroubleshooting:")
        print(f"  1. Make sure {provider} is installed")
        if provider == "ngrok":
            print(f"     Download from: https://ngrok.com/download")
        elif provider == "localtunnel":
            print(f"     Install with: npm install -g localtunnel")
        elif provider == "bore":
            print(f"     Install from: https://github.com/ekzhang/bore")
        print(f"  2. Check that port {port} is available")
        print(f"  3. Verify your internet connection\n")
        return None


def cmd_run_cli() -> None:
    """Run CLI chat interface."""
    from ..core.inference import EnigmaEngine
    
    print("\n" + "=" * 50)
    print("Enigma AI Engine CLI Chat")
    print("=" * 50)
    print("Type your message and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    try:
        engine = EnigmaEngine()
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        print(f"\n[ERROR] Model not found")
        print(f"   {e}")
        print("\nTo fix this:")
        print("   1. Train a model first: python run.py --train")
        print("   2. Or use the GUI: python run.py --gui")
        return
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print(f"\n[ERROR] Missing dependency: {e}")
        print("\nInstall required packages: pip install -r requirements.txt")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        print(f"\n[ERROR] Error loading model: {e}")
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

            print("AI: ", end="", flush=True)
            try:
                for token in engine.stream_generate(prompt, max_gen=200):
                    print(token, end="", flush=True)
            except Exception as e:
                logger.warning(f"Generation error: {e}", exc_info=True)
                print(f"\n\n[WARNING] Generation error: {e}")
            print("\n")

        except KeyboardInterrupt:
            print("\n\n[SYSTEM] Goodbye!")
            break


def cmd_gui() -> None:
    """Launch the GUI application."""
    import os
    
    # Suppress GTK/ATK stderr noise during Qt initialization
    stderr_fd = sys.stderr.fileno()
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr_fd = os.dup(stderr_fd)
        os.dup2(devnull_fd, stderr_fd)
        os.close(devnull_fd)
    except (OSError, AttributeError):
        old_stderr_fd = None
    
    print("=" * 50)
    print("  Enigma AI Engine - Starting...")
    print("=" * 50)
    
    # Import torch BEFORE PyQt5 to prevent DLL conflict on Windows
    # (PyQt5 pollutes the DLL search path, breaking torch's c10.dll loading)
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    
    try:
        from ..gui.enhanced_window import run_app
    except (ImportError, OSError) as e:
        if old_stderr_fd is not None:
            os.dup2(old_stderr_fd, stderr_fd)
            os.close(old_stderr_fd)
        logger.error(f"GUI failed to load: {e}")
        print(f"\n[ERROR] GUI failed to load: {e}")
        if isinstance(e, ImportError):
            print("\nInstall PyQt5: pip install PyQt5")
        sys.exit(1)
    
    # Restore stderr before running app
    if old_stderr_fd is not None:
        os.dup2(old_stderr_fd, stderr_fd)
        os.close(old_stderr_fd)
    
    run_app()


def cmd_background() -> None:
    """Run in background/system tray mode."""
    try:
        from ..background import main as run_background
    except ImportError as e:
        logger.error(f"Background mode requires PyQt5: {e}")
        print(f"\n[ERROR] Background mode requires PyQt5: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("ENIGMA AI ENGINE - BACKGROUND MODE")
    print("=" * 60)
    print("\nRunning in system tray...")
    print("Click tray icon or press Ctrl+Space for quick commands")
    print("Press Ctrl+C to exit\n")
    run_background()


def cmd_web(
    host: str = "0.0.0.0",
    port: int = 8080,
    instance_id: Optional[str] = None,
    new_instance: bool = False
) -> None:
    """
    Start web dashboard.
    
    Args:
        host: Host to bind to
        port: Port number
        instance_id: Instance ID for multi-instance
        new_instance: Force new instance
    """
    try:
        from ..web.app import run_web
    except ImportError as e:
        logger.error(f"Web dashboard requires flask-socketio: {e}")
        print(f"\n[ERROR] Web dashboard requires flask-socketio: {e}")
        print("\nInstall: pip install flask-socketio")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("ENIGMA AI ENGINE - WEB DASHBOARD")
    print("=" * 60)
    logger.info(f"Starting web dashboard on port {port}")
    print(f"\nStarting web server...")
    
    if instance_id or new_instance:
        from ..core.instance_manager import InstanceManager
        instance_manager = InstanceManager(instance_id=instance_id)
        print(f"Instance ID: {instance_manager.instance_id}")
    
    run_web(host=host, port=port)


def cmd_mobile_api(host: str = "0.0.0.0", port: int = 5001) -> None:
    """
    Start mobile API server.
    
    Args:
        host: Host to bind to
        port: Port number
    """
    from ..mobile.api import run_mobile_api
    run_mobile_api(host=host, port=port)


__all__ = [
    'cmd_train',
    'cmd_build',
    'cmd_serve',
    'cmd_tunnel',
    'cmd_run_cli',
    'cmd_gui',
    'cmd_background',
    'cmd_web',
    'cmd_mobile_api',
]
