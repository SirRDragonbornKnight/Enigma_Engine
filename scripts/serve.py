#!/usr/bin/env python3
"""
Start the Enigma API server.

Usage:
    python -m scripts.serve
    python -m scripts.serve --port 8080 --host 0.0.0.0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Start the Enigma API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port to listen on"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Model to serve"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--cors", action="store_true",
        help="Enable CORS"
    )
    
    args = parser.parse_args()
    
    from enigma.comms.api_server import create_app
    
    app = create_app()
    
    print(f"Starting Enigma API server on {args.host}:{args.port}")
    print("Endpoints:")
    print("  POST /generate - Generate text")
    print("  GET  /health   - Health check")
    print()
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
