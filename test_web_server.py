#!/usr/bin/env python3
"""
Test script for Enigma AI Engine Web Server

Run this to test the web interface without the full GUI.
"""

import sys
from pathlib import Path

# Add enigma_engine to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("="*60)
    print("Enigma AI Engine Web Server Test")
    print("="*60)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    try:
        import fastapi
        print("   ✓ FastAPI installed")
    except ImportError:
        print("   ✗ FastAPI not installed - run: pip install fastapi")
        return
    
    try:
        import uvicorn
        print("   ✓ Uvicorn installed")
    except ImportError:
        print("   ✗ Uvicorn not installed - run: pip install uvicorn")
        return
    
    try:
        import websockets
        print("   ✓ WebSockets installed")
    except ImportError:
        print("   ✗ WebSockets not installed - run: pip install websockets")
        return
    
    # Import web server
    print("\n2. Importing web server...")
    try:
        from enigma_engine.web.server import create_web_server
        from enigma_engine.web.discovery import get_local_ip
        print("   ✓ Web server imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import: {e}")
        return
    
    # Get local IP
    print("\n3. Getting local IP...")
    try:
        local_ip = get_local_ip()
        print(f"   ✓ Local IP: {local_ip}")
    except Exception as e:
        print(f"   ✗ Failed to get IP: {e}")
        local_ip = "localhost"
    
    # Create server
    print("\n4. Creating web server...")
    try:
        server = create_web_server(host="0.0.0.0", port=8080, require_auth=True)
        print("   ✓ Server created")
    except Exception as e:
        print(f"   ✗ Failed to create server: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print connection info
    print("\n" + "="*60)
    print("SUCCESS! Web server is ready to start.")
    print("="*60)
    print(f"\nTo access from this computer:")
    print(f"   http://localhost:8080")
    print(f"\nTo access from other devices:")
    print(f"   http://{local_ip}:8080")
    print(f"\nQR code page:")
    print(f"   http://{local_ip}:8080/qr")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Start server
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\nError running server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
