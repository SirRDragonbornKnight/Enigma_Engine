#!/usr/bin/env python3
"""
Example: Expose Enigma AI Engine API server to the internet using tunnel.

This example shows how to:
1. Start the Enigma AI Engine API server
2. Create a tunnel to expose it publicly
3. Share the public URL

Usage:
    python examples/tunnel_example.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enigma_engine.comms.tunnel_manager import TunnelManager, get_tunnel_manager
import time


def example_basic_tunnel():
    """Basic tunnel example."""
    print("=" * 60)
    print("Example 1: Basic Tunnel")
    print("=" * 60)
    
    # Create tunnel manager
    manager = TunnelManager(provider="ngrok")
    
    # Start tunnel on port 5000
    print("\nStarting tunnel on port 5000...")
    url = manager.start_tunnel(5000)
    
    if url:
        print(f"\n✓ Tunnel started!")
        print(f"  Public URL: {url}")
        print(f"\n  Try accessing: {url}/health")
        print("\n  Press Ctrl+C to stop...\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping tunnel...")
            manager.stop_tunnel()
            print("✓ Done!\n")
    else:
        print("\n✗ Failed to start tunnel")
        print("\nMake sure ngrok is installed:")
        print("  Download from: https://ngrok.com/download")
        print("  Or install with: snap install ngrok")


def example_with_auth_token():
    """Example with ngrok auth token."""
    print("=" * 60)
    print("Example 2: Tunnel with Auth Token")
    print("=" * 60)
    
    # Get auth token from environment or prompt
    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    
    if not auth_token:
        print("\n⚠️  No auth token found.")
        print("   Set NGROK_AUTH_TOKEN environment variable")
        print("   Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken")
        return
    
    # Create tunnel with auth
    manager = TunnelManager(
        provider="ngrok",
        auth_token=auth_token,
        region="us"
    )
    
    print("\nStarting authenticated tunnel...")
    url = manager.start_tunnel(5000)
    
    if url:
        print(f"\n✓ Tunnel started!")
        print(f"  Public URL: {url}")
        print("\n  Press Ctrl+C to stop...\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop_tunnel()
            print("\n✓ Done!\n")


def example_localtunnel():
    """Example using localtunnel (no auth required)."""
    print("=" * 60)
    print("Example 3: LocalTunnel (No Auth Required)")
    print("=" * 60)
    
    manager = TunnelManager(provider="localtunnel")
    
    print("\nStarting localtunnel...")
    print("Note: No account needed, but less stable than ngrok")
    
    url = manager.start_tunnel(5000)
    
    if url:
        print(f"\n✓ Tunnel started!")
        print(f"  Public URL: {url}")
        print("\n  Press Ctrl+C to stop...\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop_tunnel()
            print("\n✓ Done!\n")
    else:
        print("\n✗ Failed to start localtunnel")
        print("\nMake sure localtunnel is installed:")
        print("  npm install -g localtunnel")


def example_module_system():
    """Example using Enigma AI Engine module system."""
    print("=" * 60)
    print("Example 4: Using Module System")
    print("=" * 60)
    
    try:
        from enigma_engine.modules import ModuleManager
        
        # Create module manager
        manager = ModuleManager()
        
        # Load tunnel module
        print("\nLoading tunnel module...")
        success = manager.load('tunnel', config={
            'provider': 'ngrok',
            'port': 5000,
            'auto_start': False
        })
        
        if success:
            print("✓ Tunnel module loaded")
            
            # Get module instance via proper interface
            tunnel_mod = manager.get_module('tunnel')
            if tunnel_mod:
                # Get the actual tunnel manager instance
                tunnel_instance = tunnel_mod.get_interface() if hasattr(tunnel_mod, 'get_interface') else tunnel_mod._instance
                if tunnel_instance:
                    # Start tunnel
                    print("\nStarting tunnel...")
                    url = tunnel_instance.start_tunnel(5000)
                    
                    if url:
                        print(f"\n✓ Tunnel started!")
                        print(f"  Public URL: {url}")
                        print("\n  Press Ctrl+C to stop...\n")
                        
                        try:
                            while True:
                                time.sleep(1)
                        except KeyboardInterrupt:
                            print("\n\nUnloading module...")
                        manager.unload('tunnel')
                        print("✓ Done!\n")
            else:
                print("✗ Failed to get tunnel instance")
        else:
            print("✗ Failed to load tunnel module")
            
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Make sure Enigma AI Engine is properly installed")


def main():
    """Run examples."""
    print("\n" + "=" * 60)
    print("Enigma AI Engine Tunnel Examples")
    print("=" * 60)
    print("\nChoose an example:")
    print("  1. Basic tunnel (ngrok, no auth)")
    print("  2. Tunnel with auth token (ngrok)")
    print("  3. LocalTunnel (no account needed)")
    print("  4. Module system integration")
    print("  q. Quit")
    
    choice = input("\nChoice: ").strip()
    
    if choice == "1":
        example_basic_tunnel()
    elif choice == "2":
        example_with_auth_token()
    elif choice == "3":
        example_localtunnel()
    elif choice == "4":
        example_module_system()
    elif choice.lower() == "q":
        print("Goodbye!")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
