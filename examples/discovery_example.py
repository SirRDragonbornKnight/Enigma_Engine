#!/usr/bin/env python3
"""
Example: Device Discovery

This script demonstrates how to use the DeviceDiscovery system
to find other ForgeAI instances on your network.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_broadcast_discovery():
    """Example 1: Quick UDP broadcast discovery."""
    from forge_ai.comms.discovery import DeviceDiscovery
    
    print("\n" + "="*60)
    print("Example 1: Broadcast Discovery (Fast)")
    print("="*60 + "\n")
    
    # Create discovery instance
    discovery = DeviceDiscovery(node_name="example_client", node_port=8765)
    
    print("Sending broadcast discovery message...")
    print("Waiting 3 seconds for responses...\n")
    
    # Discover devices via UDP broadcast
    devices = discovery.broadcast_discover(timeout=3.0)
    
    if devices:
        print(f"Found {len(devices)} device(s):\n")
        for name, info in devices.items():
            print(f"  Device: {name}")
            print(f"    URL: http://{info['ip']}:{info['port']}")
            if 'model' in info:
                print(f"    Model: {info['model']}")
            print()
    else:
        print("No devices found.")
        print("Make sure other ForgeAI instances are running with:")
        print("  - Server started")
        print("  - Discovery listener enabled")
        print("  - Same network/subnet")


def example_full_scan():
    """Example 2: Full network scan."""
    from forge_ai.comms.discovery import DeviceDiscovery
    
    print("\n" + "="*60)
    print("Example 2: Full Network Scan (Thorough)")
    print("="*60 + "\n")
    
    # Create discovery instance
    discovery = DeviceDiscovery(node_name="example_scanner", node_port=8765)
    
    print("Scanning network...")
    print("This may take 30-60 seconds...\n")
    
    # Scan entire subnet
    devices = discovery.scan_network(port=8765, timeout=0.3)
    
    if devices:
        print(f"Found {len(devices)} device(s):\n")
        for name, info in devices.items():
            print(f"  Device: {name}")
            print(f"    URL: http://{info['ip']}:{info['port']}")
            if 'model' in info:
                print(f"    Model: {info['model']}")
            print()
    else:
        print("No devices found.")


def example_with_callback():
    """Example 3: Discovery with callback."""
    from forge_ai.comms.discovery import DeviceDiscovery
    import time
    
    print("\n" + "="*60)
    print("Example 3: Discovery with Callback")
    print("="*60 + "\n")
    
    # Create discovery instance
    discovery = DeviceDiscovery(node_name="callback_example", node_port=8765)
    
    # Define callback
    def on_device_discovered(name, info):
        print(f"[CALLBACK] Device discovered: {name}")
        print(f"           IP: {info['ip']}, Port: {info['port']}")
    
    # Register callback
    discovery.on_discover(on_device_discovered)
    
    # Start listener (responds to broadcasts from others)
    print("Starting discovery listener...")
    discovery.start_listener()
    
    print("Listening for other devices for 10 seconds...")
    print("Other devices scanning the network will be detected.\n")
    
    time.sleep(10)
    
    # Stop listener
    discovery.stop_listener()
    print("\nListener stopped.")


def example_server_mode():
    """Example 4: Running as a discoverable server."""
    from forge_ai.comms.discovery import DeviceDiscovery
    import socket
    import time
    
    print("\n" + "="*60)
    print("Example 4: Discoverable Server Mode")
    print("="*60 + "\n")
    
    # Get hostname for node name
    hostname = socket.gethostname()
    
    # Create discovery instance
    discovery = DeviceDiscovery(
        node_name=f"forge_{hostname}",
        node_port=8765
    )
    
    print(f"Starting discovery listener on port 5001...")
    print(f"Node name: forge_{hostname}")
    print(f"API port: 8765")
    print()
    
    # Start listener
    discovery.start_listener()
    
    print("Server is now discoverable!")
    print("Other devices can find this server by:")
    print("  1. Broadcasting discovery message")
    print("  2. Scanning the network")
    print()
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping server...")
        discovery.stop_listener()
        print("Server stopped.")


def main():
    """Main menu."""
    print("\n" + "="*60)
    print("ForgeAI Device Discovery Examples")
    print("="*60)
    
    print("\nSelect an example:")
    print("  1. Broadcast Discovery (fast, 3 seconds)")
    print("  2. Full Network Scan (thorough, 30-60 seconds)")
    print("  3. Discovery with Callback")
    print("  4. Run as Discoverable Server")
    print("  5. Run all examples")
    print()
    
    try:
        choice = input("Enter choice (1-5): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return
    
    if choice == "1":
        example_broadcast_discovery()
    elif choice == "2":
        example_full_scan()
    elif choice == "3":
        example_with_callback()
    elif choice == "4":
        example_server_mode()
    elif choice == "5":
        example_broadcast_discovery()
        example_full_scan()
        example_with_callback()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
