"""
Example: Multi-Device Communication

This example shows how to run Enigma across multiple devices.

SCENARIO 1: Pi talks to PC
===========================
Your PC (with GPU) runs the main model, your Pi sends it questions.

On PC (192.168.1.100):
    python examples/multi_device_example.py --server --name pc_brain --port 5000

On Pi:
    python examples/multi_device_example.py --client --name pi_client --connect 192.168.1.100:5000


SCENARIO 2: AI-to-AI Conversation
=================================
Two Enigma instances have a conversation with each other.

On Device 1:
    python examples/multi_device_example.py --server --name alice --port 5000

On Device 2:
    python examples/multi_device_example.py --conversation --name bob --connect 192.168.1.100:5000


SCENARIO 3: Auto-Discovery
==========================
Find all Enigma nodes on your network.

    python examples/multi_device_example.py --discover


SCENARIO 4: Disconnected Sync
=============================
Export your model to a USB drive, import on another device.

Export:
    python examples/multi_device_example.py --export my_model /media/usb/

Import on other device:
    python examples/multi_device_example.py --import /media/usb/my_model_package.zip
"""

import sys
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_server(name: str, port: int, model_name: str = None):
    """Run as a server node."""
    from enigma_engine.comms import ForgeNode, DeviceDiscovery
    
    print(f"\n{'='*60}")
    print(f"Starting Enigma Server: {name}")
    print(f"Port: {port}")
    print(f"Model: {model_name or 'default'}")
    print(f"{'='*60}\n")
    
    # Start discovery listener
    discovery = DeviceDiscovery(name, port)
    discovery.start_listener()
    print("Discovery listener active - other nodes can find me")
    
    # Start server
    node = ForgeNode(name=name, port=port, model_name=model_name)
    node.start_server(blocking=True)


def run_client(name: str, server_url: str):
    """Run as a client, connect to server."""
    from enigma_engine.comms import ForgeNode
    
    print(f"\n{'='*60}")
    print(f"Starting Enigma Client: {name}")
    print(f"Connecting to: {server_url}")
    print(f"{'='*60}\n")
    
    node = ForgeNode(name=name)
    
    if node.connect_to(server_url):
        print("\nConnected! Type messages to send to the server.")
        print("Type 'quit' to exit.\n")
        
        peer_name = list(node.peers.keys())[0]
        
        while True:
            try:
                prompt = input("You: ").strip()
                if prompt.lower() == 'quit':
                    break
                if not prompt:
                    continue
                
                response = node.ask_peer(peer_name, prompt)
                print(f"{peer_name}: {response}\n")
                
            except KeyboardInterrupt:
                break
    else:
        print("Failed to connect to server")


def run_conversation(name: str, server_url: str, turns: int = 5):
    """Start an AI-to-AI conversation."""
    from enigma_engine.comms import ForgeNode
    
    print(f"\n{'='*60}")
    print(f"AI-to-AI Conversation")
    print(f"This node: {name}")
    print(f"Peer: {server_url}")
    print(f"Turns: {turns}")
    print(f"{'='*60}\n")
    
    node = ForgeNode(name=name)
    
    if node.connect_to(server_url):
        peer_name = list(node.peers.keys())[0]
        
        initial = input("Starting message (or press Enter for default): ").strip()
        if not initial:
            initial = "Hello! I'm an AI. Let's have a conversation about anything interesting."
        
        conversation = node.start_ai_conversation(
            peer_name,
            initial_prompt=initial,
            num_turns=turns
        )
        
        # Save conversation
        import json
        output = Path("data/ai_conversations")
        output.mkdir(parents=True, exist_ok=True)
        
        import time
        filename = output / f"conversation_{int(time.time())}.json"
        with open(filename, "w") as f:
            json.dump(conversation, f, indent=2)
        print(f"\nConversation saved to {filename}")
    else:
        print("Failed to connect")


def discover_nodes():
    """Discover Enigma nodes on the network."""
    from enigma_engine.comms import DeviceDiscovery
    
    print("\n" + "="*60)
    print("Discovering Enigma Nodes on Network")
    print("="*60 + "\n")
    
    discovery = DeviceDiscovery("scanner")
    
    print("Method 1: UDP Broadcast...")
    broadcast_results = discovery.broadcast_discover(timeout=3.0)
    
    if broadcast_results:
        print(f"\nFound {len(broadcast_results)} node(s) via broadcast:")
        for name, info in broadcast_results.items():
            print(f"  {name}: http://{info['ip']}:{info['port']}")
    else:
        print("No nodes found via broadcast.")
        print("\nTrying network scan (this may take a while)...")
        scan_results = discovery.scan_network(timeout=0.3)
        
        if scan_results:
            print(f"\nFound {len(scan_results)} node(s) via scan:")
            for name, info in scan_results.items():
                print(f"  {name}: http://{info['ip']}:{info['port']}")
        else:
            print("No nodes found.")
    
    print()


def export_model(model_name: str, output_dir: str):
    """Export a model for transfer."""
    from enigma_engine.comms import ModelExporter
    
    print(f"\nExporting model '{model_name}' to {output_dir}...")
    
    try:
        path = ModelExporter.export_model(model_name, output_dir)
        print(f"\nSuccess! Package created: {path}")
        print(f"Copy this file to another device and import it.")
    except Exception as e:
        print(f"Error: {e}")


def import_model(package_path: str):
    """Import a model from package."""
    from enigma_engine.comms import ModelExporter
    
    print(f"\nImporting model from {package_path}...")
    
    try:
        name = ModelExporter.import_model(package_path)
        print(f"\nSuccess! Model imported as '{name}'")
    except Exception as e:
        print(f"Error: {e}")


def export_memories(output_path: str):
    """Export memories for offline sync."""
    from enigma_engine.comms import OfflineSync
    
    print(f"\nExporting memories to {output_path}...")
    path = OfflineSync.export_to_file(output_path)
    print(f"Done! File saved to: {path}")


def import_memories(input_path: str):
    """Import memories from file."""
    from enigma_engine.comms import OfflineSync
    
    print(f"\nImporting memories from {input_path}...")
    count = OfflineSync.import_from_file(input_path)
    print(f"Done! Imported {count} memories.")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Device Enigma Communication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--server", action="store_true", help="Run as server")
    mode.add_argument("--client", action="store_true", help="Run as client")
    mode.add_argument("--conversation", action="store_true", help="AI-to-AI conversation")
    mode.add_argument("--discover", action="store_true", help="Discover nodes on network")
    mode.add_argument("--export", metavar="MODEL", help="Export model for transfer")
    mode.add_argument("--import-model", metavar="PATH", help="Import model package")
    mode.add_argument("--export-memories", metavar="PATH", help="Export memories to file")
    mode.add_argument("--import-memories", metavar="PATH", help="Import memories from file")
    
    # Options
    parser.add_argument("--name", "-n", default="enigma_engine_node", help="Node name")
    parser.add_argument("--port", "-p", type=int, default=5000, help="Server port")
    parser.add_argument("--connect", "-c", help="Server URL to connect to")
    parser.add_argument("--model", "-m", help="Model name from registry")
    parser.add_argument("--turns", "-t", type=int, default=5, help="Conversation turns")
    parser.add_argument("output", nargs="?", help="Output path for export")
    
    args = parser.parse_args()
    
    if args.server:
        run_server(args.name, args.port, args.model)
    
    elif args.client:
        if not args.connect:
            print("Error: --connect required for client mode")
            sys.exit(1)
        run_client(args.name, args.connect)
    
    elif args.conversation:
        if not args.connect:
            print("Error: --connect required for conversation mode")
            sys.exit(1)
        run_conversation(args.name, args.connect, args.turns)
    
    elif args.discover:
        discover_nodes()
    
    elif args.export:
        if not args.output:
            print("Error: output path required for export")
            sys.exit(1)
        export_model(args.export, args.output)
    
    elif args.import_model:
        import_model(args.import_model)
    
    elif args.export_memories:
        export_memories(args.export_memories)
    
    elif args.import_memories:
        import_memories(args.import_memories)


if __name__ == "__main__":
    main()
