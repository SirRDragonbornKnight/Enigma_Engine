#!/usr/bin/env python3
"""
API Client Example

Shows how to use the AI Tester REST API from Python.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_tester.comms.remote_client import RemoteClient


def main():
    # Connect to API server (start with: python run.py --serve)
    client = RemoteClient("http://127.0.0.1:5000")
    
    print("=== Enigma API Client ===")
    print("Make sure the server is running: python run.py --serve")
    print()
    
    try:
        # Simple generation
        response = client.generate(
            prompt="Hello, my name is",
            max_gen=20,
            temperature=0.8,
        )
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the API server is running.")


if __name__ == "__main__":
    main()
