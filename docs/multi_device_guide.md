# Multi-Device Communication Guide

## Overview

ForgeAI can run across multiple devices in two modes:

### 1. NETWORKED MODE (Devices Connected)
Devices on the same network can communicate in real-time.

```
┌─────────────────────┐         ┌─────────────────────┐
│  PC (Server)        │◄───────►│  Raspberry Pi       │
│  - RTX 2080 GPU     │ Network │  - Portable         │
│  - Large Model      │         │  - Small Model      │
│  - Training         │         │  - Inference        │
│  - Port 5000        │         │  - Client           │
└─────────────────────┘         └─────────────────────┘
```

**Features:**
- Pi asks PC questions, PC responds with its powerful model
- AI-to-AI conversations between nodes
- Memory sync between devices
- Auto-discovery finds all nodes on network

### 2. DISCONNECTED MODE (No Network)
Transfer models and memories via USB drive or files.

```
┌─────────────┐      ┌─────────┐      ┌─────────────┐
│  Device A   │─────►│   USB   │─────►│  Device B   │
│  Export     │      │  Drive  │      │  Import     │
└─────────────┘      └─────────┘      └─────────────┘
```

**Features:**
- Export trained model as portable package
- Import model on any device
- Export/import memories and conversations
- Sync when devices reconnect


## Quick Start

### Server Mode (on PC)
```bash
# Start server that accepts connections
python examples/multi_device_example.py --server --name my_pc --port 5000

# With a specific model
python examples/multi_device_example.py --server --name my_pc --model my_trained_model
```

### Client Mode (on Pi)
```bash
# Connect to server and chat
python examples/multi_device_example.py --client --name my_pi --connect 192.168.1.100:5000
```

### AI-to-AI Conversation
```bash
# On server device:
python examples/multi_device_example.py --server --name alice --port 5000

# On client device:
python examples/multi_device_example.py --conversation --name bob --connect 192.168.1.100:5000 --turns 10
```

### Auto-Discovery
```bash
# Find all Enigma nodes on your network
python examples/multi_device_example.py --discover
```

### Export Model (for USB transfer)
```bash
# Export model to directory
python examples/multi_device_example.py --export my_model /media/usb/
```

### Import Model (on other device)
```bash
# Import from package
python examples/multi_device_example.py --import-model /media/usb/my_model_package.zip
```


## Python API

### Server Node
```python
from forge_ai.comms import EnigmaNode, DeviceDiscovery

# Enable auto-discovery
discovery = DeviceDiscovery("my_server", port=5000)
discovery.start_listener()

# Start server
node = EnigmaNode(name="my_server", port=5000, model_name="my_model")
node.start_server()  # Non-blocking
# or
node.start_server(blocking=True)  # Blocking
```

### Client Node
```python
from forge_ai.comms import EnigmaNode

node = EnigmaNode(name="my_client")
node.connect_to("192.168.1.100:5000")

# Ask the server
response = node.ask_peer("my_server", "What is the meaning of life?")
print(response)

# Have an AI conversation
conversation = node.start_ai_conversation("my_server", num_turns=5)
```

### Discovery
```python
from forge_ai.comms import discover_enigma_nodes

# Quick discovery
nodes = discover_enigma_nodes()
for name, info in nodes.items():
    print(f"{name}: http://{info['ip']}:{info['port']}")
```

### Memory Sync
```python
from forge_ai.comms import MemorySync, OfflineSync

# Network sync
sync = MemorySync()
sync.sync_with_peer("192.168.1.100:5000", "pc_node")

# Offline sync (USB/file)
OfflineSync.export_to_file("/media/usb/memories.json")
OfflineSync.import_from_file("/media/usb/memories.json")
```

### Model Export/Import
```python
from forge_ai.comms import ModelExporter

# Export
ModelExporter.export_model("my_model", "/media/usb/")

# Import
ModelExporter.import_model("/media/usb/my_model_package.zip")
```


## Use Cases

### 1. PC as Brain, Pi as Interface
- Train large model on PC
- Pi sends voice commands to PC
- PC responds, Pi speaks the response
- Great for desktop AI + mobile terminal

### 2. Distributed Training
- Multiple PCs with GPUs
- Each trains on different data
- Sync models when done
- Combine knowledge

### 3. Multiple Personalities
- Train different models with different personalities
- Have them converse with each other
- See how AI "personalities" interact

### 4. Offline Field Use
- Train model at home
- Export to USB
- Import on laptop for travel
- Sync memories when you return


## Network Requirements

- Same WiFi/LAN for networked mode
- Ports 5000 (API) and 5001 (discovery) should be open
- No network needed for offline sync


## Files Created

- `forge_ai/comms/network.py` - EnigmaNode for multi-device communication
- `forge_ai/comms/discovery.py` - Auto-discovery on network
- `forge_ai/comms/memory_sync.py` - Memory synchronization
- `examples/multi_device_example.py` - Example usage
