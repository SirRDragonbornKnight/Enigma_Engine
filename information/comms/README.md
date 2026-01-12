# AI Tester Comms - Multi-Device Communication

The `comms` package provides everything needed for multi-device AI communication, from simple API servers to full peer-to-peer AI networks.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    comms                             │
├──────────────┬──────────────┬──────────────┬───────────────┤
│   Network    │   Discovery  │   Sync       │   Servers     │
│ - EnigmaNode │ - UDP Scan   │ - MemorySync │ - WebServer   │
│ - Message    │ - mDNS       │ - OfflineSync│ - MobileAPI   │
│ - Exporter   │ - IP Scan    │              │ - APIServer   │
├──────────────┴──────────────┴──────────────┴───────────────┤
│   Multi-AI             │   Protocols      │   Remote       │
│ - AIConversation       │ - ProtocolMgr    │ - RemoteClient │
│ - AIParticipant        │ - Game/Robot/API │                │
└────────────────────────┴──────────────────┴────────────────┘
```

## Quick Start

### 1. Simple API Server

Run an AI server that responds to HTTP requests:

```python
from ai_tester.comms import create_api_server

# Start server on port 5000
server = create_api_server()
# Server is now running!
# POST /generate with {"prompt": "Hello"} to get a response
```

### 2. Web Interface (Phone/Browser)

Full web interface with WebSocket support:

```python
from ai_tester.comms import create_web_server

server = create_web_server(port=5000)
server.run()
# Open http://YOUR_IP:5000 in browser
# Or visit http://YOUR_IP:5000/qr for QR code to scan
```

### 3. Multi-Device Network

Connect multiple AI Tester instances:

```python
from ai_tester.comms import EnigmaNode

# On Device 1 (Server)
node1 = EnigmaNode(name="desktop", port=5000)
node1.start_server()

# On Device 2 (Client)
node2 = EnigmaNode(name="laptop", port=5001)
node2.start_server()  # Also run a server
node2.connect_to("192.168.1.100:5000")  # Connect to desktop

# Ask the other device
response = node2.ask_peer("desktop", "What is 2+2?")
```

### 4. Device Discovery

Automatically find other AI Tester nodes:

```python
from ai_tester.comms import discover_enigma_nodes

# Find all AI Tester nodes on the network
nodes = discover_enigma_nodes()
for name, info in nodes.items():
    print(f"Found: {name} at {info['ip']}:{info['port']}")
```

### 5. AI-to-AI Conversations

Let multiple AIs talk to each other:

```python
from ai_tester.comms import AIConversation

conv = AIConversation()
conv.add_participant("Alice", personality="friendly and helpful")
conv.add_participant("Bob", personality="curious and inquisitive")

for exchange in conv.converse("Let's discuss AI.", num_turns=5):
    print(f"{exchange['speaker']}: {exchange['message']}")
```

## Components

### AI TesterNode
The core multi-device communication class. Each node can:
- Run as a server (accept connections)
- Connect to other nodes (as client)
- Send/receive messages
- Generate AI responses
- Start AI-to-AI conversations

### DeviceDiscovery
Automatically discover other AI Tester nodes on your network using:
- UDP broadcast (fast, works on most networks)
- IP scanning (slower but more reliable)
- Manual connection (always works)

### MemorySync
Sync conversations and memories between devices:
- **Full sync**: Copy everything to peer
- **Delta sync**: Only send new items
- **Offline sync**: Export to USB/file for disconnected devices

### WebServer
Beautiful web interface for phone/browser access:
- Real-time WebSocket chat
- REST API fallback
- QR code for easy phone connection
- Dark theme, mobile-optimized

### MobileAPI  
Lightweight API designed for mobile apps:
- Compact JSON responses
- Voice input/output
- Per-device conversation context
- Flutter and React Native templates included

### ProtocolManager
Manage connection protocols for games, robots, and external APIs:
- Load/save protocol configs as JSON
- Support for WebSocket, HTTP, TCP, UDP, Serial, ROS, MQTT, OSC

### RemoteClient
Simple client for connecting to any AI Tester API server:

```python
from ai_tester.comms import RemoteClient

client = RemoteClient("http://192.168.1.100:5000")
if client.is_available():
    response = client.generate("Hello!")
    print(response)
```

## API Endpoints

All servers expose these standard endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface (WebServer only) |
| `/health` | GET | Server health check |
| `/info` | GET | Server information |
| `/generate` | POST | Generate AI response |
| `/connect` | POST | Register peer connection |
| `/message` | POST | Send message to node |

### Example: Generate Request

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_gen": 50}'
```

Response:
```json
{
  "text": "I'm doing well, thank you for asking!",
  "from": "ai_tester"
}
```

## Model Export/Import

Share models between devices:

```python
from ai_tester.comms import ModelExporter

# Export a model as portable package
ModelExporter.export_model("my_model", "/path/to/output")
# Creates: my_model_package.zip

# Import on another device
ModelExporter.import_model("/path/to/my_model_package.zip")
```

## Memory Sync Example

Keep multiple devices in sync:

```python
from ai_tester.comms import MemorySync

# On Device 1
sync = MemorySync()
sync.sync_with_peer("http://192.168.1.101:5000", "laptop")

# For offline devices (USB transfer)
from ai_tester.comms import OfflineSync

OfflineSync.export_to_file("/usb/ai_tester_memories.json")
# Copy USB to other device, then:
OfflineSync.import_from_file("/usb/ai_tester_memories.json")
```

## Protocol Configurations

Store connection configs in `data/protocols/`:

```
data/protocols/
├── game/
│   └── minecraft.json
├── robot/
│   └── ros_arm.json
└── api/
    └── weather_service.json
```

Example protocol config:
```json
{
  "name": "Minecraft Server",
  "protocol": "websocket",
  "host": "localhost",
  "port": 25565,
  "enabled": true,
  "commands": {
    "say": "/say {message}",
    "time": "/time set {value}"
  }
}
```

## Files in This Package

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `network.py` | EnigmaNode, Message, ModelExporter |
| `discovery.py` | DeviceDiscovery, network scanning |
| `memory_sync.py` | MemorySync, OfflineSync |
| `multi_ai.py` | AIConversation, AIParticipant |
| `protocol_manager.py` | ProtocolManager, ProtocolConfig |
| `api_server.py` | Simple Flask API server |
| `web_server.py` | WebServer with WebSocket |
| `mobile_api.py` | MobileAPI for apps |
| `remote_client.py` | RemoteClient for calling servers |

## Requirements

Core functionality (no extra dependencies):
- `network.py` - Uses urllib (built-in)
- `discovery.py` - Uses socket (built-in)
- `memory_sync.py` - Uses urllib (built-in)
- `multi_ai.py` - Uses urllib (built-in)
- `remote_client.py` - Uses urllib (built-in)

Optional (for web features):
- `flask` - For API servers
- `flask-cors` - For cross-origin requests
- `flask-socketio` - For WebSocket support
- `qrcode[pil]` - For QR code generation

Install all web features:
```bash
pip install flask flask-cors flask-socketio qrcode[pil]
```

## See Also

- [Multi-Device Guide](../../docs/multi_device_guide.md) - Detailed setup instructions
- [WEB_MOBILE.md](../../docs/WEB_MOBILE.md) - Phone/browser connection guide
- [API Examples](../../examples/api_client_example.py) - API usage examples
- [Multi-Device Example](../../examples/multi_device_example.py) - Network setup example
