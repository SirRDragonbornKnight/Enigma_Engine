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
- Auto-discovery finds all nodes on network (UDP broadcast + network scan)
- GUI interface for easy device management

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


## GUI Method (Easiest)

### Using the Network Tab

1. **Launch ForgeAI GUI**
   ```bash
   python run.py --gui
   ```

2. **Navigate to Network Tab**
   - Click on the "Network" tab in the main window

3. **Start Server (to be discoverable)**
   - Set your desired port (default: 8765)
   - Click "Start Server"
   - Your device will now be discoverable by other ForgeAI instances

4. **Discover Other Devices**
   - Choose scan mode:
     - **Broadcast**: Fast UDP discovery (3 seconds, recommended)
     - **Full Scan**: Thorough network scan (slower but more reliable)
   - Click "Scan Network"
   - Found devices will appear in the devices table with:
     - IP address
     - Port
     - Status
     - Device name
     - Model (if available)

5. **Connect to a Device**
   - Click the ">" button next to a discovered device
   - Or manually enter URL and click "Connect"

6. **Manual Device Addition**
   - Click "Add Device" to manually add a device by IP and port
   - Useful for devices on different subnets


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
# Find all Forge nodes on your network
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

### Server Node with Discovery
```python
from forge_ai.comms import ForgeNode, DeviceDiscovery

# Enable auto-discovery listener
discovery = DeviceDiscovery("my_server", port=5000)
discovery.start_listener()
print("Discovery listener active - other devices can find me")

# Start server
node = ForgeNode(name="my_server", port=5000, model_name="my_model")
node.start_server()  # Non-blocking
# or
node.start_server(blocking=True)  # Blocking
```

### Client Node with Discovery
```python
from forge_ai.comms import ForgeNode, DeviceDiscovery

# Discover nodes on network
discovery = DeviceDiscovery("my_client")

# Method 1: Broadcast discovery (fast)
nodes = discovery.broadcast_discover(timeout=3.0)
for name, info in nodes.items():
    print(f"Found: {name} at {info['ip']}:{info['port']}")

# Method 2: Full network scan (thorough)
nodes = discovery.scan_network(port=5000, timeout=0.5)

# Connect to discovered node
if nodes:
    node = ForgeNode(name="my_client")
    first_node = list(nodes.values())[0]
    node.connect_to(f"{first_node['ip']}:{first_node['port']}")
    
    # Ask the server
    response = node.ask_peer("my_server", "What is the meaning of life?")
    print(response)
```

### Discovery Callbacks
```python
from forge_ai.comms import DeviceDiscovery

discovery = DeviceDiscovery("my_client", port=5000)

# Register callback for when devices are found
def on_device_found(name, info):
    print(f"Device discovered: {name} at {info['ip']}:{info['port']}")

discovery.on_discover(on_device_found)

# Start listening for broadcasts
discovery.start_listener()
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


## Discovery Methods Explained

ForgeAI uses multiple discovery methods for maximum compatibility:

### 1. UDP Broadcast Discovery (Default)
- **Speed**: 3 seconds
- **How**: Sends UDP broadcast packet on port 5001
- **Pros**: Very fast, automatic, no configuration needed
- **Cons**: May not work across subnets or with strict firewalls
- **Use when**: Devices are on same subnet/WiFi network

### 2. Network Scan
- **Speed**: 30-60 seconds (depends on network size)
- **How**: Checks each IP in subnet for Forge API endpoint
- **Pros**: More thorough, works across complex networks
- **Cons**: Slower, more network traffic
- **Use when**: Broadcast discovery finds nothing

### 3. Manual Entry
- **Speed**: Instant (if you know the IP)
- **How**: Manually enter IP:port in GUI or code
- **Pros**: Always works, useful for static IPs
- **Cons**: Requires knowing the IP address
- **Use when**: Devices on different subnets or VPN


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

- Same WiFi/LAN for networked mode (broadcast discovery works best on same subnet)
- Ports:
  - 5000 (default API port, configurable)
  - 5001 (UDP discovery port)
- Firewall should allow:
  - UDP broadcast on port 5001
  - TCP connections on your chosen API port
- No network needed for offline sync


## Troubleshooting

### Discovery Not Finding Devices
1. **Try Full Scan**: Switch from Broadcast to Full Scan mode in GUI
2. **Check Firewall**: Ensure port 5001 (UDP) is not blocked
3. **Same Network**: Devices must be on same WiFi/LAN for broadcast
4. **Manual Entry**: Use "Add Device" button if you know the IP

### Can't Connect to Discovered Device
1. **Check Server Running**: Device must have "Start Server" enabled
2. **Port Correct**: Verify port number matches between devices
3. **Firewall**: API port (default 8765) must be open
4. **Ping Test**: Try `ping <device-ip>` to test basic connectivity

### Discovery Finds Self
- This is normal and expected
- Devices marked "This Device" are filtered out automatically


## Files Created

- `forge_ai/comms/network.py` - ForgeNode for multi-device communication
- `forge_ai/comms/discovery.py` - Auto-discovery on network (UDP + scan)
- `forge_ai/comms/memory_sync.py` - Memory synchronization
- `forge_ai/gui/tabs/network_tab.py` - GUI for device management
- `examples/multi_device_example.py` - Example usage
- `tests/test_device_discovery.py` - Discovery system tests
