# Enigma AI Engine Web Interface

Self-hosted web interface for accessing Enigma AI Engine from any device on your local network.

## Features

- **Remote Access**: Access from phones, tablets, and other computers
- **Real-Time Chat**: WebSocket-based instant messaging
- **Mobile Responsive**: Works perfectly on small screens
- **Progressive Web App**: Install as an app on mobile devices
- **Secure Authentication**: Token-based security
- **QR Code Connection**: Easy phone setup via QR code scan
- **Local Network Discovery**: mDNS/Bonjour for easy finding
- **Offline Support**: Service worker caching for offline use

## Quick Start

### Option 1: GUI Settings Tab

1. Open Enigma AI Engine GUI
2. Go to Settings tab
3. Find "Web Interface - Remote Access" section
4. Check "Enable Web Server"
5. Click "Show QR Code for Mobile" and scan with your phone

### Option 2: Command Line

```bash
# Run the test script
python test_web_server.py

# Or use Python directly
python -c "from enigma_engine.web.server import create_web_server; create_web_server().start()"
```

### Option 3: Python Script

```python
from enigma_engine.web.server import create_web_server

# Create and start server
server = create_web_server(
    host="0.0.0.0",      # Accessible from network
    port=8080,           # Port number
    require_auth=True    # Require authentication
)

server.start()
```

## Configuration

Default configuration in `enigma_engine/config/defaults.py`:

```python
"web_interface": {
    "enabled": True,
    "host": "0.0.0.0",              # Listen on all interfaces
    "port": 8080,                   # Port number
    "auto_start": False,            # Don't auto-start with GUI
    "require_auth": True,           # Require token authentication
    "allow_training": False,        # Disable training from web (security)
    "allow_settings_change": True,  # Allow changing settings
    "cors_origins": ["*"],          # Allow all origins
    "max_connections": 10,          # Max concurrent connections
    "enable_discovery": True,       # Enable mDNS discovery
    "token_lifetime_hours": 720     # 30 days token lifetime
}
```

## API Endpoints

### Chat
- `POST /api/chat` - Send message, get response
- `GET /api/conversations` - List conversations
- `GET /api/conversations/{id}` - Get specific conversation
- `DELETE /api/conversations/{id}` - Delete conversation

### Generation
- `POST /api/generate/image` - Generate image
- `POST /api/generate/code` - Generate code
- `POST /api/generate/audio` - Generate audio

### Configuration
- `GET /api/settings` - Get current settings
- `PUT /api/settings` - Update settings
- `GET /api/models` - List available models
- `POST /api/models/switch` - Switch active model

### System
- `GET /health` - Health check
- `GET /api/info` - Server information
- `GET /api/stats` - System statistics
- `GET /api/modules` - List modules
- `POST /api/modules/{id}/toggle` - Enable/disable module

### WebSocket
- `WS /ws/chat` - Real-time chat connection

## Authentication

The web server uses token-based authentication. On first run, a token is generated automatically.

To get your token:
1. Check the console output when server starts
2. Or visit `/qr` endpoint to see QR code with token
3. Token is stored in `memory/web_tokens.json`

Include token in requests:
- Query parameter: `?token=YOUR_TOKEN`
- For WebSocket: `ws://host:port/ws/chat?token=YOUR_TOKEN`

## Mobile Access

### Via QR Code (Easiest)
1. Start the web server
2. Visit `http://YOUR_IP:8080/qr` on your computer
3. Scan QR code with phone camera
4. Phone opens Enigma AI Engine automatically

### Via Manual Entry
1. Find your computer's IP address
2. On phone browser, visit: `http://YOUR_IP:8080?token=YOUR_TOKEN`
3. Bookmark for easy access

### Install as App (PWA)
1. Open Enigma AI Engine in phone browser
2. Use browser's "Add to Home Screen" option
3. Enigma AI Engine appears as an app icon
4. Works offline with cached content

## Local Network Discovery

The server advertises itself via mDNS/Bonjour as "Enigma AI Engine on [ComputerName]".

On supported devices, Enigma AI Engine will appear in:
- Network browser
- Bonjour browser apps
- Service discovery apps

## Security Notes

- **Local Network Only**: By default, server is only accessible on your local network
- **Authentication Required**: Token must be provided for all requests
- **Training Disabled**: Training is disabled by default from web interface
- **HTTPS**: For production, use a reverse proxy with SSL/TLS
- **Firewall**: Ensure port 8080 (or your chosen port) is open

## Troubleshooting

### Server won't start
- Check if port is already in use: `netstat -an | grep 8080`
- Try a different port in configuration
- Check firewall settings

### Can't connect from phone
- Ensure devices are on same network
- Check firewall isn't blocking connections
- Verify IP address is correct
- Try with authentication disabled temporarily

### WebSocket connection fails
- Check browser console for errors
- Ensure WebSocket port is open
- Try connecting via REST API instead

### Dependencies missing
```bash
pip install fastapi uvicorn websockets qrcode[pil] zeroconf
```

## File Structure

```
enigma_engine/web/
├── __init__.py           # Package exports
├── server.py             # FastAPI server
├── auth.py               # Token authentication
├── discovery.py          # mDNS/Bonjour
├── app.py                # Flask app (legacy)
├── static/
│   ├── index.html        # Main web page
│   ├── app.js            # JavaScript client
│   ├── styles.css        # Mobile-responsive CSS
│   ├── sw.js             # Service worker (PWA)
│   ├── manifest.json     # PWA manifest
│   └── icons/            # PWA icons
└── templates/            # Flask templates (legacy)
```

## Advanced Usage

### Custom Handlers

```python
from enigma_engine.web.server import ForgeWebServer

server = ForgeWebServer()

# Add custom endpoint
@server.app.get("/custom")
async def custom_endpoint():
    return {"message": "Custom endpoint"}

server.start()
```

### Integration with Existing Code

```python
from enigma_engine.web import create_web_server
from enigma_engine.core.inference import EnigmaEngine

# Create engine
engine = EnigmaEngine()

# Create server with custom generation
server = create_web_server()

# Override generation method if needed
# (Server uses EnigmaEngine by default)

server.start()
```

## Contributing

See the main CONTRIBUTING.md file for guidelines.

## License

Same as Enigma AI Engine main license.
