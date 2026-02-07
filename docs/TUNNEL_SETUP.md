# Tunnel Management Guide

Enigma AI Engine includes a tunnel management system that allows you to expose your local Enigma AI Engine server to the internet. This is useful for:

- Remote access from anywhere
- Mobile app connections
- Demos and presentations
- Team collaboration
- Testing webhooks

## Supported Tunnel Providers

### 1. ngrok (Recommended)
- **Pros**: Most reliable, stable connections, HTTPS by default
- **Cons**: Requires account (free tier available)
- **Installation**: Download from https://ngrok.com/download

```bash
# Install ngrok (Ubuntu/Debian)
snap install ngrok

# Or download binary
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xvzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/
```

### 2. localtunnel
- **Pros**: No account needed, simple setup
- **Cons**: Less stable, connections can drop
- **Installation**: Requires Node.js

```bash
npm install -g localtunnel
```

### 3. bore
- **Pros**: Fast, lightweight, Rust-based
- **Cons**: Smaller community, fewer features
- **Installation**: 

```bash
cargo install bore-cli
```

## Quick Start

### Using CLI

```bash
# Start tunnel with ngrok (requires auth token)
python run.py --tunnel --tunnel-token YOUR_NGROK_TOKEN

# Use localtunnel (no account needed)
python run.py --tunnel --tunnel-provider localtunnel

# Use bore
python run.py --tunnel --tunnel-provider bore

# Tunnel different port
python run.py --tunnel --tunnel-port 8080

# Choose region (ngrok only)
python run.py --tunnel --tunnel-region eu
```

### Using Python API

```python
from enigma_engine.comms.tunnel_manager import TunnelManager

# Create tunnel manager
manager = TunnelManager(
    provider="ngrok",
    auth_token="YOUR_TOKEN",
    region="us"
)

# Start tunnel
url = manager.start_tunnel(port=5000)
print(f"Server exposed at: {url}")

# Stop tunnel
manager.stop_tunnel()
```

### Using Module System

```python
from enigma_engine.modules import ModuleManager

manager = ModuleManager()

# Load tunnel module
manager.load('tunnel', config={
    'provider': 'ngrok',
    'auth_token': 'YOUR_TOKEN',
    'auto_start': True,
    'port': 5000
})

# Get tunnel URL
tunnel_mod = manager.get_module('tunnel')
url = tunnel_mod.instance.get_tunnel_url()
print(f"Tunnel URL: {url}")

# Unload (stops tunnel)
manager.unload('tunnel')
```

## Configuration

### Environment Variables

Set these in your `.env` file or environment:

```bash
# ngrok settings
Enigma AI Engine_TUNNEL_PROVIDER=ngrok
Enigma AI Engine_TUNNEL_TOKEN=your_ngrok_token_here
Enigma AI Engine_TUNNEL_REGION=us

# Or for localtunnel
Enigma AI Engine_TUNNEL_PROVIDER=localtunnel
```

### Config File

Edit `enigma_engine/config/__init__.py`:

```python
CONFIG = {
    # ... existing config ...
    
    "tunnel": {
        "provider": "ngrok",
        "auth_token": "YOUR_TOKEN",
        "region": "us",
        "auto_start": False,
        "port": 5000
    }
}
```

## Getting ngrok Auth Token

1. Sign up at https://ngrok.com (free tier available)
2. Go to https://dashboard.ngrok.com/get-started/your-authtoken
3. Copy your auth token
4. Use it with `--tunnel-token` or set in config

## Regions (ngrok only)

- `us` - United States (default)
- `eu` - Europe
- `ap` - Asia Pacific
- `au` - Australia
- `sa` - South America
- `jp` - Japan
- `in` - India

Choose the region closest to your users for best performance.

## Security Considerations

⚠️ **Important**: When you expose your server to the internet:

1. **Enable API authentication**: Set `require_api_key=True` in config
2. **Use HTTPS**: ngrok provides this by default
3. **Don't expose sensitive data**: Be careful what data your server has access to
4. **Monitor connections**: Keep an eye on who's connecting
5. **Use firewall rules**: If possible, restrict access by IP

## Troubleshooting

### "ngrok not found"

Install ngrok:
```bash
snap install ngrok
# OR
brew install ngrok  # macOS
```

### "localtunnel command not found"

Install Node.js and localtunnel:
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install -g localtunnel
```

### Connection drops frequently

- Try ngrok instead of localtunnel (more stable)
- Check your internet connection
- Verify no firewall is blocking the tunnel

### Port already in use

- Change the port: `--tunnel-port 8080`
- Or stop the service using that port

### ngrok auth token issues

- Make sure you copied the full token
- Check for extra spaces
- Get a fresh token from the dashboard

## Advanced Features

### Auto-reconnect

Tunnels automatically reconnect if the connection drops (up to 5 attempts):

```python
manager = TunnelManager(provider="ngrok", auto_reconnect=True)
```

### Custom Subdomain (Paid Feature)

```bash
python run.py --tunnel --tunnel-subdomain myapp --tunnel-token YOUR_TOKEN
# URL will be: https://myapp.ngrok.io
```

### Monitoring

The tunnel manager includes monitoring:

```python
# Check if tunnel is running
if manager.is_tunnel_running():
    url = manager.get_tunnel_url()
    print(f"Active tunnel: {url}")
```

## Examples

### Demo Server

Expose your server for a live demo:

```bash
# Terminal 1: Start API server
python run.py --serve

# Terminal 2: Start tunnel
python run.py --tunnel --tunnel-token YOUR_TOKEN
```

Share the tunnel URL with your audience!

### Mobile App Development

Expose API for mobile app testing:

```bash
python run.py --serve &
python run.py --tunnel --tunnel-provider localtunnel
```

Use the tunnel URL in your mobile app config.

### Remote Training

Train on one machine, monitor from another:

```bash
# On training server
python run.py --gui &
python run.py --tunnel --tunnel-port 8080
```

Access the GUI from anywhere via the tunnel URL.

## Integration with Other Modules

Tunnel works seamlessly with:

- `api_server` - Expose REST API
- `web_server` - Expose web dashboard
- `gui` - Access GUI remotely (experimental)

## Comparison Table

| Feature | ngrok | localtunnel | bore |
|---------|-------|-------------|------|
| Account Required | Yes (free) | No | No |
| Stability | Excellent | Fair | Good |
| HTTPS | Yes | Yes | No |
| Custom Subdomain | Yes (paid) | Sometimes | No |
| Speed | Fast | Medium | Very Fast |
| Reconnect | Automatic | Manual | Manual |

## Support

- GitHub Issues: https://github.com/SirRDragonbornKnight/enigma_engine/issues
- Discord: [Join our community]
- Documentation: See `docs/` folder

## License

Same as Enigma AI Engine (see LICENSE file)
