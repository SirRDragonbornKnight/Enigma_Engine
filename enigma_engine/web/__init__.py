"""Web package for Enigma AI Engine."""

# Import conditionally to avoid errors when Flask is not available
try:
    from .app import run_web
except Exception:
    run_web = None

# FastAPI server (optional - requires fastapi package)
try:
    from .server import ForgeWebServer, create_web_server
except ImportError:
    ForgeWebServer = None
    create_web_server = None

from .auth import WebAuth, get_auth
from .discovery import LocalDiscovery, get_local_ip

__all__ = [
    'run_web',
    'ForgeWebServer',
    'create_web_server',
    'WebAuth',
    'get_auth',
    'LocalDiscovery',
    'get_local_ip'
]
