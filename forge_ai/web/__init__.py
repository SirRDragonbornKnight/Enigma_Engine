"""Web package for ForgeAI."""

from .app import run_web
from .server import ForgeWebServer, create_web_server
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
