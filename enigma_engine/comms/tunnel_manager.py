"""
================================================================================
🌐 TUNNEL MANAGER - EXPOSE SERVER TO THE INTERNET
================================================================================

Manages secure tunnels to expose Enigma AI Engine servers to the internet using services
like ngrok, localtunnel, or bore. Useful for remote access, mobile apps, demos.

📍 FILE: enigma_engine/comms/tunnel_manager.py
🏷️ TYPE: Network Tunneling
🎯 MAIN CLASSES: TunnelManager, TunnelProvider

┌─────────────────────────────────────────────────────────────────────────────┐
│  SUPPORTED TUNNEL PROVIDERS:                                                │
│                                                                             │
│  • ngrok      - Most reliable, requires account (free tier available)      │
│  • localtunnel - Simple, no account needed (less stable)                   │
│  • bore       - Rust-based, fast, no account needed                        │
└─────────────────────────────────────────────────────────────────────────────┘

🚀 USAGE:
    from enigma_engine.comms.tunnel_manager import TunnelManager
    
    # Create manager
    manager = TunnelManager(provider='ngrok', auth_token='your_token')
    
    # Start tunnel
    tunnel_url = manager.start_tunnel(port=5000)
    print(f"Server exposed at: {tunnel_url}")
    
    # Stop tunnel
    manager.stop_tunnel()

🔗 CONNECTED FILES:
    → USES:      enigma_engine/comms/api_server.py (Flask server to expose)
    → USES:      enigma_engine/comms/web_server.py (Web UI to expose)
    ← USED BY:   enigma_engine/modules/registry.py (TunnelModule)
"""

import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TunnelProvider(Enum):
    """Available tunnel providers."""
    NGROK = "ngrok"
    LOCALTUNNEL = "localtunnel"
    BORE = "bore"


@dataclass
class TunnelConfig:
    """Configuration for tunnel."""
    provider: TunnelProvider
    port: int
    auth_token: Optional[str] = None
    region: Optional[str] = None  # For ngrok: us, eu, ap, au, sa, jp, in
    subdomain: Optional[str] = None  # Custom subdomain (requires paid plan)
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5


class TunnelManager:
    """
    Manages secure tunnels to expose local servers to the internet.
    
    Supports multiple tunnel providers with automatic reconnection.
    """
    
    def __init__(
        self,
        provider: str = "ngrok",
        auth_token: Optional[str] = None,
        region: Optional[str] = None,
        subdomain: Optional[str] = None,
        auto_reconnect: bool = True
    ):
        """
        Initialize tunnel manager.
        
        Args:
            provider: Tunnel provider (ngrok, localtunnel, bore)
            auth_token: Authentication token (required for ngrok)
            region: Server region for ngrok (us, eu, ap, etc.)
            subdomain: Custom subdomain (requires paid plan)
            auto_reconnect: Auto-reconnect on disconnect
        """
        self.provider = TunnelProvider(provider)
        self.auth_token = auth_token
        self.region = region
        self.subdomain = subdomain
        self.auto_reconnect = auto_reconnect
        
        self.tunnel_url: Optional[str] = None
        self.tunnel_process: Optional[subprocess.Popen] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_running = False
        self._stop_monitoring = False
        
        logger.info(f"Initialized TunnelManager with provider: {provider}")
    
    def start_tunnel(self, port: int) -> Optional[str]:
        """
        Start tunnel to expose local port.
        
        Args:
            port: Local port to expose
            
        Returns:
            Public tunnel URL or None if failed
        """
        if self.is_running:
            logger.warning("Tunnel already running")
            return self.tunnel_url
        
        logger.info(f"Starting {self.provider.value} tunnel on port {port}")
        
        try:
            if self.provider == TunnelProvider.NGROK:
                return self._start_ngrok(port)
            elif self.provider == TunnelProvider.LOCALTUNNEL:
                return self._start_localtunnel(port)
            elif self.provider == TunnelProvider.BORE:
                return self._start_bore(port)
            else:
                logger.error(f"Unknown provider: {self.provider}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to start tunnel: {e}")
            return None
    
    def _start_ngrok(self, port: int) -> Optional[str]:
        """Start ngrok tunnel."""
        # Check if ngrok is installed
        try:
            subprocess.run(
                ["ngrok", "version"],
                capture_output=True,
                check=True,
                timeout=10
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("ngrok not found. Install from https://ngrok.com/download")
            return None
        
        # Configure auth token if provided
        if self.auth_token:
            try:
                subprocess.run(
                    ["ngrok", "config", "add-authtoken", self.auth_token],
                    capture_output=True,
                    check=True,
                    timeout=15
                )
                logger.info("ngrok auth token configured")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to set ngrok auth token: {e}")
                return None
        
        # Build ngrok command
        cmd = ["ngrok", "http", str(port), "--log=stdout"]
        
        if self.region:
            cmd.extend(["--region", self.region])
        
        if self.subdomain:
            cmd.extend(["--subdomain", self.subdomain])
        
        # Start ngrok process
        try:
            self.tunnel_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for ngrok to start and get URL
            time.sleep(2)
            
            # Get tunnel URL from ngrok API
            tunnel_url = self._get_ngrok_url()
            
            if tunnel_url:
                self.tunnel_url = tunnel_url
                self.is_running = True
                logger.info(f"ngrok tunnel started: {tunnel_url}")
                
                # Start monitoring thread
                if self.auto_reconnect:
                    self._start_monitoring(port)
                
                return tunnel_url
            else:
                logger.error("Failed to get ngrok tunnel URL")
                self.stop_tunnel()
                return None
                
        except Exception as e:
            logger.error(f"Failed to start ngrok: {e}")
            return None
    
    def _get_ngrok_url(self, max_attempts: int = 5) -> Optional[str]:
        """Get tunnel URL from ngrok API."""
        for attempt in range(max_attempts):
            try:
                response = requests.get("http://localhost:4040/api/tunnels", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    tunnels = data.get("tunnels", [])
                    for tunnel in tunnels:
                        if tunnel.get("proto") == "https":
                            return tunnel.get("public_url")
                    # Fallback to http if https not found
                    for tunnel in tunnels:
                        if tunnel.get("proto") == "http":
                            return tunnel.get("public_url")
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} to get ngrok URL failed: {e}")
            
            time.sleep(1)
        
        return None
    
    def _start_localtunnel(self, port: int) -> Optional[str]:
        """Start localtunnel tunnel."""
        # Check if localtunnel is installed
        try:
            subprocess.run(
                ["lt", "--version"],
                capture_output=True,
                check=True,
                shell=False,
                timeout=10
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("localtunnel not found. Install with: npm install -g localtunnel")
            return None
        
        # Build localtunnel command
        cmd = ["lt", "--port", str(port)]
        
        if self.subdomain:
            cmd.extend(["--subdomain", self.subdomain])
        
        # Start localtunnel process
        try:
            self.tunnel_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Wait for localtunnel to start and parse URL from output with timeout
            tunnel_url = None
            max_wait = 10  # seconds
            start_time = time.time()
            
            if self.tunnel_process.stdout:
                while time.time() - start_time < max_wait:
                    # Check if process is still running
                    if self.tunnel_process.poll() is not None:
                        break
                    
                    # Try to read with small timeout
                    try:
                        import select
                        import sys
                        if sys.platform != 'win32':
                            # Unix: use select
                            ready, _, _ = select.select([self.tunnel_process.stdout], [], [], 0.5)
                            if ready:
                                line = self.tunnel_process.stdout.readline()
                                if line:
                                    match = re.search(r'https://[a-z0-9-]+\.loca\.lt', line)
                                    if match:
                                        tunnel_url = match.group(0)
                                        break
                        else:
                            # Windows: use simple readline with timeout check
                            time.sleep(0.5)
                            # On Windows, this might block, but we have overall timeout
                            line = self.tunnel_process.stdout.readline()
                            if line:
                                match = re.search(r'https://[a-z0-9-]+\.loca\.lt', line)
                                if match:
                                    tunnel_url = match.group(0)
                                    break
                    except (ImportError, AttributeError):
                        # Fallback: wait and try
                        time.sleep(0.5)
            
            if tunnel_url:
                self.tunnel_url = tunnel_url
                self.is_running = True
                logger.info(f"localtunnel started: {tunnel_url}")
                
                if self.auto_reconnect:
                    self._start_monitoring(port)
                
                return tunnel_url
            else:
                logger.error("Failed to get localtunnel URL within timeout")
                self.stop_tunnel()
                return None
                
        except Exception as e:
            logger.error(f"Failed to start localtunnel: {e}")
            return None
    
    def _start_bore(self, port: int) -> Optional[str]:
        """Start bore tunnel."""
        # Check if bore is installed
        try:
            subprocess.run(
                ["bore", "--version"],
                capture_output=True,
                check=True,
                timeout=10
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("bore not found. Install from https://github.com/ekzhang/bore")
            return None
        
        # Build bore command
        cmd = ["bore", "local", str(port), "--to", "bore.pub"]
        
        # Start bore process
        try:
            self.tunnel_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Wait for bore to start and parse URL from output with timeout
            tunnel_url = None
            max_wait = 10  # seconds
            start_time = time.time()
            
            if self.tunnel_process.stderr:
                while time.time() - start_time < max_wait:
                    # Check if process is still running
                    if self.tunnel_process.poll() is not None:
                        break
                    
                    # Try to read with small timeout
                    try:
                        import select
                        import sys
                        if sys.platform != 'win32':
                            # Unix: use select
                            ready, _, _ = select.select([self.tunnel_process.stderr], [], [], 0.5)
                            if ready:
                                line = self.tunnel_process.stderr.readline()
                                if line:
                                    match = re.search(r'bore\.pub:[0-9]+', line)
                                    if match:
                                        tunnel_url = f"http://{match.group(0)}"
                                        break
                        else:
                            # Windows: use simple readline with timeout check
                            time.sleep(0.5)
                            # On Windows, this might block, but we have overall timeout
                            line = self.tunnel_process.stderr.readline()
                            if line:
                                match = re.search(r'bore\.pub:[0-9]+', line)
                                if match:
                                    tunnel_url = f"http://{match.group(0)}"
                                    break
                    except (ImportError, AttributeError):
                        # Fallback: wait and try
                        time.sleep(0.5)
            
            if tunnel_url:
                self.tunnel_url = tunnel_url
                self.is_running = True
                logger.info(f"bore tunnel started: {tunnel_url}")
                
                if self.auto_reconnect:
                    self._start_monitoring(port)
                
                return tunnel_url
            else:
                logger.error("Failed to get bore tunnel URL within timeout")
                self.stop_tunnel()
                return None
                
        except Exception as e:
            logger.error(f"Failed to start bore: {e}")
            return None
    
    def _start_monitoring(self, port: int):
        """Start monitoring thread for auto-reconnect."""
        self._stop_monitoring = False
        
        def monitor():
            attempts = 0
            while not self._stop_monitoring and self.is_running:
                time.sleep(10)  # Check every 10 seconds
                
                # Check if process is still running
                if self.tunnel_process and self.tunnel_process.poll() is not None:
                    logger.warning("Tunnel process died, attempting reconnect...")
                    self.is_running = False
                    
                    if attempts < 5:
                        attempts += 1
                        logger.info(f"Reconnect attempt {attempts}/5")
                        time.sleep(5)
                        result = self.start_tunnel(port)
                        if result:
                            attempts = 0  # Reset on success
                    else:
                        logger.error("Max reconnect attempts reached")
                        break
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
    
    def stop_tunnel(self):
        """Stop the tunnel."""
        logger.info("Stopping tunnel...")
        
        self._stop_monitoring = True
        self.is_running = False
        
        if self.tunnel_process:
            try:
                self.tunnel_process.terminate()
                self.tunnel_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.tunnel_process.kill()
            except Exception as e:
                logger.error(f"Error stopping tunnel: {e}")
            finally:
                self.tunnel_process = None
        
        self.tunnel_url = None
        logger.info("Tunnel stopped")
    
    def get_tunnel_url(self) -> Optional[str]:
        """Get current tunnel URL."""
        return self.tunnel_url if self.is_running else None
    
    def is_tunnel_running(self) -> bool:
        """Check if tunnel is running."""
        return self.is_running
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop_tunnel()
        except Exception:
            pass  # Ignore cleanup errors during shutdown


# Singleton instance for easy access
_tunnel_manager: Optional[TunnelManager] = None


def get_tunnel_manager(
    provider: str = "ngrok",
    auth_token: Optional[str] = None,
    **kwargs
) -> TunnelManager:
    """
    Get or create global tunnel manager instance.
    
    Args:
        provider: Tunnel provider (ngrok, localtunnel, bore)
        auth_token: Authentication token for ngrok
        **kwargs: Additional configuration options
        
    Returns:
        TunnelManager instance
    """
    global _tunnel_manager
    
    if _tunnel_manager is None:
        _tunnel_manager = TunnelManager(
            provider=provider,
            auth_token=auth_token,
            **kwargs
        )
    
    return _tunnel_manager


def create_tunnel(port: int, **kwargs) -> Optional[str]:
    """
    Quick helper to create a tunnel.
    
    Args:
        port: Local port to expose
        **kwargs: Tunnel configuration (provider, auth_token, etc.)
        
    Returns:
        Public tunnel URL or None if failed
    """
    manager = get_tunnel_manager(**kwargs)
    return manager.start_tunnel(port)
