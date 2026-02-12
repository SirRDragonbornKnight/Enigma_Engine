"""
Device Discovery for Forge Network

Automatically find other Forge nodes on the local network.
Uses multiple discovery methods:
  1. mDNS/Zeroconf (if available)
  2. UDP broadcast
  3. Manual IP scanning
"""

import json
import logging
import socket
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class DeviceDiscovery:
    """
    Discover other Forge nodes on the network.
    """
    
    BROADCAST_PORT = 5001
    DISCOVERY_MESSAGE = b"AI_TESTER_DISCOVER"
    RESPONSE_PREFIX = b"AI_TESTER_NODE:"
    
    def __init__(self, node_name: str, node_port: int = 5000):
        """
        Args:
            node_name: Name of this node
            node_port: Port this node's API is on
        """
        self.node_name = node_name
        self.node_port = node_port
        
        # Discovered devices: {name: {"ip": ..., "port": ..., "last_seen": ...}}
        self.discovered: dict[str, dict] = {}
        
        # Discovery callbacks
        self._on_discover: list[Callable] = []
        
        # Listener thread
        self._listener_thread = None
        self._running = False
    
    def on_discover(self, callback: Callable[[str, dict], None]):
        """
        Register a callback for when a device is discovered.
        
        callback(name, info) will be called.
        """
        self._on_discover.append(callback)
    
    def _notify_discover(self, name: str, info: dict):
        """Notify all callbacks of a discovery."""
        for cb in self._on_discover:
            try:
                cb(name, info)
            except Exception as e:
                # Log but don't fail on callback errors
                logger.warning(f"Discovery callback failed: {e}")
    
    def get_local_ip(self) -> str:
        """Get this machine's local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except OSError:
            return "127.0.0.1"
    
    def start_listener(self):
        """Start listening for discovery broadcasts."""
        if self._running:
            return
        
        self._running = True
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener_thread.start()
        logger.info(f"Discovery listener started on port {self.BROADCAST_PORT}")
    
    def stop_listener(self):
        """Stop the discovery listener."""
        self._running = False
    
    def _listen_loop(self):
        """Main listener loop."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(1.0)
        
        try:
            sock.bind(("", self.BROADCAST_PORT))
        except OSError as e:
            logger.error(f"Could not bind to port {self.BROADCAST_PORT}: {e}")
            return
        
        while self._running:
            try:
                data, addr = sock.recvfrom(1024)
                
                if data == self.DISCOVERY_MESSAGE:
                    # Someone is looking for nodes - respond
                    response = self.RESPONSE_PREFIX + json.dumps({
                        "name": self.node_name,
                        "port": self.node_port,
                    }).encode()
                    sock.sendto(response, addr)
                    
                elif data.startswith(self.RESPONSE_PREFIX):
                    # Got a response from another node
                    info_json = data[len(self.RESPONSE_PREFIX):]
                    try:
                        info = json.loads(info_json.decode())
                        name = info.get("name", "unknown")
                        
                        if name != self.node_name:  # Don't discover ourselves
                            device_info = {
                                "ip": addr[0],
                                "port": info.get("port", 5000),
                                "last_seen": time.time(),
                            }
                            
                            is_new = name not in self.discovered
                            self.discovered[name] = device_info
                            
                            if is_new:
                                logger.info(f"Discovered node: {name} at {addr[0]}:{device_info['port']}")
                                self._notify_discover(name, device_info)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        # Invalid discovery message format
                        pass
                        
            except socket.timeout:
                pass  # Intentionally silent
            except Exception as e:
                if self._running:
                    logger.error(f"Discovery error: {e}")
        
        sock.close()
    
    def broadcast_discover(self, timeout: float = 3.0) -> dict[str, dict]:
        """
        Send a discovery broadcast and collect responses.
        
        Args:
            timeout: How long to wait for responses
            
        Returns:
            Dict of discovered devices
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.5)
        
        # Send broadcast
        try:
            sock.sendto(self.DISCOVERY_MESSAGE, ("<broadcast>", self.BROADCAST_PORT))
        except OSError as e:
            # Try specific broadcast address
            local_ip = self.get_local_ip()
            if "." in local_ip:
                parts = local_ip.split(".")
                broadcast = f"{parts[0]}.{parts[1]}.{parts[2]}.255"
                try:
                    sock.sendto(self.DISCOVERY_MESSAGE, (broadcast, self.BROADCAST_PORT))
                except OSError:
                    pass  # Broadcast failed
        
        # Collect responses
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                data, addr = sock.recvfrom(1024)
                
                if data.startswith(self.RESPONSE_PREFIX):
                    info_json = data[len(self.RESPONSE_PREFIX):]
                    try:
                        info = json.loads(info_json.decode())
                        name = info.get("name", "unknown")
                        
                        if name != self.node_name:
                            self.discovered[name] = {
                                "ip": addr[0],
                                "port": info.get("port", 5000),
                                "last_seen": time.time(),
                            }
                    except (json.JSONDecodeError, KeyError, AttributeError):
                        pass  # Invalid response format
            except socket.timeout:
                pass  # Intentionally silent
        
        sock.close()
        return self.discovered
    
    def scan_network(
        self,
        port: int = 5000,
        subnet: str = None,
        timeout: float = 0.5
    ) -> dict[str, dict]:
        """
        Scan the local network for Forge nodes.
        
        Args:
            port: Port to check for Forge API
            subnet: Subnet to scan (e.g., "192.168.1"). If None, auto-detect
            timeout: Timeout per host
            
        Returns:
            Dict of discovered devices
        """
        import urllib.request
        
        if subnet is None:
            local_ip = self.get_local_ip()
            parts = local_ip.split(".")
            subnet = f"{parts[0]}.{parts[1]}.{parts[2]}"
        
        logger.info("Scanning %s.0/24 for Forge nodes...", subnet)
        
        def check_host(ip):
            try:
                url = f"http://{ip}:{port}/info"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    data = json.loads(response.read().decode())
                    name = data.get("name", "unknown")
                    if name != self.node_name:
                        return name, {
                            "ip": ip,
                            "port": port,
                            "last_seen": time.time(),
                            "model": data.get("model"),
                        }
            except (urllib.error.URLError, json.JSONDecodeError, socket.timeout):
                pass  # Host not reachable or invalid response
            return None, None
        
        # Scan in parallel using threads
        results = []
        threads = []
        
        for i in range(1, 255):
            ip = f"{subnet}.{i}"
            t = threading.Thread(target=lambda ip=ip: results.append(check_host(ip)))
            t.start()
            threads.append(t)
            
            # Limit concurrent threads
            if len(threads) >= 50:
                for t in threads:
                    t.join()
                threads = []
        
        # Wait for remaining threads
        for t in threads:
            t.join()
        
        # Process results
        for name, info in results:
            if name and info:
                self.discovered[name] = info
                logger.info("Found: %s at %s:%s", name, info['ip'], info['port'])
        
        return self.discovered
    
    def get_device_url(self, name: str) -> Optional[str]:
        """Get the URL for a discovered device."""
        if name in self.discovered:
            info = self.discovered[name]
            return f"http://{info['ip']}:{info['port']}"
        return None
    
    def discover_federated_peers(self, timeout: float = 3.0) -> list[dict]:
        """
        Discover devices that support federated learning.
        
        Scans the network for Forge nodes with federated learning enabled.
        
        Args:
            timeout: How long to wait for responses
        
        Returns:
            List of peer devices with federated learning capabilities
        """
        import urllib.request

        # First, discover all nodes
        self.broadcast_discover(timeout)
        
        peers = []
        
        # Check which nodes have federated learning enabled
        for name, info in self.discovered.items():
            try:
                url = f"http://{info['ip']}:{info['port']}/federated/info"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=1.0) as response:
                    data = json.loads(response.read().decode())
                    if data.get("enabled", False):
                        peers.append({
                            "name": name,
                            "ip": info["ip"],
                            "port": info["port"],
                            "device_id": data.get("device_id"),
                            "privacy_level": data.get("privacy_level"),
                            "current_round": data.get("current_round", 0),
                        })
                        logger.info("Found federated peer: %s (privacy: %s)", name, data.get('privacy_level'))
            except (urllib.error.URLError, json.JSONDecodeError, socket.timeout):
                # Node doesn't support federated learning or is unreachable
                pass
        
        return peers


# Convenience function
def discover_enigma_engine_nodes(node_name: str = "scanner", timeout: float = 3.0) -> dict[str, dict]:
    """
    Quick function to discover Forge nodes on the network.
    
    Args:
        node_name: Name for this scanner (to avoid self-discovery)
        timeout: How long to wait for responses
        
    Returns:
        Dict of discovered devices
    """
    discovery = DeviceDiscovery(node_name)
    return discovery.broadcast_discover(timeout)


def discover_federated_peers(node_name: str = "scanner", timeout: float = 3.0) -> list[dict]:
    """
    Quick function to discover federated learning peers.
    
    Args:
        node_name: Name for this scanner
        timeout: How long to wait for responses
    
    Returns:
        List of peer devices with federated learning enabled
    """
    discovery = DeviceDiscovery(node_name)
    return discovery.discover_federated_peers(timeout)


if __name__ == "__main__":
    print("Scanning for Forge nodes...")
    nodes = discover_enigma_engine_nodes()
    
    if nodes:
        print(f"\nFound {len(nodes)} node(s):")
        for name, info in nodes.items():
            print(f"  {name}: {info['ip']}:{info['port']}")
    else:
        print("No nodes found. Make sure other Forge instances are running with discovery enabled.")
