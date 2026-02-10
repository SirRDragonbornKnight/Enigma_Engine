"""
================================================================================
FEDERATION DISCOVERY - FIND AVAILABLE FEDERATIONS
================================================================================

Discover available federations on the network.
Like device discovery but for federations.

FILE: enigma_engine/federated/discovery.py
TYPE: Federation Discovery
MAIN CLASS: FederationDiscovery

HOW IT WORKS:
    - Broadcasts federation availability
    - Listens for other federations
    - Returns list of discovered federations

USAGE:
    discovery = FederationDiscovery()
    federations = discovery.discover_federations()
"""

import json
import logging
import socket
import threading
import time
from typing import Optional

from .federation import FederationInfo, FederationMode

logger = logging.getLogger(__name__)


class FederationDiscovery:
    """
    Discover available federations.
    
    Like device discovery but for federations.
    """
    
    def __init__(self, discovery_port: int = 5555):
        """
        Initialize federation discovery.
        
        Args:
            discovery_port: UDP port for discovery broadcast
        """
        self.discovery_port = discovery_port
        self.discovered_federations: dict[str, FederationInfo] = {}
        self.advertised_federations: dict[str, FederationInfo] = {}
        
        self._listener_thread: Optional[threading.Thread] = None
        self._broadcaster_thread: Optional[threading.Thread] = None
        self._running = False
        
        logger.info(f"Initialized federation discovery on port {discovery_port}")
    
    def start(self):
        """Start discovery service."""
        if self._running:
            logger.warning("Discovery already running")
            return
        
        self._running = True
        
        # Start listener thread
        self._listener_thread = threading.Thread(
            target=self._listen_for_federations,
            daemon=True
        )
        self._listener_thread.start()
        
        # Start broadcaster thread
        self._broadcaster_thread = threading.Thread(
            target=self._broadcast_federations,
            daemon=True
        )
        self._broadcaster_thread.start()
        
        logger.info("Federation discovery started")
    
    def stop(self):
        """Stop discovery service."""
        self._running = False
        
        if self._listener_thread:
            self._listener_thread.join(timeout=2)
        
        if self._broadcaster_thread:
            self._broadcaster_thread.join(timeout=2)
        
        logger.info("Federation discovery stopped")
    
    def discover_federations(self, timeout: float = 5.0) -> list[FederationInfo]:
        """
        Find available federations on network.
        
        Args:
            timeout: How long to search (seconds)
        
        Returns:
            List of discovered federations
        """
        logger.info(f"Discovering federations (timeout={timeout}s)")
        
        # Clear old discoveries
        self.discovered_federations.clear()
        
        # Start discovery if not running
        was_running = self._running
        if not was_running:
            self.start()
        
        # Wait for timeout
        time.sleep(timeout)
        
        # Stop if we started it
        if not was_running:
            self.stop()
        
        federations = list(self.discovered_federations.values())
        logger.info(f"Discovered {len(federations)} federations")
        
        return federations
    
    def advertise_federation(self, federation: FederationInfo):
        """
        Advertise federation for others to join.
        
        Args:
            federation: Federation to advertise
        """
        self.advertised_federations[federation.id] = federation
        logger.info(f"Advertising federation '{federation.name}' ({federation.id})")
        
        # Ensure discovery is running
        if not self._running:
            self.start()
    
    def stop_advertising(self, federation_id: str):
        """
        Stop advertising a federation.
        
        Args:
            federation_id: Federation ID to stop advertising
        """
        if federation_id in self.advertised_federations:
            del self.advertised_federations[federation_id]
            logger.info(f"Stopped advertising federation {federation_id}")
    
    def _listen_for_federations(self):
        """
        Listen for federation announcements.
        
        Runs in background thread.
        """
        sock = None
        try:
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to localhost only for security
            # Only listen on localhost to prevent external access
            sock.bind(('127.0.0.1', self.discovery_port))
            sock.settimeout(1.0)
            
            logger.debug("Listening for federation broadcasts")
            
            while self._running:
                try:
                    data, addr = sock.recvfrom(4096)
                    
                    # Parse message
                    message = json.loads(data.decode('utf-8'))
                    
                    if message.get('type') == 'federation_announce':
                        self._handle_announcement(message, addr)
                
                except socket.timeout:
                    continue
                except json.JSONDecodeError:
                    logger.debug("Invalid JSON received")
                except Exception as e:
                    logger.debug(f"Error receiving broadcast: {e}")
            
        except Exception as e:
            logger.error(f"Error in listener thread: {e}")
        finally:
            if sock:
                sock.close()
    
    def _broadcast_federations(self):
        """
        Broadcast advertised federations.
        
        Runs in background thread.
        """
        sock = None
        try:
            # Create UDP socket for broadcasting
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            logger.debug("Broadcasting federations")
            
            while self._running:
                # Broadcast each advertised federation
                for federation_id, federation in self.advertised_federations.items():
                    try:
                        message = {
                            'type': 'federation_announce',
                            'federation': federation.to_dict(),
                        }
                        
                        data = json.dumps(message).encode('utf-8')
                        sock.sendto(data, ('<broadcast>', self.discovery_port))
                    
                    except Exception as e:
                        logger.debug(f"Error broadcasting federation {federation_id}: {e}")
                
                # Broadcast every 2 seconds
                time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error in broadcaster thread: {e}")
        finally:
            if sock:
                sock.close()
    
    def _handle_announcement(self, message: dict, addr: tuple):
        """
        Handle federation announcement.
        
        Args:
            message: Announcement message
            addr: Source address
        """
        try:
            federation_data = message.get('federation', {})
            federation = FederationInfo.from_dict(federation_data)
            
            # Store discovered federation
            self.discovered_federations[federation.id] = federation
            
            logger.debug(
                f"Discovered federation '{federation.name}' from {addr[0]}"
            )
            
        except Exception as e:
            logger.debug(f"Error handling announcement: {e}")
    
    def get_discovered_federations(self) -> list[FederationInfo]:
        """
        Get list of currently discovered federations.
        
        Returns:
            List of federations
        """
        return list(self.discovered_federations.values())
    
    def get_advertised_federations(self) -> list[FederationInfo]:
        """
        Get list of federations being advertised.
        
        Returns:
            List of federations
        """
        return list(self.advertised_federations.values())


class FederationDirectory:
    """
    Central directory of federations.
    
    Optional centralized discovery for public federations.
    """
    
    def __init__(self, server_url: Optional[str] = None):
        """
        Initialize federation directory.
        
        Args:
            server_url: URL of central directory server (optional)
        """
        self.server_url = server_url
        self.federations: dict[str, FederationInfo] = {}
        
    def register_federation(self, federation: FederationInfo):
        """
        Register federation with directory.
        
        Args:
            federation: Federation to register
        """
        self.federations[federation.id] = federation
        logger.info(f"Registered federation '{federation.name}' with directory")
        
        # In real implementation, send to server
    
    def unregister_federation(self, federation_id: str):
        """
        Unregister federation from directory.
        
        Args:
            federation_id: Federation ID to unregister
        """
        if federation_id in self.federations:
            del self.federations[federation_id]
            logger.info(f"Unregistered federation {federation_id} from directory")
    
    def search_federations(
        self,
        mode: Optional[FederationMode] = None,
        name_filter: Optional[str] = None
    ) -> list[FederationInfo]:
        """
        Search for federations in directory.
        
        Args:
            mode: Filter by federation mode
            name_filter: Filter by name (substring match)
        
        Returns:
            List of matching federations
        """
        results = list(self.federations.values())
        
        # Apply filters
        if mode:
            results = [f for f in results if f.mode == mode]
        
        if name_filter:
            results = [f for f in results if name_filter.lower() in f.name.lower()]
        
        return results
