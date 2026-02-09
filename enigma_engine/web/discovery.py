"""
Local Network Discovery for Enigma AI Engine

Uses mDNS/Bonjour to advertise Enigma AI Engine service on the local network,
making it easy to discover from phones, tablets, and other computers.
"""

import logging
import socket
from typing import Optional

logger = logging.getLogger(__name__)


class LocalDiscovery:
    """
    Advertise Enigma AI Engine service on local network using mDNS/Bonjour.
    
    Makes Enigma AI Engine discoverable as "Enigma AI Engine on [ComputerName]" in network browsers.
    """
    
    def __init__(self):
        self.zeroconf = None
        self.service_info = None
        self._hostname = socket.gethostname()
    
    def advertise(self, port: int, version: str = "1.0") -> bool:
        """
        Advertise Enigma AI Engine service on local network.
        
        Args:
            port: Port number the web server is running on
            version: Version string for the service
            
        Returns:
            True if service was advertised successfully
        """
        try:
            import socket

            from zeroconf import ServiceInfo, Zeroconf

            # Get local IP
            local_ip = self._get_local_ip()
            if not local_ip:
                logger.warning("Could not determine local IP address")
                return False
            
            # Create service info
            service_name = f"Enigma AI Engine on {self._hostname}"
            service_type = "_Enigma AI Engine._tcp.local."
            
            self.service_info = ServiceInfo(
                service_type,
                f"{service_name}.{service_type}",
                addresses=[socket.inet_aton(local_ip)],
                port=port,
                properties={
                    'path': '/',
                    'version': version,
                    'hostname': self._hostname
                },
                server=f"{self._hostname}.local."
            )
            
            # Register service
            self.zeroconf = Zeroconf()
            self.zeroconf.register_service(self.service_info)
            
            logger.info(f"Enigma AI Engine service advertised on {local_ip}:{port}")
            return True
            
        except ImportError:
            logger.warning("zeroconf not available - install with: pip install zeroconf")
            return False
        except Exception as e:
            logger.error(f"Failed to advertise service: {e}")
            return False
    
    def stop(self):
        """Stop advertising the service."""
        if self.zeroconf and self.service_info:
            try:
                self.zeroconf.unregister_service(self.service_info)
                self.zeroconf.close()
                logger.info("Stopped advertising Enigma AI Engine service")
            except Exception as e:
                logger.error(f"Error stopping service advertisement: {e}")
    
    def _get_local_ip(self) -> Optional[str]:
        """
        Get the local IP address.
        
        Returns:
            Local IP address or None if unavailable
        """
        s = None
        try:
            # Create a socket to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            return local_ip
        except Exception as e:
            logger.warning(f"Could not determine local IP: {e}")
            return None
        finally:
            if s:
                s.close()


def get_local_ip() -> str:
    """
    Get the local IP address of this machine.
    
    Returns:
        Local IP address as string, or "localhost" if unavailable
    """
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        return ip
    except Exception:
        return "localhost"
    finally:
        if s:
            s.close()
