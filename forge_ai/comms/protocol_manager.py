"""
Protocol Configuration Manager

Handles loading and managing connection protocols from JSON config files.
Protocols are stored in data/protocols/ with subfolders for each type:
  - game/   : Game connections
  - robot/  : Robot connections  
  - api/    : API connections

Each protocol config is a JSON file with connection settings.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from ..config import CONFIG


# Protocol types and their default settings
PROTOCOL_DEFAULTS = {
    "websocket": {"port": 8765, "endpoint": "/"},
    "http": {"port": 80, "endpoint": "/api"},
    "tcp": {"port": 9000},
    "udp": {"port": 9000},
    "serial": {"baud": 115200},
    "ros": {"port": 11311},
    "mqtt": {"port": 1883},
    "osc": {"port": 8000},
    "gpio": {},
}


@dataclass
class ProtocolConfig:
    """Represents a protocol configuration."""
    name: str
    protocol: str
    enabled: bool = False
    host: str = "localhost"
    port: int = 0
    endpoint: str = "/"
    baud: int = 115200
    description: str = ""
    commands: Dict[str, str] = None
    headers: Dict[str, str] = None
    extra: Dict[str, Any] = None
    file_path: str = ""
    
    def __post_init__(self):
        if self.commands is None:
            self.commands = {}
        if self.headers is None:
            self.headers = {}
        if self.extra is None:
            self.extra = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            "name": self.name,
            "description": self.description,
            "protocol": self.protocol,
            "enabled": self.enabled,
            "host": self.host,
            "port": self.port,
            "endpoint": self.endpoint,
            "baud": self.baud,
            "commands": self.commands,
            "headers": self.headers,
            **self.extra
        }
    
    @classmethod
    def from_dict(cls, data: Dict, file_path: str = "") -> "ProtocolConfig":
        """Create from dictionary."""
        # Extract known fields
        known = ["name", "description", "protocol", "enabled", "host", 
                 "port", "endpoint", "baud", "commands", "headers"]
        extra = {k: v for k, v in data.items() if k not in known}
        
        return cls(
            name=data.get("name", "Unnamed"),
            protocol=data.get("protocol", "http"),
            enabled=data.get("enabled", False),
            host=data.get("host", "localhost"),
            port=data.get("port", 0),
            endpoint=data.get("endpoint", "/"),
            baud=data.get("baud", 115200),
            description=data.get("description", ""),
            commands=data.get("commands", {}),
            headers=data.get("headers", {}),
            extra=extra,
            file_path=file_path
        )


class ProtocolManager:
    """
    Manages protocol configurations from JSON files.
    
    Usage:
        manager = ProtocolManager()
        
        # Get all game protocols
        games = manager.get_protocols("game")
        
        # Get enabled protocols only
        enabled = manager.get_enabled_protocols()
        
        # Save a new protocol
        manager.save_protocol("game", config)
    """
    
    def __init__(self, base_dir: Path = None):
        """Initialize protocol manager."""
        if base_dir is None:
            base_dir = Path(CONFIG.get("data_dir", "data")) / "protocols"
        
        self.base_dir = Path(base_dir)
        self._ensure_dirs()
        self._protocols: Dict[str, List[ProtocolConfig]] = {
            "game": [],
            "robot": [],
            "api": []
        }
        self.reload()
    
    def _ensure_dirs(self):
        """Create protocol directories if they don't exist."""
        for subdir in ["game", "robot", "api"]:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def reload(self):
        """Reload all protocols from disk."""
        self._protocols = {"game": [], "robot": [], "api": []}
        
        for category in self._protocols.keys():
            cat_dir = self.base_dir / category
            if cat_dir.exists():
                for config_file in cat_dir.glob("*.json"):
                    try:
                        with open(config_file) as f:
                            data = json.load(f)
                        config = ProtocolConfig.from_dict(data, str(config_file))
                        self._protocols[category].append(config)
                    except Exception as e:
                        print(f"[!] Failed to load {config_file}: {e}")
    
    def get_protocols(self, category: str) -> List[ProtocolConfig]:
        """Get all protocols in a category."""
        return self._protocols.get(category, [])
    
    def get_enabled_protocols(self, category: str = None) -> List[ProtocolConfig]:
        """Get enabled protocols, optionally filtered by category."""
        enabled = []
        categories = [category] if category else self._protocols.keys()
        
        for cat in categories:
            for proto in self._protocols.get(cat, []):
                if proto.enabled:
                    enabled.append(proto)
        
        return enabled
    
    def get_protocol_by_name(self, name: str) -> Optional[ProtocolConfig]:
        """Find a protocol by name across all categories."""
        for protos in self._protocols.values():
            for proto in protos:
                if proto.name == name:
                    return proto
        return None
    
    def save_protocol(self, category: str, config: ProtocolConfig) -> str:
        """
        Save a protocol configuration to disk.
        
        Args:
            category: "game", "robot", or "api"
            config: The protocol configuration
            
        Returns:
            Path to saved file
        """
        # Generate filename from name
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" 
                           for c in config.name.lower())
        file_path = self.base_dir / category / f"{safe_name}.json"
        
        with open(file_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        
        config.file_path = str(file_path)
        
        # Reload to update internal state
        self.reload()
        
        return str(file_path)
    
    def delete_protocol(self, config: ProtocolConfig) -> bool:
        """Delete a protocol configuration."""
        if config.file_path and Path(config.file_path).exists():
            Path(config.file_path).unlink()
            self.reload()
            return True
        return False
    
    def list_available_protocols(self) -> List[str]:
        """List all available protocol types."""
        return list(PROTOCOL_DEFAULTS.keys())
    
    def get_protocol_defaults(self, protocol: str) -> Dict:
        """Get default settings for a protocol type."""
        return PROTOCOL_DEFAULTS.get(protocol, {})


# Global instance
_manager: Optional[ProtocolManager] = None


def get_protocol_manager() -> ProtocolManager:
    """Get the global protocol manager instance."""
    global _manager
    if _manager is None:
        _manager = ProtocolManager()
    return _manager
