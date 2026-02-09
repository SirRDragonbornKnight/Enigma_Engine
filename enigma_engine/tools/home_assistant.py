"""
Home Assistant Integration for Enigma AI Engine

Control smart home devices through Home Assistant.

Features:
- Device discovery
- State monitoring
- Service calls
- Automation triggers
- Scene activation

Usage:
    from enigma_engine.tools.home_assistant import HomeAssistant, get_ha
    
    ha = get_ha()
    ha.connect("http://localhost:8123", "your_token")
    
    # Control lights
    ha.turn_on("light.living_room")
    
    # Get sensor values
    temp = ha.get_state("sensor.temperature")
"""

import logging
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeviceDomain(Enum):
    """Home Assistant device domains."""
    LIGHT = "light"
    SWITCH = "switch"
    SENSOR = "sensor"
    BINARY_SENSOR = "binary_sensor"
    CLIMATE = "climate"
    COVER = "cover"
    MEDIA_PLAYER = "media_player"
    CAMERA = "camera"
    LOCK = "lock"
    VACUUM = "vacuum"
    FAN = "fan"
    SCENE = "scene"
    AUTOMATION = "automation"
    SCRIPT = "script"


@dataclass
class EntityState:
    """Home Assistant entity state."""
    entity_id: str
    state: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    last_changed: str = ""
    last_updated: str = ""


@dataclass
class Device:
    """Home Assistant device."""
    device_id: str
    name: str
    manufacturer: str = ""
    model: str = ""
    entities: List[str] = field(default_factory=list)


@dataclass
class Automation:
    """Home Assistant automation."""
    automation_id: str
    alias: str
    trigger: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class HomeAssistantClient:
    """Client for Home Assistant API."""
    
    def __init__(self, url: str, token: str):
        """
        Initialize client.
        
        Args:
            url: Home Assistant URL (e.g., http://localhost:8123)
            token: Long-lived access token
        """
        self._url = url.rstrip('/')
        self._token = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make API request."""
        try:
            import requests
            
            url = f"{self._url}/api/{endpoint}"
            
            if method == "GET":
                response = requests.get(url, headers=self._headers)
            elif method == "POST":
                response = requests.post(url, headers=self._headers, json=data)
            else:
                return None
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"HA API error: {response.status_code}")
                return None
                
        except ImportError:
            logger.error("requests library not installed")
            return None
        except Exception as e:
            logger.error(f"HA API request failed: {e}")
            return None
    
    def get_states(self) -> List[EntityState]:
        """Get all entity states."""
        data = self._request("GET", "states")
        if not data:
            return []
        
        return [
            EntityState(
                entity_id=s.get("entity_id", ""),
                state=s.get("state", ""),
                attributes=s.get("attributes", {}),
                last_changed=s.get("last_changed", ""),
                last_updated=s.get("last_updated", "")
            )
            for s in data
        ]
    
    def get_state(self, entity_id: str) -> Optional[EntityState]:
        """Get state of specific entity."""
        data = self._request("GET", f"states/{entity_id}")
        if not data:
            return None
        
        return EntityState(
            entity_id=data.get("entity_id", ""),
            state=data.get("state", ""),
            attributes=data.get("attributes", {}),
            last_changed=data.get("last_changed", ""),
            last_updated=data.get("last_updated", "")
        )
    
    def set_state(
        self,
        entity_id: str,
        state: str,
        attributes: Optional[Dict] = None
    ) -> bool:
        """Set entity state."""
        data = {"state": state}
        if attributes:
            data["attributes"] = attributes
        
        result = self._request("POST", f"states/{entity_id}", data)
        return result is not None
    
    def call_service(
        self,
        domain: str,
        service: str,
        entity_id: Optional[str] = None,
        **service_data
    ) -> bool:
        """
        Call Home Assistant service.
        
        Args:
            domain: Service domain (e.g., "light")
            service: Service name (e.g., "turn_on")
            entity_id: Target entity
            **service_data: Additional service data
        """
        data = dict(service_data)
        if entity_id:
            data["entity_id"] = entity_id
        
        result = self._request("POST", f"services/{domain}/{service}", data)
        return result is not None
    
    def fire_event(
        self,
        event_type: str,
        event_data: Optional[Dict] = None
    ) -> bool:
        """Fire Home Assistant event."""
        result = self._request("POST", f"events/{event_type}", event_data or {})
        return result is not None


class HomeAssistant:
    """High-level Home Assistant integration."""
    
    def __init__(self):
        """Initialize integration."""
        self._client: Optional[HomeAssistantClient] = None
        self._entities: Dict[str, EntityState] = {}
        self._devices: Dict[str, Device] = {}
        
        # Callbacks
        self._state_callbacks: Dict[str, List[Callable]] = {}
    
    def connect(self, url: str, token: str) -> bool:
        """
        Connect to Home Assistant.
        
        Args:
            url: Home Assistant URL
            token: Access token
            
        Returns:
            True if connected
        """
        self._client = HomeAssistantClient(url, token)
        
        # Test connection
        states = self._client.get_states()
        if states:
            self._entities = {s.entity_id: s for s in states}
            logger.info(f"Connected to Home Assistant, found {len(states)} entities")
            return True
        
        logger.error("Failed to connect to Home Assistant")
        return False
    
    def refresh(self):
        """Refresh all entity states."""
        if not self._client:
            return
        
        states = self._client.get_states()
        for state in states:
            old_state = self._entities.get(state.entity_id)
            self._entities[state.entity_id] = state
            
            # Check for changes
            if old_state and old_state.state != state.state:
                self._notify_change(state.entity_id, state)
    
    def _notify_change(self, entity_id: str, state: EntityState):
        """Notify callbacks of state change."""
        callbacks = self._state_callbacks.get(entity_id, [])
        for callback in callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")
    
    def on_state_change(
        self,
        entity_id: str,
        callback: Callable[[EntityState], None]
    ):
        """Register callback for state changes."""
        if entity_id not in self._state_callbacks:
            self._state_callbacks[entity_id] = []
        self._state_callbacks[entity_id].append(callback)
    
    # Entity queries
    def get_state(self, entity_id: str) -> Optional[str]:
        """Get entity state value."""
        if self._client:
            state = self._client.get_state(entity_id)
            if state:
                return state.state
        return self._entities.get(entity_id, EntityState(entity_id, "")).state
    
    def get_attribute(
        self,
        entity_id: str,
        attribute: str
    ) -> Optional[Any]:
        """Get entity attribute."""
        state = self._entities.get(entity_id)
        if state:
            return state.attributes.get(attribute)
        return None
    
    def get_entities(
        self,
        domain: Optional[DeviceDomain] = None
    ) -> List[str]:
        """Get list of entity IDs."""
        entities = list(self._entities.keys())
        
        if domain:
            entities = [e for e in entities if e.startswith(f"{domain.value}.")]
        
        return entities
    
    # Light controls
    def turn_on(
        self,
        entity_id: str,
        brightness: Optional[int] = None,
        color_temp: Optional[int] = None,
        rgb_color: Optional[tuple] = None
    ) -> bool:
        """Turn on light/switch."""
        if not self._client:
            return False
        
        domain = entity_id.split(".")[0]
        kwargs = {}
        
        if brightness is not None:
            kwargs["brightness"] = brightness
        if color_temp is not None:
            kwargs["color_temp"] = color_temp
        if rgb_color is not None:
            kwargs["rgb_color"] = list(rgb_color)
        
        return self._client.call_service(domain, "turn_on", entity_id, **kwargs)
    
    def turn_off(self, entity_id: str) -> bool:
        """Turn off light/switch."""
        if not self._client:
            return False
        
        domain = entity_id.split(".")[0]
        return self._client.call_service(domain, "turn_off", entity_id)
    
    def toggle(self, entity_id: str) -> bool:
        """Toggle light/switch."""
        if not self._client:
            return False
        
        domain = entity_id.split(".")[0]
        return self._client.call_service(domain, "toggle", entity_id)
    
    # Climate controls
    def set_temperature(
        self,
        entity_id: str,
        temperature: float,
        hvac_mode: Optional[str] = None
    ) -> bool:
        """Set thermostat temperature."""
        if not self._client:
            return False
        
        kwargs = {"temperature": temperature}
        if hvac_mode:
            kwargs["hvac_mode"] = hvac_mode
        
        return self._client.call_service("climate", "set_temperature", entity_id, **kwargs)
    
    def set_hvac_mode(self, entity_id: str, mode: str) -> bool:
        """Set HVAC mode (heat, cool, auto, off)."""
        if not self._client:
            return False
        
        return self._client.call_service("climate", "set_hvac_mode", entity_id, hvac_mode=mode)
    
    # Cover controls
    def open_cover(self, entity_id: str) -> bool:
        """Open cover/blind."""
        if not self._client:
            return False
        return self._client.call_service("cover", "open_cover", entity_id)
    
    def close_cover(self, entity_id: str) -> bool:
        """Close cover/blind."""
        if not self._client:
            return False
        return self._client.call_service("cover", "close_cover", entity_id)
    
    def set_cover_position(self, entity_id: str, position: int) -> bool:
        """Set cover position (0-100)."""
        if not self._client:
            return False
        return self._client.call_service(
            "cover", "set_cover_position", entity_id, position=position
        )
    
    # Lock controls
    def lock(self, entity_id: str) -> bool:
        """Lock a lock."""
        if not self._client:
            return False
        return self._client.call_service("lock", "lock", entity_id)
    
    def unlock(self, entity_id: str) -> bool:
        """Unlock a lock."""
        if not self._client:
            return False
        return self._client.call_service("lock", "unlock", entity_id)
    
    # Media controls
    def media_play(self, entity_id: str) -> bool:
        """Play media."""
        if not self._client:
            return False
        return self._client.call_service("media_player", "media_play", entity_id)
    
    def media_pause(self, entity_id: str) -> bool:
        """Pause media."""
        if not self._client:
            return False
        return self._client.call_service("media_player", "media_pause", entity_id)
    
    def set_volume(self, entity_id: str, volume: float) -> bool:
        """Set volume (0.0-1.0)."""
        if not self._client:
            return False
        return self._client.call_service(
            "media_player", "volume_set", entity_id, volume_level=volume
        )
    
    # Vacuum controls
    def vacuum_start(self, entity_id: str) -> bool:
        """Start vacuum."""
        if not self._client:
            return False
        return self._client.call_service("vacuum", "start", entity_id)
    
    def vacuum_stop(self, entity_id: str) -> bool:
        """Stop vacuum."""
        if not self._client:
            return False
        return self._client.call_service("vacuum", "stop", entity_id)
    
    def vacuum_return_home(self, entity_id: str) -> bool:
        """Return vacuum to dock."""
        if not self._client:
            return False
        return self._client.call_service("vacuum", "return_to_base", entity_id)
    
    # Scene/automation
    def activate_scene(self, scene_id: str) -> bool:
        """Activate a scene."""
        if not self._client:
            return False
        return self._client.call_service("scene", "turn_on", scene_id)
    
    def trigger_automation(self, automation_id: str) -> bool:
        """Trigger an automation."""
        if not self._client:
            return False
        return self._client.call_service("automation", "trigger", automation_id)
    
    def run_script(self, script_id: str) -> bool:
        """Run a script."""
        if not self._client:
            return False
        return self._client.call_service("script", "turn_on", script_id)
    
    # Convenience methods
    def all_lights_off(self) -> bool:
        """Turn off all lights."""
        if not self._client:
            return False
        return self._client.call_service("light", "turn_off", "all")
    
    def goodnight(self):
        """Goodnight routine - turn off lights, lock doors."""
        # Turn off all lights
        for entity in self.get_entities(DeviceDomain.LIGHT):
            self.turn_off(entity)
        
        # Lock all locks
        for entity in self.get_entities(DeviceDomain.LOCK):
            self.lock(entity)
    
    def welcome_home(self):
        """Welcome home routine."""
        # Turn on entry lights
        for entity in self.get_entities(DeviceDomain.LIGHT):
            if "entry" in entity.lower() or "hall" in entity.lower():
                self.turn_on(entity)


# Mock client for testing
class MockHomeAssistantClient(HomeAssistantClient):
    """Mock client for testing without Home Assistant."""
    
    def __init__(self):
        self._states: Dict[str, EntityState] = {
            "light.living_room": EntityState("light.living_room", "off"),
            "light.bedroom": EntityState("light.bedroom", "off"),
            "switch.tv": EntityState("switch.tv", "off"),
            "sensor.temperature": EntityState("sensor.temperature", "72"),
            "climate.thermostat": EntityState("climate.thermostat", "heat"),
        }
    
    def get_states(self) -> List[EntityState]:
        return list(self._states.values())
    
    def get_state(self, entity_id: str) -> Optional[EntityState]:
        return self._states.get(entity_id)
    
    def call_service(self, domain, service, entity_id=None, **kwargs) -> bool:
        if entity_id and entity_id in self._states:
            if service == "turn_on":
                self._states[entity_id].state = "on"
            elif service == "turn_off":
                self._states[entity_id].state = "off"
        return True


# Global instance
_ha: Optional[HomeAssistant] = None


def get_ha() -> HomeAssistant:
    """Get or create global Home Assistant instance."""
    global _ha
    if _ha is None:
        _ha = HomeAssistant()
    return _ha
