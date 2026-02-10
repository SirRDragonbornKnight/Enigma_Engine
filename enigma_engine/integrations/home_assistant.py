"""
Home Assistant Integration
===========================

Control smart home devices through conversation via Home Assistant.

Features:
- Connect to Home Assistant instance
- Query device states (lights, switches, sensors)
- Control devices (turn on/off, set brightness, etc.)
- Scene activation
- Automation triggers
- Natural language device commands

Setup:
    1. Get a Long-Lived Access Token from Home Assistant
       (Profile -> Long-Lived Access Tokens -> Create Token)
    2. Set HASS_URL and HASS_TOKEN environment variables, or
       pass them to HomeAssistant() constructor

Usage:
    from enigma_engine.integrations.home_assistant import HomeAssistant
    
    hass = HomeAssistant(
        url="http://homeassistant.local:8123",
        token="your_long_lived_token"
    )
    
    # Get all lights
    lights = hass.get_entities(domain="light")
    
    # Turn on a light
    hass.turn_on("light.living_room", brightness=200)
    
    # Natural language control
    hass.execute_command("Turn off all the lights in the bedroom")
    
    # Register as Enigma tools
    tools = hass.as_enigma_tools()
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check for requests
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    logger.warning("requests not installed - Home Assistant integration unavailable")


@dataclass
class Entity:
    """Represents a Home Assistant entity."""
    entity_id: str
    state: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    last_changed: str = ""
    last_updated: str = ""
    
    @property
    def domain(self) -> str:
        """Get entity domain (e.g., 'light', 'switch')."""
        return self.entity_id.split(".")[0]
    
    @property
    def name(self) -> str:
        """Get friendly name or entity name."""
        return self.attributes.get("friendly_name", self.entity_id.split(".")[-1])
    
    @property
    def is_on(self) -> bool:
        """Check if entity is on."""
        return self.state.lower() in ("on", "true", "home", "playing", "open")
    
    @property
    def is_off(self) -> bool:
        """Check if entity is off."""
        return self.state.lower() in ("off", "false", "away", "idle", "closed")
    
    @property
    def is_unavailable(self) -> bool:
        """Check if entity is unavailable."""
        return self.state.lower() in ("unavailable", "unknown")
    
    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "state": self.state,
            "friendly_name": self.name,
            "domain": self.domain,
            "attributes": self.attributes,
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'Entity':
        return Entity(
            entity_id=data.get("entity_id", ""),
            state=data.get("state", "unknown"),
            attributes=data.get("attributes", {}),
            last_changed=data.get("last_changed", ""),
            last_updated=data.get("last_updated", ""),
        )


@dataclass
class Scene:
    """Represents a Home Assistant scene."""
    entity_id: str
    name: str
    
    @staticmethod
    def from_entity(entity: Entity) -> 'Scene':
        return Scene(
            entity_id=entity.entity_id,
            name=entity.name
        )


@dataclass
class Area:
    """Represents a Home Assistant area/room."""
    area_id: str
    name: str
    entity_ids: List[str] = field(default_factory=list)


class HomeAssistantError(Exception):
    """Base exception for Home Assistant errors."""


class HomeAssistant:
    """
    Home Assistant integration for smart home control.
    
    Connects to a Home Assistant instance and provides methods
    to query and control devices.
    """
    
    def __init__(
        self,
        url: str = None,
        token: str = None,
        verify_ssl: bool = True
    ):
        """
        Initialize Home Assistant connection.
        
        Args:
            url: Home Assistant URL (e.g., http://homeassistant.local:8123)
                 Falls back to HASS_URL environment variable
            token: Long-lived access token
                   Falls back to HASS_TOKEN environment variable
            verify_ssl: Whether to verify SSL certificates
        """
        self.url = url or os.environ.get("HASS_URL", "http://homeassistant.local:8123")
        self.token = token or os.environ.get("HASS_TOKEN", "")
        self.verify_ssl = verify_ssl
        
        # Remove trailing slash
        self.url = self.url.rstrip("/")
        
        # Cache
        self._entities_cache: Dict[str, Entity] = {}
        self._areas_cache: Dict[str, Area] = {}
        self._last_cache_update: float = 0
        self._cache_ttl: float = 30.0  # Seconds
        
        # Connection state
        self._connected = False
    
    @property
    def is_configured(self) -> bool:
        """Check if Home Assistant is configured."""
        return bool(self.url and self.token)
    
    @property
    def api_url(self) -> str:
        """Get the API base URL."""
        return f"{self.url}/api"
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: dict = None,
        timeout: int = 10
    ) -> dict:
        """
        Make an API request to Home Assistant.
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint (e.g., /states)
            data: Request body for POST
            timeout: Request timeout in seconds
            
        Returns:
            Response JSON
            
        Raises:
            HomeAssistantError: On API errors
        """
        if not _REQUESTS_AVAILABLE:
            raise HomeAssistantError("requests library not installed")
        
        if not self.token:
            raise HomeAssistantError("No Home Assistant token configured")
        
        url = f"{self.api_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        
        try:
            if method.upper() == "GET":
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                    verify=self.verify_ssl
                )
            elif method.upper() == "POST":
                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=timeout,
                    verify=self.verify_ssl
                )
            else:
                raise HomeAssistantError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json() if response.text else {}
            
        except requests.exceptions.ConnectionError as e:
            raise HomeAssistantError(f"Cannot connect to Home Assistant: {e}")
        except requests.exceptions.Timeout:
            raise HomeAssistantError("Home Assistant request timed out")
        except requests.exceptions.HTTPError as e:
            raise HomeAssistantError(f"Home Assistant API error: {e}")
        except json.JSONDecodeError:
            raise HomeAssistantError("Invalid JSON response from Home Assistant")
    
    def test_connection(self) -> bool:
        """
        Test the connection to Home Assistant.
        
        Returns:
            True if connection successful
        """
        try:
            result = self._request("GET", "/")
            self._connected = "message" in result
            return self._connected
        except HomeAssistantError as e:
            logger.error(f"Home Assistant connection test failed: {e}")
            self._connected = False
            return False
    
    def get_state(self, entity_id: str) -> Optional[Entity]:
        """
        Get the state of an entity.
        
        Args:
            entity_id: Entity ID (e.g., light.living_room)
            
        Returns:
            Entity object or None if not found
        """
        try:
            data = self._request("GET", f"/states/{entity_id}")
            return Entity.from_dict(data)
        except HomeAssistantError as e:
            logger.error(f"Failed to get state for {entity_id}: {e}")
            return None
    
    def get_all_states(self, force_refresh: bool = False) -> List[Entity]:
        """
        Get states of all entities.
        
        Args:
            force_refresh: Bypass cache
            
        Returns:
            List of Entity objects
        """
        import time
        
        # Check cache
        if not force_refresh:
            if time.time() - self._last_cache_update < self._cache_ttl:
                return list(self._entities_cache.values())
        
        try:
            data = self._request("GET", "/states")
            entities = [Entity.from_dict(e) for e in data]
            
            # Update cache
            self._entities_cache = {e.entity_id: e for e in entities}
            self._last_cache_update = time.time()
            
            return entities
            
        except HomeAssistantError as e:
            logger.error(f"Failed to get all states: {e}")
            return []
    
    def get_entities(
        self,
        domain: str = None,
        area: str = None,
        state: str = None
    ) -> List[Entity]:
        """
        Get entities with optional filtering.
        
        Args:
            domain: Filter by domain (light, switch, sensor, etc.)
            area: Filter by area name
            state: Filter by state (on, off)
            
        Returns:
            Filtered list of entities
        """
        entities = self.get_all_states()
        
        if domain:
            entities = [e for e in entities if e.domain == domain.lower()]
        
        if state:
            state_lower = state.lower()
            entities = [e for e in entities if e.state.lower() == state_lower]
        
        if area:
            # TODO: Implement area filtering when area registry is available
            pass
        
        return entities
    
    def get_scenes(self) -> List[Scene]:
        """Get all available scenes."""
        entities = self.get_entities(domain="scene")
        return [Scene.from_entity(e) for e in entities]
    
    def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str = None,
        **kwargs
    ) -> bool:
        """
        Call a Home Assistant service.
        
        Args:
            domain: Service domain (light, switch, scene, etc.)
            service: Service name (turn_on, turn_off, toggle, etc.)
            entity_id: Target entity ID
            **kwargs: Additional service data
            
        Returns:
            True if service call successful
        """
        data = dict(kwargs)
        if entity_id:
            data["entity_id"] = entity_id
        
        try:
            self._request("POST", f"/services/{domain}/{service}", data)
            logger.info(f"Service called: {domain}.{service} for {entity_id}")
            return True
        except HomeAssistantError as e:
            logger.error(f"Service call failed: {e}")
            return False
    
    def turn_on(self, entity_id: str, **kwargs) -> bool:
        """
        Turn on an entity.
        
        Args:
            entity_id: Entity to turn on
            **kwargs: Additional parameters (brightness, color_temp, etc.)
        """
        domain = entity_id.split(".")[0]
        return self.call_service(domain, "turn_on", entity_id, **kwargs)
    
    def turn_off(self, entity_id: str) -> bool:
        """Turn off an entity."""
        domain = entity_id.split(".")[0]
        return self.call_service(domain, "turn_off", entity_id)
    
    def toggle(self, entity_id: str) -> bool:
        """Toggle an entity."""
        domain = entity_id.split(".")[0]
        return self.call_service(domain, "toggle", entity_id)
    
    def set_light(
        self,
        entity_id: str,
        brightness: int = None,
        color_temp: int = None,
        rgb_color: Tuple[int, int, int] = None,
        transition: float = None
    ) -> bool:
        """
        Set light properties.
        
        Args:
            entity_id: Light entity
            brightness: Brightness 0-255
            color_temp: Color temperature in mireds
            rgb_color: RGB color tuple
            transition: Transition time in seconds
        """
        kwargs = {}
        if brightness is not None:
            kwargs["brightness"] = max(0, min(255, brightness))
        if color_temp is not None:
            kwargs["color_temp"] = color_temp
        if rgb_color is not None:
            kwargs["rgb_color"] = list(rgb_color)
        if transition is not None:
            kwargs["transition"] = transition
        
        return self.turn_on(entity_id, **kwargs)
    
    def activate_scene(self, scene_id: str) -> bool:
        """Activate a scene."""
        if not scene_id.startswith("scene."):
            scene_id = f"scene.{scene_id}"
        return self.call_service("scene", "turn_on", scene_id)
    
    def trigger_automation(self, automation_id: str) -> bool:
        """Trigger an automation."""
        if not automation_id.startswith("automation."):
            automation_id = f"automation.{automation_id}"
        return self.call_service("automation", "trigger", automation_id)
    
    # Natural Language Processing
    
    def execute_command(self, command: str) -> Tuple[bool, str]:
        """
        Execute a natural language command.
        
        Args:
            command: Natural language command (e.g., "Turn on the living room lights")
            
        Returns:
            Tuple of (success, message)
        """
        command_lower = command.lower()
        
        # Parse action
        action = None
        if any(word in command_lower for word in ["turn on", "switch on", "enable", "activate"]):
            action = "turn_on"
        elif any(word in command_lower for word in ["turn off", "switch off", "disable", "deactivate"]):
            action = "turn_off"
        elif "toggle" in command_lower:
            action = "toggle"
        elif any(word in command_lower for word in ["dim", "brightness", "set"]):
            action = "set_brightness"
        
        if not action:
            return False, "Could not understand the command. Try 'turn on/off [device]'."
        
        # Parse target - look for device names
        entities = self.get_all_states()
        matched_entity = None
        best_match_score = 0
        
        for entity in entities:
            # Skip non-controllable entities
            if entity.domain in ("sensor", "binary_sensor", "weather", "person", "zone"):
                continue
            
            # Check for name match
            name_lower = entity.name.lower()
            entity_id_lower = entity.entity_id.lower()
            
            # Direct match
            if name_lower in command_lower or entity_id_lower in command_lower:
                matched_entity = entity
                break
            
            # Word-based matching
            words = name_lower.replace("_", " ").split()
            score = sum(1 for word in words if word in command_lower)
            
            if score > best_match_score:
                best_match_score = score
                matched_entity = entity
        
        if not matched_entity or best_match_score == 0:
            # Try matching all lights in a room
            room_keywords = ["living room", "bedroom", "kitchen", "bathroom", "office", "garage"]
            for room in room_keywords:
                if room in command_lower:
                    # Find all lights in this room
                    room_entities = [
                        e for e in entities
                        if room.replace(" ", "_") in e.entity_id.lower()
                        or room.replace(" ", "") in e.entity_id.lower()
                        or room in e.name.lower()
                    ]
                    if room_entities:
                        # Execute for all matched entities
                        success = True
                        for entity in room_entities:
                            if action == "turn_on":
                                success = success and self.turn_on(entity.entity_id)
                            elif action == "turn_off":
                                success = success and self.turn_off(entity.entity_id)
                            elif action == "toggle":
                                success = success and self.toggle(entity.entity_id)
                        
                        return success, f"{'Turned on' if action == 'turn_on' else 'Turned off' if action == 'turn_off' else 'Toggled'} {len(room_entities)} device(s) in {room}"
            
            return False, "Could not find matching device. Try being more specific."
        
        # Execute action
        if action == "turn_on":
            success = self.turn_on(matched_entity.entity_id)
            return success, f"Turned on {matched_entity.name}" if success else f"Failed to turn on {matched_entity.name}"
        elif action == "turn_off":
            success = self.turn_off(matched_entity.entity_id)
            return success, f"Turned off {matched_entity.name}" if success else f"Failed to turn off {matched_entity.name}"
        elif action == "toggle":
            success = self.toggle(matched_entity.entity_id)
            return success, f"Toggled {matched_entity.name}" if success else f"Failed to toggle {matched_entity.name}"
        elif action == "set_brightness":
            # Parse brightness percentage
            match = re.search(r'(\d+)\s*%?', command_lower)
            if match:
                pct = int(match.group(1))
                brightness = int(pct * 255 / 100)
                success = self.set_light(matched_entity.entity_id, brightness=brightness)
                return success, f"Set {matched_entity.name} to {pct}%" if success else f"Failed to set brightness"
            return False, "Could not parse brightness value"
        
        return False, "Unknown action"
    
    def as_enigma_tools(self) -> List[dict]:
        """
        Get Home Assistant tools for Enigma's tool system.
        
        Returns:
            List of tool definitions compatible with Enigma's tool system
        """
        from enigma_engine.tools.tool_definitions import ToolDefinition, ToolParameter
        
        tools = []
        
        # Get devices tool
        tools.append(ToolDefinition(
            name="home_get_devices",
            description="List smart home devices by type (light, switch, sensor, etc.)",
            parameters=[
                ToolParameter(
                    name="domain",
                    param_type="string",
                    description="Device type: light, switch, sensor, climate, media_player, etc.",
                    required=False
                )
            ],
            category="home",
            function=lambda domain=None: [
                e.to_dict() for e in self.get_entities(domain=domain)
            ]
        ))
        
        # Control device tool
        tools.append(ToolDefinition(
            name="home_control",
            description="Control a smart home device (turn on, turn off, toggle)",
            parameters=[
                ToolParameter(
                    name="entity_id",
                    param_type="string",
                    description="Device entity ID (e.g., light.living_room)",
                    required=True
                ),
                ToolParameter(
                    name="action",
                    param_type="string",
                    description="Action: turn_on, turn_off, toggle",
                    required=True
                ),
                ToolParameter(
                    name="brightness",
                    param_type="integer",
                    description="Brightness 0-255 (for lights)",
                    required=False
                )
            ],
            category="home",
            function=self._tool_control
        ))
        
        # Natural language command tool
        tools.append(ToolDefinition(
            name="home_command",
            description="Execute a natural language smart home command",
            parameters=[
                ToolParameter(
                    name="command",
                    param_type="string",
                    description="Natural language command (e.g., 'Turn off the bedroom lights')",
                    required=True
                )
            ],
            category="home",
            function=lambda command: {"success": (r := self.execute_command(command))[0], "message": r[1]}
        ))
        
        # Get scenes tool
        tools.append(ToolDefinition(
            name="home_get_scenes",
            description="List available smart home scenes",
            parameters=[],
            category="home",
            function=lambda: [{"id": s.entity_id, "name": s.name} for s in self.get_scenes()]
        ))
        
        # Activate scene tool
        tools.append(ToolDefinition(
            name="home_activate_scene",
            description="Activate a smart home scene",
            parameters=[
                ToolParameter(
                    name="scene_id",
                    param_type="string",
                    description="Scene ID (e.g., scene.movie_night)",
                    required=True
                )
            ],
            category="home",
            function=lambda scene_id: {"success": self.activate_scene(scene_id)}
        ))
        
        return tools
    
    def _tool_control(self, entity_id: str, action: str, brightness: int = None) -> dict:
        """Tool wrapper for device control."""
        action_lower = action.lower()
        
        if action_lower == "turn_on":
            kwargs = {}
            if brightness is not None:
                kwargs["brightness"] = brightness
            success = self.turn_on(entity_id, **kwargs)
        elif action_lower == "turn_off":
            success = self.turn_off(entity_id)
        elif action_lower == "toggle":
            success = self.toggle(entity_id)
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
        
        return {"success": success, "entity_id": entity_id, "action": action}


# Singleton instance
_hass_instance: Optional[HomeAssistant] = None


def get_home_assistant(url: str = None, token: str = None) -> HomeAssistant:
    """
    Get the Home Assistant singleton instance.
    
    Args:
        url: Home Assistant URL (optional, uses env var if not provided)
        token: Access token (optional, uses env var if not provided)
        
    Returns:
        HomeAssistant instance
    """
    global _hass_instance
    
    if _hass_instance is None or (url and url != _hass_instance.url):
        _hass_instance = HomeAssistant(url=url, token=token)
    
    return _hass_instance


def register_home_assistant_tools() -> bool:
    """
    Register Home Assistant tools with Enigma's tool system.
    
    Returns:
        True if tools registered successfully
    """
    try:
        from enigma_engine.tools.tool_registry import ToolRegistry
        
        hass = get_home_assistant()
        if not hass.is_configured:
            logger.warning("Home Assistant not configured - tools not registered")
            return False
        
        registry = ToolRegistry.get_instance()
        tools = hass.as_enigma_tools()
        
        for tool in tools:
            registry.register(tool)
        
        logger.info(f"Registered {len(tools)} Home Assistant tools")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register Home Assistant tools: {e}")
        return False


__all__ = [
    "HomeAssistant",
    "HomeAssistantError",
    "Entity",
    "Scene",
    "Area",
    "get_home_assistant",
    "register_home_assistant_tools",
]
