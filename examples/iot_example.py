"""
IoT and Home Automation Example for ForgeAI

This example shows how to connect ForgeAI to smart home devices.
Control lights, switches, sensors, and more!

SUPPORTED PLATFORMS:
- Home Assistant (local or cloud)
- Raspberry Pi GPIO
- MQTT devices
- Generic HTTP devices

USAGE:
    python examples/iot_example.py
    
Or import in your own code:
    from examples.iot_example import create_homeassistant, create_gpio_controller
"""

import json
import time
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

IOT_CONFIG_DIR = Path.home() / ".forge_ai" / "iot"
IOT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


class DeviceState(Enum):
    """Device states."""
    UNKNOWN = "unknown"
    ON = "on"
    OFF = "off"
    UNAVAILABLE = "unavailable"


@dataclass
class Device:
    """Represents a smart device."""
    id: str
    name: str
    type: str  # light, switch, sensor, climate, etc.
    state: DeviceState = DeviceState.UNKNOWN
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


# =============================================================================
# HOME ASSISTANT INTEGRATION
# =============================================================================

class HomeAssistant:
    """
    Control Home Assistant devices.
    
    Setup:
        1. Get a long-lived access token from Home Assistant
           (Profile -> Long-Lived Access Tokens -> Create Token)
        2. Configure with your HA URL and token
    
    Example:
        ha = HomeAssistant("http://192.168.1.100:8123", "your-token")
        ha.turn_on("light.living_room")
        ha.set_brightness("light.living_room", 50)
    """
    
    CONFIG_FILE = IOT_CONFIG_DIR / "homeassistant.json"
    
    def __init__(self, url: str = None, token: str = None):
        """
        Initialize Home Assistant connection.
        
        Args:
            url: Home Assistant URL (e.g., http://192.168.1.100:8123)
            token: Long-lived access token
        """
        # Try to load from config if not provided
        if url is None or token is None:
            config = self._load_config()
            url = url or config.get('url')
            token = token or config.get('token')
        
        self.url = url.rstrip('/') if url else None
        self.token = token
        self._devices: Dict[str, Device] = {}
    
    def _load_config(self) -> Dict:
        """Load saved configuration."""
        if self.CONFIG_FILE.exists():
            with open(self.CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def save_config(self):
        """Save configuration for later use."""
        config = {'url': self.url, 'token': self.token}
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[HA] Config saved to {self.CONFIG_FILE}")
    
    def _request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Make API request to Home Assistant."""
        if not self.url or not self.token:
            return {"error": "Home Assistant not configured"}
        
        url = f"{self.url}/api/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        
        try:
            if data:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers=headers,
                    method=method
                )
            else:
                req = urllib.request.Request(url, headers=headers, method=method)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode())
                
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.reason}"}
        except urllib.error.URLError as e:
            return {"error": f"Connection failed: {e.reason}"}
        except Exception as e:
            return {"error": str(e)}
    
    def test_connection(self) -> bool:
        """Test connection to Home Assistant."""
        result = self._request("")
        if "error" in result:
            print(f"[HA] Connection failed: {result['error']}")
            return False
        print(f"[HA] Connected: {result.get('message', 'OK')}")
        return True
    
    def get_states(self) -> List[Device]:
        """Get all device states."""
        result = self._request("states")
        
        if isinstance(result, list):
            devices = []
            for state in result:
                entity_id = state.get('entity_id', '')
                device = Device(
                    id=entity_id,
                    name=state.get('attributes', {}).get('friendly_name', entity_id),
                    type=entity_id.split('.')[0] if '.' in entity_id else 'unknown',
                    state=DeviceState.ON if state.get('state') == 'on' else DeviceState.OFF,
                    attributes=state.get('attributes', {})
                )
                devices.append(device)
                self._devices[entity_id] = device
            return devices
        
        return []
    
    def get_state(self, entity_id: str) -> Optional[Device]:
        """Get state of specific device."""
        result = self._request(f"states/{entity_id}")
        
        if "error" not in result:
            device = Device(
                id=entity_id,
                name=result.get('attributes', {}).get('friendly_name', entity_id),
                type=entity_id.split('.')[0],
                state=DeviceState.ON if result.get('state') == 'on' else DeviceState.OFF,
                attributes=result.get('attributes', {})
            )
            self._devices[entity_id] = device
            return device
        
        return None
    
    def turn_on(self, entity_id: str, **kwargs) -> bool:
        """
        Turn on a device.
        
        Args:
            entity_id: Device ID (e.g., "light.living_room")
            **kwargs: Additional parameters (brightness, color_temp, etc.)
        """
        domain = entity_id.split('.')[0]
        data = {"entity_id": entity_id, **kwargs}
        result = self._request(f"services/{domain}/turn_on", method="POST", data=data)
        return "error" not in result
    
    def turn_off(self, entity_id: str) -> bool:
        """Turn off a device."""
        domain = entity_id.split('.')[0]
        data = {"entity_id": entity_id}
        result = self._request(f"services/{domain}/turn_off", method="POST", data=data)
        return "error" not in result
    
    def toggle(self, entity_id: str) -> bool:
        """Toggle a device."""
        domain = entity_id.split('.')[0]
        data = {"entity_id": entity_id}
        result = self._request(f"services/{domain}/toggle", method="POST", data=data)
        return "error" not in result
    
    def set_brightness(self, entity_id: str, brightness: int) -> bool:
        """
        Set light brightness.
        
        Args:
            entity_id: Light entity ID
            brightness: 0-255 (or 0-100 will be converted)
        """
        # Convert percentage to 0-255 if needed
        if brightness <= 100:
            brightness = int(brightness * 255 / 100)
        
        return self.turn_on(entity_id, brightness=brightness)
    
    def set_color(self, entity_id: str, r: int, g: int, b: int) -> bool:
        """Set light color (RGB)."""
        return self.turn_on(entity_id, rgb_color=[r, g, b])
    
    def set_temperature(self, entity_id: str, temp: float) -> bool:
        """Set thermostat temperature."""
        data = {"entity_id": entity_id, "temperature": temp}
        result = self._request("services/climate/set_temperature", method="POST", data=data)
        return "error" not in result
    
    def call_service(self, domain: str, service: str, **kwargs) -> bool:
        """
        Call any Home Assistant service.
        
        Args:
            domain: Service domain (light, switch, automation, etc.)
            service: Service name (turn_on, turn_off, trigger, etc.)
            **kwargs: Service data
        """
        result = self._request(f"services/{domain}/{service}", method="POST", data=kwargs)
        return "error" not in result


# =============================================================================
# RASPBERRY PI GPIO
# =============================================================================

class GPIOController:
    """
    Control Raspberry Pi GPIO pins.
    
    Example:
        gpio = GPIOController()
        gpio.setup_output(18)
        gpio.write(18, True)  # Turn on
        gpio.write(18, False) # Turn off
        
        # PWM for dimming
        gpio.setup_pwm(18, frequency=1000)
        gpio.set_duty_cycle(18, 50)  # 50% brightness
    """
    
    def __init__(self):
        self._gpio = None
        self._pwm_channels: Dict[int, Any] = {}
        self._setup_gpio()
    
    def _setup_gpio(self):
        """Initialize GPIO library."""
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self._gpio = GPIO
            print("[GPIO] Initialized (BCM mode)")
        except ImportError:
            print("[GPIO] RPi.GPIO not available (not on Pi?)")
            print("  Install with: pip install RPi.GPIO")
        except Exception as e:
            print(f"[GPIO] Init failed: {e}")
    
    def cleanup(self):
        """Clean up GPIO on exit."""
        if self._gpio:
            self._gpio.cleanup()
    
    def setup_output(self, pin: int, initial: bool = False):
        """Set up pin as output."""
        if self._gpio:
            state = self._gpio.HIGH if initial else self._gpio.LOW
            self._gpio.setup(pin, self._gpio.OUT, initial=state)
            print(f"[GPIO] Pin {pin} set as OUTPUT")
    
    def setup_input(self, pin: int, pull_up: bool = False):
        """Set up pin as input."""
        if self._gpio:
            pud = self._gpio.PUD_UP if pull_up else self._gpio.PUD_DOWN
            self._gpio.setup(pin, self._gpio.IN, pull_up_down=pud)
            print(f"[GPIO] Pin {pin} set as INPUT")
    
    def write(self, pin: int, value: bool):
        """Write to output pin."""
        if self._gpio:
            state = self._gpio.HIGH if value else self._gpio.LOW
            self._gpio.output(pin, state)
    
    def read(self, pin: int) -> Optional[bool]:
        """Read from input pin."""
        if self._gpio:
            return self._gpio.input(pin) == self._gpio.HIGH
        return None
    
    def setup_pwm(self, pin: int, frequency: int = 1000):
        """Set up PWM on a pin."""
        if self._gpio:
            self.setup_output(pin)
            pwm = self._gpio.PWM(pin, frequency)
            pwm.start(0)
            self._pwm_channels[pin] = pwm
            print(f"[GPIO] PWM on pin {pin} at {frequency}Hz")
    
    def set_duty_cycle(self, pin: int, duty: float):
        """Set PWM duty cycle (0-100)."""
        if pin in self._pwm_channels:
            self._pwm_channels[pin].ChangeDutyCycle(duty)
    
    def stop_pwm(self, pin: int):
        """Stop PWM on a pin."""
        if pin in self._pwm_channels:
            self._pwm_channels[pin].stop()
            del self._pwm_channels[pin]


# =============================================================================
# MQTT
# =============================================================================

class MQTTClient:
    """
    MQTT client for IoT devices.
    
    Works with:
        - Tasmota devices
        - Zigbee2MQTT
        - Custom ESP8266/ESP32 devices
        - Any MQTT-enabled device
    
    Example:
        mqtt = MQTTClient("192.168.1.100")
        mqtt.connect()
        mqtt.publish("cmnd/tasmota_plug/POWER", "ON")
        mqtt.subscribe("stat/tasmota_plug/POWER", callback)
    """
    
    def __init__(self, broker: str, port: int = 1883, 
                 username: str = None, password: str = None):
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self._client = None
        self._callbacks: Dict[str, List] = {}
    
    def connect(self) -> bool:
        """Connect to MQTT broker."""
        try:
            import paho.mqtt.client as mqtt
            
            self._client = mqtt.Client()
            
            if self.username:
                self._client.username_pw_set(self.username, self.password)
            
            self._client.on_message = self._on_message
            self._client.connect(self.broker, self.port, 60)
            self._client.loop_start()
            
            print(f"[MQTT] Connected to {self.broker}:{self.port}")
            return True
            
        except ImportError:
            print("[MQTT] paho-mqtt not installed:")
            print("  pip install paho-mqtt")
            return False
        except Exception as e:
            print(f"[MQTT] Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from broker."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
    
    def publish(self, topic: str, payload: str, retain: bool = False) -> bool:
        """
        Publish message to topic.
        
        Args:
            topic: MQTT topic
            payload: Message payload
            retain: Whether to retain message
        """
        if self._client:
            result = self._client.publish(topic, payload, retain=retain)
            return result.rc == 0
        return False
    
    def subscribe(self, topic: str, callback=None):
        """
        Subscribe to topic.
        
        Args:
            topic: MQTT topic (can include wildcards)
            callback: Function to call on message (topic, payload)
        """
        if self._client:
            self._client.subscribe(topic)
            if callback:
                if topic not in self._callbacks:
                    self._callbacks[topic] = []
                self._callbacks[topic].append(callback)
    
    def _on_message(self, client, userdata, msg):
        """Handle incoming messages."""
        topic = msg.topic
        payload = msg.payload.decode()
        
        for pattern, callbacks in self._callbacks.items():
            if self._topic_matches(pattern, topic):
                for cb in callbacks:
                    try:
                        cb(topic, payload)
                    except Exception as e:
                        print(f"[MQTT] Callback error: {e}")
    
    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if topic matches pattern (with wildcards)."""
        if pattern == topic:
            return True
        if '#' in pattern:
            prefix = pattern.replace('#', '')
            return topic.startswith(prefix)
        if '+' in pattern:
            # Simple single-level wildcard matching
            parts_p = pattern.split('/')
            parts_t = topic.split('/')
            if len(parts_p) != len(parts_t):
                return False
            return all(p == '+' or p == t for p, t in zip(parts_p, parts_t))
        return False


# =============================================================================
# GENERIC HTTP DEVICE
# =============================================================================

class HTTPDevice:
    """
    Control any device with HTTP API.
    
    Works with:
        - Tasmota (HTTP commands)
        - Shelly devices
        - ESP8266/ESP32 web servers
        - Any REST API device
    
    Example:
        device = HTTPDevice("http://192.168.1.50")
        device.get("/status")
        device.post("/control", {"power": "on"})
    """
    
    def __init__(self, base_url: str, auth: tuple = None):
        """
        Args:
            base_url: Device base URL
            auth: Optional (username, password) tuple
        """
        self.base_url = base_url.rstrip('/')
        self.auth = auth
    
    def get(self, endpoint: str = "") -> Dict[str, Any]:
        """GET request to device."""
        return self._request("GET", endpoint)
    
    def post(self, endpoint: str, data: Dict) -> Dict[str, Any]:
        """POST request to device."""
        return self._request("POST", endpoint, data)
    
    def _request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make HTTP request."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            headers = {"Content-Type": "application/json"}
            
            if data:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers=headers,
                    method=method
                )
            else:
                req = urllib.request.Request(url, method=method)
            
            # Add basic auth if configured
            if self.auth:
                import base64
                credentials = base64.b64encode(f"{self.auth[0]}:{self.auth[1]}".encode()).decode()
                req.add_header("Authorization", f"Basic {credentials}")
            
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read().decode()
                try:
                    return {"success": True, "data": json.loads(content)}
                except json.JSONDecodeError:
                    return {"success": True, "data": content}
                    
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_homeassistant(url: str = None, token: str = None) -> HomeAssistant:
    """Create Home Assistant controller."""
    return HomeAssistant(url, token)


def create_gpio_controller() -> GPIOController:
    """Create GPIO controller (Raspberry Pi only)."""
    return GPIOController()


def create_mqtt_client(broker: str, port: int = 1883) -> MQTTClient:
    """Create MQTT client."""
    return MQTTClient(broker, port)


def create_http_device(url: str) -> HTTPDevice:
    """Create generic HTTP device controller."""
    return HTTPDevice(url)


# =============================================================================
# SIMULATED DEVICES (for testing)
# =============================================================================

class SimulatedDevice:
    """Simulated smart device for testing."""
    
    def __init__(self, name: str, device_type: str = "light"):
        self.name = name
        self.type = device_type
        self.state = False
        self.brightness = 100
        self.color = (255, 255, 255)
    
    def turn_on(self):
        self.state = True
        print(f"[SIM] {self.name} turned ON")
    
    def turn_off(self):
        self.state = False
        print(f"[SIM] {self.name} turned OFF")
    
    def set_brightness(self, value: int):
        self.brightness = value
        print(f"[SIM] {self.name} brightness set to {value}%")
    
    def get_state(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "state": "on" if self.state else "off",
            "brightness": self.brightness,
            "color": self.color,
        }


# =============================================================================
# MAIN - Run this file directly to test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ForgeAI IoT Example")
    print("=" * 60)
    
    # Test simulated device
    print("\n[1] Testing simulated device...")
    light = SimulatedDevice("Living Room Light", "light")
    light.turn_on()
    light.set_brightness(75)
    light.turn_off()
    print(f"State: {light.get_state()}")
    
    # Test Home Assistant (if configured)
    print("\n[2] Testing Home Assistant...")
    ha = HomeAssistant()
    
    if ha.url and ha.token:
        if ha.test_connection():
            print("Getting devices...")
            devices = ha.get_states()
            print(f"Found {len(devices)} devices")
            
            # Show some devices
            for d in devices[:5]:
                print(f"  - {d.name} ({d.type}): {d.state.value}")
    else:
        print("Home Assistant not configured.")
        print("Configure with:")
        print("  ha = HomeAssistant('http://YOUR_HA_IP:8123', 'YOUR_TOKEN')")
        print("  ha.save_config()")
    
    # Test GPIO (only works on Raspberry Pi)
    print("\n[3] Testing GPIO...")
    gpio = GPIOController()
    
    if gpio._gpio:
        print("GPIO available!")
        # Example: Blink an LED on pin 18
        # gpio.setup_output(18)
        # gpio.write(18, True)
        # time.sleep(0.5)
        # gpio.write(18, False)
    else:
        print("GPIO not available (not on Raspberry Pi)")
    
    # Test HTTP device
    print("\n[4] Testing HTTP device...")
    print("Example for Tasmota device:")
    print("  device = HTTPDevice('http://192.168.1.50')")
    print("  device.get('/cm?cmnd=Power%20On')")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nSupported integrations:")
    print("  - Home Assistant (needs URL + token)")
    print("  - GPIO (Raspberry Pi only)")
    print("  - MQTT (pip install paho-mqtt)")
    print("  - HTTP devices (any REST API)")
