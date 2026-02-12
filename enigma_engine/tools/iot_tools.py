"""
IoT Tools - Home Assistant, Raspberry Pi GPIO, MQTT, Camera.

Tools:
  - homeassistant_control: Control Home Assistant devices
  - homeassistant_status: Get device states from Home Assistant
  - gpio_read: Read GPIO pin state
  - gpio_write: Write to GPIO pin
  - gpio_pwm: PWM output on GPIO pin
  - mqtt_publish: Publish MQTT message
  - mqtt_subscribe: Subscribe to MQTT topic
  - camera_capture: Capture image from camera
  - camera_stream: Get camera stream URL
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .tool_registry import Tool, RichParameter

logger = logging.getLogger(__name__)

# Configuration
IOT_CONFIG_DIR = Path.home() / ".enigma_engine" / "iot"
IOT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_OUTPUT_DIR = Path.home() / ".enigma_engine" / "outputs" / "camera"
CAMERA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# HOME ASSISTANT TOOLS
# ============================================================================

class HomeAssistantConfig:
    """Manages Home Assistant configuration."""
    
    CONFIG_FILE = IOT_CONFIG_DIR / "homeassistant.json"
    
    @classmethod
    def load(cls) -> dict:
        if cls.CONFIG_FILE.exists():
            with open(cls.CONFIG_FILE) as f:
                return json.load(f)
        return {}
    
    @classmethod
    def save(cls, config: dict):
        with open(cls.CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def get_url(cls) -> Optional[str]:
        config = cls.load()
        return config.get('url')
    
    @classmethod
    def get_token(cls) -> Optional[str]:
        config = cls.load()
        return config.get('token')


class HomeAssistantSetupTool(Tool):
    """Configure Home Assistant connection."""
    
    name = "homeassistant_setup"
    description = "Configure the connection to Home Assistant."
    parameters = {
        "url": "Home Assistant URL (e.g., 'http://192.168.1.100:8123')",
        "token": "Long-lived access token from Home Assistant",
    }
    category = "iot"
    rich_parameters = [
        RichParameter(name="url", type="string", description="Home Assistant URL", required=True),
        RichParameter(name="token", type="string", description="Long-lived access token", required=True),
    ]
    examples = ["homeassistant_setup(url='http://192.168.1.100:8123', token='your_token')"]
    
    def execute(self, url: str, token: str, **kwargs) -> dict[str, Any]:
        try:
            # Validate URL format
            if not url.startswith(('http://', 'https://')):
                url = f"http://{url}"
            
            # Remove trailing slash
            url = url.rstrip('/')
            
            # Test connection
            import urllib.request
            
            req = urllib.request.Request(
                f"{url}/api/",
                headers={'Authorization': f'Bearer {token}'}
            )
            
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())
                    message = data.get('message', 'Connected')
            except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
                logger.debug(f"Home Assistant connection test failed: {e}")
                message = "Could not verify connection, but credentials saved"
            
            # Save config
            HomeAssistantConfig.save({'url': url, 'token': token})
            
            return {
                "success": True,
                "message": f"Home Assistant configured: {message}",
                "url": url,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class HomeAssistantControlTool(Tool):
    """Control Home Assistant devices."""
    
    name = "homeassistant_control"
    description = "Control Home Assistant devices (lights, switches, etc.)."
    parameters = {
        "entity_id": "Entity ID (e.g., 'light.living_room', 'switch.fan')",
        "action": "Action: 'turn_on', 'turn_off', 'toggle', or service name",
        "data": "Optional JSON data for the service call (e.g., '{\"brightness\": 255}')",
    }
    category = "iot"
    rich_parameters = [
        RichParameter(name="entity_id", type="string", description="Entity ID", required=True),
        RichParameter(name="action", type="string", description="Action to perform", required=True, enum=["turn_on", "turn_off", "toggle"]),
        RichParameter(name="data", type="string", description="Optional JSON data", required=False),
    ]
    examples = ["homeassistant_control(entity_id='light.living_room', action='turn_on')", "homeassistant_control(entity_id='light.bedroom', action='turn_on', data='{\"brightness\": 128}')"]
    
    def execute(self, entity_id: str, action: str, data: str = None, **kwargs) -> dict[str, Any]:
        try:
            import urllib.request
            
            url = HomeAssistantConfig.get_url()
            token = HomeAssistantConfig.get_token()
            
            if not url or not token:
                return {"success": False, "error": "Home Assistant not configured. Use homeassistant_setup first."}
            
            # Determine domain and service
            domain = entity_id.split('.')[0]
            
            # Map common actions
            service = action
            if action in ['turn_on', 'turn_off', 'toggle']:
                service = action
            
            # Build request
            service_url = f"{url}/api/services/{domain}/{service}"
            
            payload = {"entity_id": entity_id}
            if data:
                extra_data = json.loads(data) if isinstance(data, str) else data
                payload.update(extra_data)
            
            req = urllib.request.Request(
                service_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json',
                },
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
            
            return {
                "success": True,
                "entity_id": entity_id,
                "action": action,
                "result": result,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class HomeAssistantStatusTool(Tool):
    """Get Home Assistant device states."""
    
    name = "homeassistant_status"
    description = "Get the current state of Home Assistant entities."
    parameters = {
        "entity_id": "Entity ID or 'all' for all entities",
        "domain": "Filter by domain (e.g., 'light', 'switch', 'sensor')",
    }
    category = "iot"
    rich_parameters = [
        RichParameter(name="entity_id", type="string", description="Entity ID or 'all'", required=False, default="all"),
        RichParameter(name="domain", type="string", description="Filter by domain", required=False, enum=["light", "switch", "sensor", "binary_sensor", "climate", "cover"]),
    ]
    examples = ["homeassistant_status()", "homeassistant_status(domain='light')"]
    
    def execute(self, entity_id: str = "all", domain: str = None, **kwargs) -> dict[str, Any]:
        try:
            import urllib.request
            
            url = HomeAssistantConfig.get_url()
            token = HomeAssistantConfig.get_token()
            
            if not url or not token:
                return {"success": False, "error": "Home Assistant not configured. Use homeassistant_setup first."}
            
            if entity_id and entity_id != "all":
                api_url = f"{url}/api/states/{entity_id}"
            else:
                api_url = f"{url}/api/states"
            
            req = urllib.request.Request(
                api_url,
                headers={'Authorization': f'Bearer {token}'}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            # Filter by domain if specified
            if domain and isinstance(data, list):
                data = [e for e in data if e['entity_id'].startswith(f"{domain}.")]
            
            # Format response
            if isinstance(data, list):
                entities = []
                for entity in data[:50]:  # Limit results
                    entities.append({
                        "entity_id": entity['entity_id'],
                        "state": entity['state'],
                        "friendly_name": entity.get('attributes', {}).get('friendly_name', ''),
                    })
                return {
                    "success": True,
                    "count": len(entities),
                    "entities": entities,
                }
            else:
                return {
                    "success": True,
                    "entity_id": data['entity_id'],
                    "state": data['state'],
                    "attributes": data.get('attributes', {}),
                }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# GPIO TOOLS (Raspberry Pi)
# ============================================================================

class GPIOReadTool(Tool):
    """Read GPIO pin state."""
    
    name = "gpio_read"
    description = "Read the state of a Raspberry Pi GPIO pin."
    parameters = {
        "pin": "GPIO pin number (BCM numbering)",
        "pull": "Pull-up/down resistor: 'up', 'down', 'none' (default: none)",
    }
    category = "iot"
    rich_parameters = [
        RichParameter(name="pin", type="integer", description="GPIO pin (BCM numbering)", required=True, min_value=0, max_value=27),
        RichParameter(name="pull", type="string", description="Pull resistor", required=False, default="none", enum=["up", "down", "none"]),
    ]
    examples = ["gpio_read(pin=17)", "gpio_read(pin=23, pull='up')"]
    
    def execute(self, pin: int, pull: str = "none", **kwargs) -> dict[str, Any]:
        try:
            pin = int(pin)
            
            # Try RPi.GPIO
            try:
                import RPi.GPIO as GPIO
                
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                
                pull_map = {
                    'up': GPIO.PUD_UP,
                    'down': GPIO.PUD_DOWN,
                    'none': GPIO.PUD_OFF,
                }
                
                GPIO.setup(pin, GPIO.IN, pull_up_down=pull_map.get(pull, GPIO.PUD_OFF))
                value = GPIO.input(pin)
                
                return {
                    "success": True,
                    "pin": pin,
                    "value": value,
                    "state": "HIGH" if value else "LOW",
                }
                
            except ImportError:
                pass  # Intentionally silent
            
            # Try gpiozero
            try:
                from gpiozero import InputDevice
                
                pull_up = True if pull == 'up' else (False if pull == 'down' else None)
                device = InputDevice(pin, pull_up=pull_up)
                value = device.value
                device.close()
                
                return {
                    "success": True,
                    "pin": pin,
                    "value": int(value),
                    "state": "HIGH" if value else "LOW",
                }
                
            except ImportError:
                pass  # Intentionally silent
            
            # Fallback: read from sysfs
            try:
                # Export pin
                with open('/sys/class/gpio/export', 'w') as f:
                    f.write(str(pin))
            except OSError:
                pass  # Pin may already be exported
            
            try:
                # Set direction
                with open(f'/sys/class/gpio/gpio{pin}/direction', 'w') as f:
                    f.write('in')
                
                # Read value
                with open(f'/sys/class/gpio/gpio{pin}/value') as f:
                    value = int(f.read().strip())
                
                return {
                    "success": True,
                    "pin": pin,
                    "value": value,
                    "state": "HIGH" if value else "LOW",
                    "method": "sysfs",
                }
            except OSError:
                pass  # sysfs not available
            
            return {
                "success": False,
                "error": "GPIO not available. Install: pip install RPi.GPIO or gpiozero"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class GPIOWriteTool(Tool):
    """Write to GPIO pin."""
    
    name = "gpio_write"
    description = "Set the state of a Raspberry Pi GPIO pin."
    parameters = {
        "pin": "GPIO pin number (BCM numbering)",
        "value": "Value: 1/HIGH or 0/LOW",
    }
    category = "iot"
    rich_parameters = [
        RichParameter(name="pin", type="integer", description="GPIO pin (BCM numbering)", required=True, min_value=0, max_value=27),
        RichParameter(name="value", type="integer", description="Pin value (0=LOW, 1=HIGH)", required=True, enum=[0, 1]),
    ]
    examples = ["gpio_write(pin=17, value=1)", "gpio_write(pin=23, value=0)"]
    
    def execute(self, pin: int, value: Any, **kwargs) -> dict[str, Any]:
        try:
            pin = int(pin)
            
            # Parse value
            if isinstance(value, str):
                value = 1 if value.upper() in ['1', 'HIGH', 'ON', 'TRUE'] else 0
            else:
                value = 1 if value else 0
            
            # Try RPi.GPIO
            try:
                import RPi.GPIO as GPIO
                
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, value)
                
                return {
                    "success": True,
                    "pin": pin,
                    "value": value,
                    "state": "HIGH" if value else "LOW",
                }
                
            except ImportError:
                pass  # Intentionally silent
            
            # Try gpiozero
            try:
                from gpiozero import OutputDevice
                
                device = OutputDevice(pin)
                if value:
                    device.on()
                else:
                    device.off()
                # Note: Don't close immediately to keep state
                
                return {
                    "success": True,
                    "pin": pin,
                    "value": value,
                    "state": "HIGH" if value else "LOW",
                }
                
            except ImportError:
                pass  # Intentionally silent
            
            return {
                "success": False,
                "error": "GPIO not available. Install: pip install RPi.GPIO or gpiozero"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class GPIOPWMTool(Tool):
    """PWM output on GPIO pin."""
    
    name = "gpio_pwm"
    description = "Output PWM signal on a GPIO pin (for LED dimming, servo control, etc.)."
    parameters = {
        "pin": "GPIO pin number (BCM numbering)",
        "duty_cycle": "Duty cycle 0-100 (percentage)",
        "frequency": "PWM frequency in Hz (default: 1000)",
    }
    category = "iot"
    rich_parameters = [
        RichParameter(name="pin", type="integer", description="GPIO pin (BCM numbering)", required=True, min_value=0, max_value=27),
        RichParameter(name="duty_cycle", type="number", description="Duty cycle (0-100%)", required=True, min_value=0, max_value=100),
        RichParameter(name="frequency", type="integer", description="PWM frequency (Hz)", required=False, default=1000, min_value=1, max_value=50000),
    ]
    examples = ["gpio_pwm(pin=18, duty_cycle=50)", "gpio_pwm(pin=18, duty_cycle=75, frequency=500)"]
    
    # Store active PWM instances
    _pwm_instances = {}
    
    def execute(self, pin: int, duty_cycle: float, frequency: int = 1000, **kwargs) -> dict[str, Any]:
        try:
            pin = int(pin)
            duty_cycle = float(duty_cycle)
            frequency = int(frequency)
            
            # Validate duty cycle
            duty_cycle = max(0, min(100, duty_cycle))
            
            # Try RPi.GPIO
            try:
                import RPi.GPIO as GPIO
                
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                
                # Stop existing PWM on this pin
                if pin in self._pwm_instances:
                    try:
                        self._pwm_instances[pin].stop()
                    except (RuntimeError, AttributeError):
                        pass  # PWM instance may be invalid
                
                GPIO.setup(pin, GPIO.OUT)
                pwm = GPIO.PWM(pin, frequency)
                pwm.start(duty_cycle)
                
                self._pwm_instances[pin] = pwm
                
                return {
                    "success": True,
                    "pin": pin,
                    "duty_cycle": duty_cycle,
                    "frequency": frequency,
                }
                
            except ImportError:
                pass  # Intentionally silent
            
            # Try gpiozero
            try:
                from gpiozero import PWMOutputDevice
                
                device = PWMOutputDevice(pin, frequency=frequency)
                device.value = duty_cycle / 100.0
                
                return {
                    "success": True,
                    "pin": pin,
                    "duty_cycle": duty_cycle,
                    "frequency": frequency,
                }
                
            except ImportError:
                pass  # Intentionally silent
            
            return {
                "success": False,
                "error": "GPIO not available. Install: pip install RPi.GPIO or gpiozero"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# MQTT TOOLS
# ============================================================================

class MQTTPublishTool(Tool):
    """Publish MQTT message."""
    
    name = "mqtt_publish"
    description = "Publish a message to an MQTT topic."
    parameters = {
        "topic": "MQTT topic to publish to",
        "message": "Message to publish (string or JSON)",
        "broker": "MQTT broker address (default: localhost)",
        "port": "MQTT broker port (default: 1883)",
        "retain": "Retain message (default: False)",
    }
    category = "iot"
    rich_parameters = [
        RichParameter(name="topic", type="string", description="MQTT topic", required=True),
        RichParameter(name="message", type="string", description="Message to publish", required=True),
        RichParameter(name="broker", type="string", description="Broker address", required=False, default="localhost"),
        RichParameter(name="port", type="integer", description="Broker port", required=False, default=1883),
        RichParameter(name="retain", type="boolean", description="Retain message", required=False, default=False),
    ]
    examples = ["mqtt_publish(topic='home/temperature', message='22.5')", "mqtt_publish(topic='sensors/status', message='{\"online\": true}', retain=True)"]
    
    def execute(self, topic: str, message: str, broker: str = "localhost",
                port: int = 1883, retain: bool = False, **kwargs) -> dict[str, Any]:
        try:
            # Try paho-mqtt
            try:
                import paho.mqtt.publish as publish
                
                publish.single(
                    topic,
                    payload=message,
                    hostname=broker,
                    port=int(port),
                    retain=retain
                )
                
                return {
                    "success": True,
                    "topic": topic,
                    "message": message,
                    "broker": broker,
                }
                
            except ImportError:
                pass  # Intentionally silent
            
            # Fallback: use mosquitto_pub command
            try:
                import subprocess
                
                cmd = ['mosquitto_pub', '-h', broker, '-p', str(port), '-t', topic, '-m', message]
                if retain:
                    cmd.append('-r')
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "topic": topic,
                        "message": message,
                        "broker": broker,
                        "method": "mosquitto_pub",
                    }
                else:
                    return {"success": False, "error": result.stderr}
                    
            except FileNotFoundError:
                pass  # Intentionally silent
            
            return {
                "success": False,
                "error": "MQTT not available. Install: pip install paho-mqtt or apt install mosquitto-clients"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class MQTTSubscribeTool(Tool):
    """Subscribe to MQTT topic and get messages."""
    
    name = "mqtt_subscribe"
    description = "Subscribe to an MQTT topic and get the latest message."
    parameters = {
        "topic": "MQTT topic to subscribe to",
        "broker": "MQTT broker address (default: localhost)",
        "port": "MQTT broker port (default: 1883)",
        "timeout": "Wait timeout in seconds (default: 5)",
    }
    category = "iot"
    rich_parameters = [
        RichParameter(name="topic", type="string", description="MQTT topic", required=True),
        RichParameter(name="broker", type="string", description="Broker address", required=False, default="localhost"),
        RichParameter(name="port", type="integer", description="Broker port", required=False, default=1883),
        RichParameter(name="timeout", type="integer", description="Timeout in seconds", required=False, default=5, min_value=1, max_value=60),
    ]
    examples = ["mqtt_subscribe(topic='home/temperature')"]
    
    def execute(self, topic: str, broker: str = "localhost",
                port: int = 1883, timeout: int = 5, **kwargs) -> dict[str, Any]:
        try:
            # Try paho-mqtt
            try:
                import paho.mqtt.subscribe as subscribe
                
                msg = subscribe.simple(
                    topic,
                    hostname=broker,
                    port=int(port),
                    msg_count=1
                )
                
                return {
                    "success": True,
                    "topic": msg.topic,
                    "message": msg.payload.decode('utf-8'),
                }
                
            except ImportError:
                pass  # Intentionally silent
            
            # Fallback: use mosquitto_sub command
            try:
                import subprocess
                
                cmd = ['mosquitto_sub', '-h', broker, '-p', str(port), '-t', topic, '-C', '1']
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout))
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "topic": topic,
                        "message": result.stdout.strip(),
                        "method": "mosquitto_sub",
                    }
                else:
                    return {"success": False, "error": result.stderr}
                    
            except subprocess.TimeoutExpired:
                return {"success": False, "error": f"No message received within {timeout}s"}
            except FileNotFoundError:
                pass  # Intentionally silent
            
            return {
                "success": False,
                "error": "MQTT not available. Install: pip install paho-mqtt or apt install mosquitto-clients"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# CAMERA TOOLS
# ============================================================================

class CameraCaptureTool(Tool):
    """Capture image from camera."""
    
    name = "camera_capture"
    description = "Capture an image from a connected camera (webcam, Pi camera, etc.)."
    parameters = {
        "output_path": "Path for the output image (default: auto-generated)",
        "camera_id": "Camera ID for multi-camera systems (default: 0)",
        "resolution": "Resolution as 'WIDTHxHEIGHT' (e.g., '1920x1080')",
    }
    category = "iot"
    rich_parameters = [
        RichParameter(name="output_path", type="string", description="Output image path", required=False),
        RichParameter(name="camera_id", type="integer", description="Camera ID", required=False, default=0, min_value=0, max_value=10),
        RichParameter(name="resolution", type="string", description="Resolution (WIDTHxHEIGHT)", required=False),
    ]
    examples = ["camera_capture()", "camera_capture(resolution='1920x1080')"]
    
    def execute(self, output_path: str = None, camera_id: int = 0,
                resolution: str = None, **kwargs) -> dict[str, Any]:
        try:
            if not output_path:
                output_path = CAMERA_OUTPUT_DIR / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            else:
                output_path = Path(output_path).expanduser().resolve()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try OpenCV
            try:
                import cv2
                
                cap = cv2.VideoCapture(int(camera_id))
                
                if resolution:
                    w, h = map(int, resolution.split('x'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                
                # Wait for camera to warm up
                time.sleep(0.5)
                
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    cv2.imwrite(str(output_path), frame)
                    return {
                        "success": True,
                        "output_path": str(output_path),
                        "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                    }
                else:
                    return {"success": False, "error": "Failed to capture frame"}
                    
            except ImportError:
                pass  # Intentionally silent
            
            # Try picamera for Raspberry Pi
            try:
                from picamera import PiCamera
                
                camera = PiCamera()
                
                if resolution:
                    w, h = map(int, resolution.split('x'))
                    camera.resolution = (w, h)
                
                camera.start_preview()
                time.sleep(2)  # Camera warm-up
                camera.capture(str(output_path))
                camera.stop_preview()
                camera.close()
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "method": "picamera",
                }
                
            except ImportError:
                pass  # Intentionally silent
            
            # Try picamera2 (newer Raspberry Pi)
            try:
                from picamera2 import Picamera2
                
                picam2 = Picamera2()
                config = picam2.create_still_configuration()
                picam2.configure(config)
                picam2.start()
                time.sleep(2)
                picam2.capture_file(str(output_path))
                picam2.stop()
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "method": "picamera2",
                }
                
            except ImportError:
                pass  # Intentionally silent
            
            # Try fswebcam command
            try:
                import subprocess
                
                cmd = ['fswebcam', '-r', resolution or '1280x720', '--no-banner', str(output_path)]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and output_path.exists():
                    return {
                        "success": True,
                        "output_path": str(output_path),
                        "method": "fswebcam",
                    }
                    
            except FileNotFoundError:
                pass  # Intentionally silent
            
            return {
                "success": False,
                "error": "No camera library available. Install: pip install opencv-python or apt install fswebcam"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class CameraListTool(Tool):
    """List available cameras."""
    
    name = "camera_list"
    description = "List available cameras on the system."
    parameters = {}
    category = "iot"
    rich_parameters = []
    examples = ["camera_list()"]
    
    def execute(self, **kwargs) -> dict[str, Any]:
        try:
            cameras = []
            
            # Try OpenCV
            try:
                import cv2

                # Check first 10 camera indices
                for i in range(10):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cameras.append({
                            "id": i,
                            "resolution": f"{w}x{h}",
                        })
                        cap.release()
                    else:
                        break
                        
            except ImportError:
                pass  # Intentionally silent
            
            # Check for Pi camera
            if Path('/dev/video0').exists() and not cameras:
                cameras.append({"id": 0, "type": "video device"})
            
            if Path('/dev/vchiq').exists():
                cameras.append({"id": "picamera", "type": "Raspberry Pi Camera"})
            
            return {
                "success": True,
                "count": len(cameras),
                "cameras": cameras,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class CameraStreamTool(Tool):
    """Start camera stream."""
    
    name = "camera_stream"
    description = "Start an MJPEG stream from the camera (useful for viewing remotely)."
    parameters = {
        "port": "Port for the stream server (default: 8081)",
        "camera_id": "Camera ID (default: 0)",
    }
    category = "iot"
    rich_parameters = [
        RichParameter(name="port", type="integer", description="Stream server port", required=False, default=8081, min_value=1024, max_value=65535),
        RichParameter(name="camera_id", type="integer", description="Camera ID", required=False, default=0),
    ]
    examples = ["camera_stream()", "camera_stream(port=8080)"]
    
    def execute(self, port: int = 8081, camera_id: int = 0, **kwargs) -> dict[str, Any]:
        try:
            import socket

            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                local_ip = s.getsockname()[0]
            finally:
                s.close()
            
            # Try to start motion (if installed)
            try:
                import subprocess

                # Check if motion is installed
                result = subprocess.run(['which', 'motion'], capture_output=True, timeout=5)
                
                if result.returncode == 0:
                    # Start motion with basic config
                    subprocess.Popen(['motion'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    return {
                        "success": True,
                        "stream_url": f"http://{local_ip}:8081/",
                        "method": "motion",
                        "note": "Stream started using motion. Stop with: pkill motion",
                    }
                    
            except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
                logger.debug(f"Camera streaming failed: {e}")
            
            return {
                "success": False,
                "error": "Camera streaming requires 'motion'. Install: apt install motion",
                "local_ip": local_ip,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
