"""
================================================================================
Cross-Device Integration - Unified control for Pi + Phone + Gaming PC.
================================================================================

Ties together all cross-device components:
- Device detection and configuration
- Automatic mode switching
- Unified API for all devices
- Easy setup for common scenarios

USAGE:
    from enigma_engine.integration import CrossDeviceSystem
    
    # Quick setup for gaming PC with phone avatar and Pi robot
    system = CrossDeviceSystem()
    system.setup_gaming_pc(
        phone_ip="192.168.1.100",
        pi_ip="192.168.1.101"
    )
    system.start()
    
    # System automatically:
    # - Detects when games are running
    # - Syncs avatar to phone
    # - Controls robot via Pi
    # - Manages resources intelligently
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class SystemRole(Enum):
    """Role this device plays in the system."""
    STANDALONE = auto()      # Single device, all local
    GAMING_HOST = auto()     # Gaming PC - main AI host
    AVATAR_DISPLAY = auto()  # Phone - displays avatar only
    ROBOT_CONTROLLER = auto()# Pi - controls robot hardware
    COMPUTE_NODE = auto()    # Additional AI compute
    SENSOR_NODE = auto()     # Collects sensor data


@dataclass
class DeviceEndpoint:
    """Remote device connection info."""
    name: str
    ip: str
    port: int = 5050
    role: SystemRole = SystemRole.COMPUTE_NODE
    connected: bool = False


class CrossDeviceSystem:
    """
    Unified cross-device AI system.
    
    Integrates:
    - Gaming mode for PC gaming
    - Mobile avatar display
    - Pi robot control
    - Voice pipeline
    - Battery management
    - Performance monitoring
    - Network optimization
    - Device sync
    """
    
    def __init__(self):
        # Components (lazy loaded)
        self._gaming_mode = None
        self._adaptive_engine = None
        self._mobile_avatar = None
        self._pi_robot = None
        self._voice_pipeline = None
        self._battery_manager = None
        self._perf_monitor = None
        self._network_optimizer = None
        self._device_sync = None
        
        # Configuration
        self._role = SystemRole.STANDALONE
        self._devices: dict[str, DeviceEndpoint] = {}
        
        # State
        self._running = False
        self._lock = threading.Lock()
        
        # Event callbacks
        self._callbacks: dict[str, list[Callable]] = {
            "game_started": [],
            "game_ended": [],
            "battery_low": [],
            "device_connected": [],
            "device_disconnected": [],
        }
    
    def setup_standalone(self):
        """Setup for single device operation."""
        self._role = SystemRole.STANDALONE
        
        # Load minimal components
        self._init_performance_monitor()
        self._init_battery_manager()
        
        logger.info("Configured as standalone device")
    
    def setup_gaming_pc(
        self,
        phone_ip: str = None,
        pi_ip: str = None,
        enable_voice: bool = True,
    ):
        """
        Setup as gaming PC host.
        
        Args:
            phone_ip: IP of phone for avatar display
            pi_ip: IP of Raspberry Pi for robot control
            enable_voice: Enable voice input/output
        """
        self._role = SystemRole.GAMING_HOST
        
        # Initialize all components
        self._init_gaming_mode()
        self._init_adaptive_engine()
        self._init_performance_monitor()
        self._init_network_optimizer()
        self._init_device_sync(role="master")
        
        if enable_voice:
            self._init_voice_pipeline()
        
        # Register remote devices
        if phone_ip:
            self._devices["phone"] = DeviceEndpoint(
                name="Phone Avatar",
                ip=phone_ip,
                role=SystemRole.AVATAR_DISPLAY,
            )
            self._init_mobile_avatar_server()
        
        if pi_ip:
            self._devices["pi"] = DeviceEndpoint(
                name="Pi Robot",
                ip=pi_ip,
                role=SystemRole.ROBOT_CONTROLLER,
            )
        
        # Setup gaming mode callbacks
        if self._gaming_mode:
            self._gaming_mode.on_game_started(self._handle_game_started)
            self._gaming_mode.on_game_ended(self._handle_game_ended)
        
        logger.info(f"Configured as gaming host with {len(self._devices)} remote devices")
    
    def setup_phone_avatar(self, host_ip: str, host_port: int = 5050):
        """
        Setup as phone avatar display.
        
        Args:
            host_ip: IP of the gaming PC host
            host_port: Port of the host sync service
        """
        self._role = SystemRole.AVATAR_DISPLAY
        
        # Minimal components
        self._init_battery_manager()
        self._init_mobile_avatar_client(host_ip, host_port)
        self._init_device_sync(role="client", master_url=f"http://{host_ip}:{host_port}")
        
        # Power-aware operation
        if self._battery_manager:
            self._battery_manager.on_state_change(self._handle_power_change)
        
        logger.info(f"Configured as phone avatar, connecting to {host_ip}")
    
    def setup_pi_robot(self, host_ip: str, host_port: int = 5050):
        """
        Setup as Pi robot controller.
        
        Args:
            host_ip: IP of the gaming PC host
            host_port: Port of the host sync service
        """
        self._role = SystemRole.ROBOT_CONTROLLER
        
        # Robot components
        self._init_pi_robot()
        self._init_network_optimizer()
        self._init_device_sync(role="client", master_url=f"http://{host_ip}:{host_port}")
        
        logger.info(f"Configured as Pi robot, connecting to {host_ip}")
    
    def start(self):
        """Start all configured components."""
        self._running = True
        
        # Start components based on role
        if self._gaming_mode:
            self._gaming_mode.start()
        
        if self._adaptive_engine:
            pass  # Engine doesn't have start method
        
        if self._voice_pipeline:
            self._voice_pipeline.start()
        
        if self._battery_manager:
            self._battery_manager.start()
        
        if self._perf_monitor:
            self._perf_monitor.start()
        
        if self._device_sync:
            self._device_sync.start()
        
        if self._mobile_avatar:
            pass  # Avatar is event-driven
        
        if self._pi_robot:
            self._pi_robot.start()
        
        logger.info(f"Cross-device system started as {self._role.name}")
    
    def stop(self):
        """Stop all components."""
        self._running = False
        
        if self._gaming_mode:
            self._gaming_mode.stop()
        
        if self._voice_pipeline:
            self._voice_pipeline.stop()
        
        if self._battery_manager:
            self._battery_manager.stop()
        
        if self._perf_monitor:
            self._perf_monitor.stop()
        
        if self._device_sync:
            self._device_sync.stop()
        
        if self._pi_robot:
            self._pi_robot.stop()
        
        logger.info("Cross-device system stopped")
    
    def get_status(self) -> dict[str, Any]:
        """Get system status."""
        status = {
            "role": self._role.name,
            "running": self._running,
            "devices": {},
            "components": {},
        }
        
        # Device status
        for name, device in self._devices.items():
            status["devices"][name] = {
                "ip": device.ip,
                "role": device.role.name,
                "connected": device.connected,
            }
        
        # Component status
        status["components"]["gaming_mode"] = self._gaming_mode is not None
        status["components"]["voice"] = self._voice_pipeline is not None
        status["components"]["battery"] = self._battery_manager is not None
        status["components"]["performance"] = self._perf_monitor is not None
        status["components"]["sync"] = self._device_sync is not None
        
        # Metrics
        if self._perf_monitor:
            metrics = self._perf_monitor.get_metrics()
            status["metrics"] = {
                "cpu_percent": metrics.cpu_percent,
                "ram_percent": metrics.ram_percent,
                "gpu_percent": metrics.gpu_percent,
            }
        
        if self._battery_manager:
            status["battery"] = {
                "level": self._battery_manager.get_level(),
                "charging": self._battery_manager.is_charging(),
                "state": self._battery_manager.get_state().name,
            }
        
        if self._gaming_mode:
            status["gaming"] = {
                "active": self._gaming_mode.is_gaming_active(),
                "game": self._gaming_mode.get_current_game() or "None",
            }
        
        return status
    
    def on(self, event: str, callback: Callable):
        """Register event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate AI response with automatic optimization."""
        if self._adaptive_engine:
            return self._adaptive_engine.generate(prompt, **kwargs)
        
        # Fallback to pooled engine (efficient reuse)
        try:
            from enigma_engine.core.engine_pool import (
                create_fallback_response,
                get_engine,
                release_engine,
            )
            engine = get_engine()
            if engine is None:
                return create_fallback_response("No engine available")
            try:
                return engine.generate(prompt, **kwargs)
            finally:
                release_engine(engine)
        except ImportError:
            # Double fallback to direct creation
            try:
                from enigma_engine.core import EnigmaEngine
                engine = EnigmaEngine()
                return engine.generate(prompt, **kwargs)
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return f"Error: {e}"
    
    def speak(self, text: str):
        """Speak text via voice pipeline."""
        if self._voice_pipeline:
            self._voice_pipeline.speak(text)
        else:
            logger.info(f"Speech (no voice): {text}")
    
    def update_avatar(self, expression: str = None, animation: str = None, text: str = None):
        """Update avatar state across all displays."""
        if self._device_sync:
            updates = {}
            if expression:
                updates["avatar_expression"] = expression
            if animation:
                updates["avatar_animation"] = animation
            if text:
                updates["avatar_text"] = text
            
            self._device_sync.update(**updates)
    
    def move_robot(self, direction: str, speed: float = 0.5):
        """Send movement command to robot."""
        if self._pi_robot:
            # Local robot control
            if direction == "forward":
                self._pi_robot.move_forward(speed)
            elif direction == "backward":
                self._pi_robot.move_backward(speed)
            elif direction == "left":
                self._pi_robot.turn_left(speed)
            elif direction == "right":
                self._pi_robot.turn_right(speed)
            elif direction == "stop":
                self._pi_robot.stop()
        elif self._device_sync and "pi" in self._devices:
            # Remote robot control
            self._device_sync.send_command("pi", "move", {
                "direction": direction,
                "speed": speed,
            })
    
    # Component initialization
    
    def _init_gaming_mode(self):
        """Initialize gaming mode."""
        try:
            from enigma_engine.core.gaming_mode import GamingMode
            self._gaming_mode = GamingMode()
            logger.debug("Gaming mode initialized")
        except Exception as e:
            logger.warning(f"Gaming mode not available: {e}")
    
    def _init_adaptive_engine(self):
        """Initialize adaptive engine."""
        try:
            from enigma_engine.core.adaptive_engine import AdaptiveEngine
            self._adaptive_engine = AdaptiveEngine()
            logger.debug("Adaptive engine initialized")
        except Exception as e:
            logger.warning(f"Adaptive engine not available: {e}")
    
    def _init_voice_pipeline(self):
        """Initialize voice pipeline."""
        try:
            from enigma_engine.voice.voice_pipeline import VoicePipeline
            self._voice_pipeline = VoicePipeline()
            logger.debug("Voice pipeline initialized")
        except Exception as e:
            logger.warning(f"Voice pipeline not available: {e}")
    
    def _init_battery_manager(self):
        """Initialize battery manager."""
        try:
            from enigma_engine.utils.battery_manager import BatteryManager
            self._battery_manager = BatteryManager()
            logger.debug("Battery manager initialized")
        except Exception as e:
            logger.warning(f"Battery manager not available: {e}")
    
    def _init_performance_monitor(self):
        """Initialize performance monitor."""
        try:
            from enigma_engine.utils.performance_monitor import PerformanceMonitor
            self._perf_monitor = PerformanceMonitor()
            logger.debug("Performance monitor initialized")
        except Exception as e:
            logger.warning(f"Performance monitor not available: {e}")
    
    def _init_network_optimizer(self):
        """Initialize network optimizer."""
        try:
            from enigma_engine.comms.network_optimizer import NetworkOptimizer
            self._network_optimizer = NetworkOptimizer()
            logger.debug("Network optimizer initialized")
        except Exception as e:
            logger.warning(f"Network optimizer not available: {e}")
    
    def _init_device_sync(self, role: str = "client", master_url: str = None):
        """Initialize device sync."""
        try:
            from enigma_engine.comms.device_sync import DeviceSync
            if role == "master":
                self._device_sync = DeviceSync(role="master")
            else:
                self._device_sync = DeviceSync(role="client", master_url=master_url)
            logger.debug(f"Device sync initialized as {role}")
        except Exception as e:
            logger.warning(f"Device sync not available: {e}")
    
    def _init_mobile_avatar_server(self):
        """Initialize mobile avatar server."""
        try:
            pass

            # Server is integrated with device sync
            logger.debug("Mobile avatar server initialized")
        except Exception as e:
            logger.warning(f"Mobile avatar server not available: {e}")
    
    def _init_mobile_avatar_client(self, host_ip: str, host_port: int):
        """Initialize mobile avatar client."""
        try:
            from enigma_engine.avatar.mobile_avatar import MobileAvatar
            self._mobile_avatar = MobileAvatar(
                server_url=f"http://{host_ip}:{host_port}"
            )
            logger.debug("Mobile avatar client initialized")
        except Exception as e:
            logger.warning(f"Mobile avatar client not available: {e}")
    
    def _init_pi_robot(self):
        """Initialize Pi robot controller."""
        try:
            from enigma_engine.tools.pi_robot import PiRobotController
            self._pi_robot = PiRobotController()
            logger.debug("Pi robot controller initialized")
        except Exception as e:
            logger.warning(f"Pi robot not available: {e}")
    
    # Event handlers
    
    def _handle_game_started(self, game_name: str):
        """Handle game started event."""
        logger.info(f"Game started: {game_name}")
        
        # Notify voice to reduce
        if self._voice_pipeline:
            self._voice_pipeline.enable_gaming_mode()
        
        # Update avatar
        self.update_avatar(expression="focused", text=f"Gaming: {game_name}")
        
        # Trigger callbacks
        for callback in self._callbacks["game_started"]:
            try:
                callback(game_name)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _handle_game_ended(self, game_name: str):
        """Handle game ended event."""
        logger.info(f"Game ended: {game_name}")
        
        # Restore voice
        if self._voice_pipeline:
            self._voice_pipeline.disable_gaming_mode()
        
        # Update avatar
        self.update_avatar(expression="happy", text="Ready to help!")
        
        # Trigger callbacks
        for callback in self._callbacks["game_ended"]:
            try:
                callback(game_name)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _handle_power_change(self, state, profile):
        """Handle power state change."""
        logger.info(f"Power state: {state.name}")
        
        # Adjust components based on power
        if self._mobile_avatar:
            # Reduce avatar FPS on low power
            pass
        
        # Trigger low battery callback
        if state.name in ("LOW_POWER", "CRITICAL"):
            for callback in self._callbacks["battery_low"]:
                try:
                    callback(self._battery_manager.get_level())
                except Exception as e:
                    logger.error(f"Callback error: {e}")


# Convenience functions

def quick_setup_gaming_pc(phone_ip: str = None, pi_ip: str = None) -> CrossDeviceSystem:
    """Quick setup for gaming PC."""
    system = CrossDeviceSystem()
    system.setup_gaming_pc(phone_ip=phone_ip, pi_ip=pi_ip)
    system.start()
    return system


def quick_setup_phone(host_ip: str) -> CrossDeviceSystem:
    """Quick setup for phone avatar."""
    system = CrossDeviceSystem()
    system.setup_phone_avatar(host_ip=host_ip)
    system.start()
    return system


def quick_setup_pi(host_ip: str) -> CrossDeviceSystem:
    """Quick setup for Pi robot."""
    system = CrossDeviceSystem()
    system.setup_pi_robot(host_ip=host_ip)
    system.start()
    return system


__all__ = [
    'CrossDeviceSystem',
    'SystemRole',
    'DeviceEndpoint',
    'quick_setup_gaming_pc',
    'quick_setup_phone',
    'quick_setup_pi',
]
