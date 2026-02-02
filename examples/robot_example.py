"""
Robot Integration Example for ForgeAI

This example shows how to connect ForgeAI to physical robots.
Copy and modify for your specific hardware.

SUPPORTED INTERFACES:
- Serial/USB (Arduino, ESP32, microcontrollers)
- GPIO (Raspberry Pi direct pin control)
- Network/HTTP (WiFi robots, web APIs)
- Simulated (testing without hardware)

USAGE:
    python examples/robot_example.py
    
Or import in your own code:
    from examples.robot_example import create_my_robot
"""

import time
from typing import Dict, Optional

# ForgeAI imports
from forge_ai.tools.robot_tools import (
    RobotInterface,
    RobotController,
    RobotState,
    SerialRobotInterface,
    GPIORobotInterface,
    NetworkRobotInterface,
    SimulatedRobotInterface,
)
from forge_ai.tools.robot_modes import (
    RobotModeController,
    RobotMode,
    CameraConfig,
)


# =============================================================================
# EXAMPLE 1: Simple Arduino Robot (Serial)
# =============================================================================

def create_arduino_robot(port: str = "/dev/ttyUSB0") -> RobotController:
    """
    Connect to an Arduino-based robot via Serial.
    
    Arduino should respond to commands like:
        MOVE shoulder 45 1.0
        GRIP open
        HOME
    
    Args:
        port: Serial port (Linux: /dev/ttyUSB0, Windows: COM3)
    
    Returns:
        Configured RobotController
    """
    controller = RobotController()
    
    # Create serial interface
    arduino = SerialRobotInterface(
        port=port,
        baudrate=115200,
        name="arduino_arm"
    )
    
    # Register with controller
    controller.register("arm", arduino)
    
    return controller


# =============================================================================
# EXAMPLE 2: Raspberry Pi GPIO Robot (Servos on Pi pins)
# =============================================================================

def create_gpio_robot() -> RobotController:
    """
    Control servos directly from Raspberry Pi GPIO pins.
    
    Requires:
        pip install RPi.GPIO
    
    Returns:
        Configured RobotController
    """
    controller = RobotController()
    
    # Create GPIO interface
    gpio_bot = GPIORobotInterface(name="pi_robot")
    
    # Define which GPIO pins control which joints
    # Format: joint_name, GPIO pin number (BCM)
    gpio_bot.add_servo("head_pan", pin=18)
    gpio_bot.add_servo("head_tilt", pin=23)
    gpio_bot.add_servo("jaw", pin=24)
    gpio_bot.add_servo("arm_left", pin=25)
    gpio_bot.add_servo("arm_right", pin=12)
    
    controller.register("gpio", gpio_bot)
    
    return controller


# =============================================================================
# EXAMPLE 3: WiFi/Network Robot (ESP32, HTTP API)
# =============================================================================

def create_network_robot(url: str = "http://192.168.1.100") -> RobotController:
    """
    Connect to a robot with HTTP/REST API.
    
    Robot should have endpoints:
        GET  /status      -> {"ok": true}
        POST /move        -> {"joint": "arm", "angle": 45, "speed": 1.0}
        POST /gripper     -> {"action": "open"}
    
    Args:
        url: Base URL of robot's web server
    
    Returns:
        Configured RobotController
    """
    controller = RobotController()
    
    wifi_bot = NetworkRobotInterface(
        url=url,
        name="wifi_robot"
    )
    
    controller.register("wifi", wifi_bot)
    
    return controller


# =============================================================================
# EXAMPLE 4: Custom Animatronic (like FNAF, Spirit Halloween, etc.)
# =============================================================================

class AnimatronicInterface(RobotInterface):
    """
    Custom interface for animatronic figures.
    
    Uses PCA9685 servo driver for 16-channel PWM control.
    Modify JOINTS dict for your specific animatronic.
    
    Hardware needed:
        - PCA9685 16-channel PWM driver ($5-15)
        - Servos (MG996R, SG90, etc.)
        - 5V power supply (sized for your servos)
        - Raspberry Pi or similar
    
    Wiring:
        PCA9685 VCC -> Pi 3.3V
        PCA9685 GND -> Pi GND  
        PCA9685 SDA -> Pi SDA (GPIO 2)
        PCA9685 SCL -> Pi SCL (GPIO 3)
        PCA9685 V+  -> 5V servo power supply
    """
    
    # ==========================================================
    # MODIFY THIS FOR YOUR ANIMATRONIC
    # Format: "joint_name": (channel, min_angle, max_angle, home_angle)
    # ==========================================================
    JOINTS = {
        # Head joints
        "head_pan": (0, -90, 90, 0),       # Left/right
        "head_tilt": (1, -30, 45, 0),      # Up/down
        "jaw": (2, 0, 35, 0),              # Open/close
        
        # Eyes
        "eye_pan": (3, -30, 30, 0),        # Eyes left/right
        "eye_tilt": (4, -20, 20, 0),       # Eyes up/down
        "eyelid_left": (5, 0, 40, 0),      # Left eyelid
        "eyelid_right": (6, 0, 40, 0),     # Right eyelid
        
        # Body
        "torso": (7, -45, 45, 0),          # Torso rotation
        
        # Left arm
        "shoulder_left": (8, -90, 90, 0),
        "elbow_left": (9, 0, 135, 45),
        "wrist_left": (10, -90, 90, 0),
        
        # Right arm  
        "shoulder_right": (11, -90, 90, 0),
        "elbow_right": (12, 0, 135, 45),
        "wrist_right": (13, -90, 90, 0),
        
        # Extra (fingers, props, etc.)
        "extra_1": (14, 0, 180, 90),
        "extra_2": (15, 0, 180, 90),
    }
    
    def __init__(self, i2c_address: int = 0x40, name: str = "animatronic"):
        super().__init__(name)
        self.i2c_address = i2c_address
        self.pca = None
        self._positions: Dict[str, float] = {}
    
    def connect(self) -> bool:
        """Connect to PCA9685 servo driver."""
        try:
            # Try to import Adafruit library
            from adafruit_pca9685 import PCA9685
            import board
            import busio
            
            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c, address=self.i2c_address)
            self.pca.frequency = 50  # 50Hz standard for servos
            
            # Initialize all positions to home
            for joint, (ch, min_a, max_a, home) in self.JOINTS.items():
                self._positions[joint] = home
            
            self.state = RobotState.CONNECTED
            print(f"[ANIMATRONIC] Connected: {self.name}")
            print(f"[ANIMATRONIC] Joints: {list(self.JOINTS.keys())}")
            return True
            
        except ImportError:
            print("[ANIMATRONIC] ERROR: Install libraries with:")
            print("  pip install adafruit-circuitpython-pca9685")
            self.state = RobotState.ERROR
            return False
        except Exception as e:
            print(f"[ANIMATRONIC] Connection failed: {e}")
            self.state = RobotState.ERROR
            return False
    
    def disconnect(self) -> bool:
        """Disconnect and disable servos."""
        if self.pca:
            # Disable all channels
            for i in range(16):
                self.pca.channels[i].duty_cycle = 0
        self.state = RobotState.DISCONNECTED
        return True
    
    def move_joint(self, joint: str, angle: float, speed: float = 1.0) -> bool:
        """Move a joint to specified angle."""
        if joint not in self.JOINTS:
            print(f"[ANIMATRONIC] Unknown joint: {joint}")
            return False
        
        if not self.pca:
            print("[ANIMATRONIC] Not connected!")
            return False
        
        channel, min_angle, max_angle, _ = self.JOINTS[joint]
        
        # Clamp angle to safe limits
        angle = max(min_angle, min(max_angle, angle))
        
        # Convert angle to PWM duty cycle
        # Standard servo: 500-2500µs pulse width for 0-180°
        # Adjust these values if your servos are different
        pulse_min = 500   # µs at 0°
        pulse_max = 2500  # µs at 180°
        
        # Map angle (-90 to +90 or 0 to 180) to pulse width
        normalized = (angle - min_angle) / (max_angle - min_angle)
        pulse = pulse_min + normalized * (pulse_max - pulse_min)
        
        # Convert to 16-bit duty cycle (0-65535)
        # 50Hz = 20ms period, so pulse/20000 * 65535
        duty = int(pulse / 20000 * 65535)
        
        self.pca.channels[channel].duty_cycle = duty
        self._positions[joint] = angle
        
        return True
    
    def get_joint_position(self, joint: str) -> Optional[float]:
        """Get current position of a joint."""
        return self._positions.get(joint)
    
    def home(self) -> bool:
        """Move all joints to home position."""
        print("[ANIMATRONIC] Homing all joints...")
        for joint, (_, _, _, home_pos) in self.JOINTS.items():
            self.move_joint(joint, home_pos)
            time.sleep(0.05)  # Stagger for smooth motion
        return True
    
    def stop(self) -> bool:
        """Emergency stop - disable all servos immediately."""
        print("[ANIMATRONIC] EMERGENCY STOP!")
        if self.pca:
            for i in range(16):
                self.pca.channels[i].duty_cycle = 0
        return True
    
    # ==========================================================
    # ANIMATION HELPERS
    # ==========================================================
    
    def look_at(self, x: float, y: float):
        """
        Look toward a point (for face tracking).
        
        Args:
            x: Horizontal position (-1 to 1, left to right)
            y: Vertical position (-1 to 1, down to up)
        """
        self.move_joint("head_pan", x * 45)
        self.move_joint("head_tilt", y * 30)
        self.move_joint("eye_pan", x * 20)
        self.move_joint("eye_tilt", y * 15)
    
    def speak_sync(self, intensity: float):
        """
        Sync jaw to speech (call this with audio levels).
        
        Args:
            intensity: Speech loudness 0-1
        """
        jaw_angle = intensity * 30  # 0-30 degrees
        self.move_joint("jaw", jaw_angle)
    
    def blink(self, duration: float = 0.15):
        """Blink both eyes."""
        self.move_joint("eyelid_left", 40)
        self.move_joint("eyelid_right", 40)
        time.sleep(duration)
        self.move_joint("eyelid_left", 0)
        self.move_joint("eyelid_right", 0)
    
    def wave(self, hand: str = "right"):
        """Simple wave animation."""
        arm = f"shoulder_{hand}"
        elbow = f"elbow_{hand}"
        wrist = f"wrist_{hand}"
        
        # Raise arm
        self.move_joint(arm, 45)
        self.move_joint(elbow, 90)
        time.sleep(0.3)
        
        # Wave back and forth
        for _ in range(3):
            self.move_joint(wrist, 30)
            time.sleep(0.2)
            self.move_joint(wrist, -30)
            time.sleep(0.2)
        
        # Return to home
        self.move_joint(arm, 0)
        self.move_joint(elbow, 45)
        self.move_joint(wrist, 0)


def create_animatronic(i2c_address: int = 0x40) -> RobotController:
    """
    Create controller for a custom animatronic.
    
    Args:
        i2c_address: I2C address of PCA9685 (default 0x40)
    
    Returns:
        Configured RobotController
    """
    controller = RobotController()
    
    animatronic = AnimatronicInterface(
        i2c_address=i2c_address,
        name="my_animatronic"
    )
    
    controller.register("animatronic", animatronic)
    
    return controller


# =============================================================================
# EXAMPLE 5: Testing Without Hardware (Simulated)
# =============================================================================

def create_simulated_robot() -> RobotController:
    """
    Create a simulated robot for testing.
    No hardware needed - just logs all commands.
    
    Returns:
        Configured RobotController
    """
    controller = RobotController()
    
    sim = SimulatedRobotInterface(
        name="test_robot",
        joints=["head", "arm_left", "arm_right", "gripper"]
    )
    
    controller.register("sim", sim)
    
    return controller


# =============================================================================
# USAGE WITH FORGEAI
# =============================================================================

def setup_forgeai_robot(controller: RobotController, use_camera: bool = False):
    """
    Connect robot controller to ForgeAI's systems.
    
    Args:
        controller: Your configured RobotController
        use_camera: Enable camera feed for vision
    """
    # Setup mode controller for safety
    mode_ctrl = RobotModeController(controller)
    
    # Start in SAFE mode (limited speed/range)
    mode_ctrl.set_mode(RobotMode.SAFE)
    
    # Optional: Setup camera for vision feedback
    if use_camera:
        mode_ctrl.setup_camera(CameraConfig(
            enabled=True,
            device_id=0,  # Usually 0 for built-in/USB webcam
            resolution=(640, 480),
            fps=30,
        ))
        mode_ctrl.start_camera()
    
    # Register E-STOP callback
    def on_estop(reason):
        print(f"!!! E-STOP TRIGGERED: {reason} !!!")
    
    mode_ctrl._on_estop.append(on_estop)
    
    return mode_ctrl


# =============================================================================
# MAIN - Run this file directly to test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ForgeAI Robot Example")
    print("=" * 60)
    
    # Use simulated robot for testing (no hardware needed)
    print("\n[1] Creating simulated robot...")
    controller = create_simulated_robot()
    
    print("\n[2] Connecting...")
    controller.connect("sim")
    
    print("\n[3] Setting up ForgeAI integration...")
    mode_ctrl = setup_forgeai_robot(controller, use_camera=False)
    
    print("\n[4] Testing movements...")
    robot = controller._robots["sim"]
    
    robot.move_joint("head", 45)
    robot.move_joint("arm_left", 90)
    robot.gripper("close")
    robot.home()
    
    print("\n[5] Testing mode switching...")
    print(f"Current mode: {mode_ctrl.mode.name}")
    print(f"AI can control: {mode_ctrl.can_ai_control}")
    
    mode_ctrl.enter_auto()
    print(f"Switched to: {mode_ctrl.mode.name}")
    print(f"AI can control: {mode_ctrl.can_ai_control}")
    
    print("\n[6] Testing E-STOP...")
    mode_ctrl.emergency_stop("Test stop")
    print(f"E-STOP active: {mode_ctrl.is_estop}")
    
    print("\n[7] Disconnecting...")
    controller.disconnect("sim")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nTo use with real hardware, modify one of:")
    print("  - create_arduino_robot() for Serial/USB")
    print("  - create_gpio_robot() for Raspberry Pi GPIO")
    print("  - create_network_robot() for WiFi/HTTP")
    print("  - create_animatronic() for servo-based animatronics")
