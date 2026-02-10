"""
GPIO Integration for Raspberry Pi

Control Raspberry Pi GPIO pins from Enigma AI Engine.
Supports digital I/O, PWM, I2C, SPI sensors and actuators.

FILE: enigma_engine/edge/gpio_control.py
TYPE: Edge
MAIN CLASSES: GPIOController, PWMController, I2CDevice, SPIDevice
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPIO libraries
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    GPIO = None

try:
    import smbus2 as smbus
    HAS_I2C = True
except ImportError:
    try:
        import smbus
        HAS_I2C = True
    except ImportError:
        HAS_I2C = False
        smbus = None

try:
    import spidev
    HAS_SPI = True
except ImportError:
    HAS_SPI = False
    spidev = None


class PinMode(Enum):
    """GPIO pin modes."""
    INPUT = auto()
    OUTPUT = auto()
    PWM = auto()
    I2C = auto()
    SPI = auto()


class PullUpDown(Enum):
    """Pull up/down resistor configuration."""
    OFF = auto()
    UP = auto()
    DOWN = auto()


class EdgeDetect(Enum):
    """Edge detection modes."""
    RISING = auto()
    FALLING = auto()
    BOTH = auto()


@dataclass
class PinConfig:
    """GPIO pin configuration."""
    pin: int
    mode: PinMode = PinMode.INPUT
    initial: bool = False
    pull: PullUpDown = PullUpDown.OFF
    pwm_frequency: int = 1000
    bounce_time_ms: int = 200


@dataclass
class GPIOEvent:
    """GPIO event data."""
    pin: int
    state: bool
    timestamp: float
    edge: EdgeDetect


class GPIOController:
    """Main GPIO controller."""
    
    # Raspberry Pi 4 GPIO pins (BCM numbering)
    VALID_PINS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
                  18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    
    def __init__(self, numbering_mode: str = "BCM"):
        """
        Initialize GPIO controller.
        
        Args:
            numbering_mode: "BCM" for Broadcom numbering, "BOARD" for physical pins
        """
        self._pins: dict[int, PinConfig] = {}
        self._callbacks: dict[int, list[Callable[[GPIOEvent], None]]] = {}
        self._pwm_objects: dict[int, Any] = {}
        self._initialized = False
        self._numbering_mode = numbering_mode
        
        if HAS_GPIO:
            self._initialize_gpio()
    
    def _initialize_gpio(self):
        """Initialize GPIO subsystem."""
        try:
            # Suppress warnings about channels already in use
            GPIO.setwarnings(False)
            
            # Set numbering mode
            if self._numbering_mode == "BCM":
                GPIO.setmode(GPIO.BCM)
            else:
                GPIO.setmode(GPIO.BOARD)
            
            self._initialized = True
            logger.info(f"GPIO initialized with {self._numbering_mode} numbering")
        except Exception as e:
            logger.error(f"Failed to initialize GPIO: {e}")
    
    def setup_pin(self, config: PinConfig) -> bool:
        """
        Setup a GPIO pin.
        
        Args:
            config: Pin configuration
            
        Returns:
            True if successful
        """
        if not HAS_GPIO:
            logger.warning("GPIO library not available")
            return False
        
        if config.pin not in self.VALID_PINS:
            logger.error(f"Invalid pin number: {config.pin}")
            return False
        
        try:
            # Set pull up/down
            pull = GPIO.PUD_OFF
            if config.pull == PullUpDown.UP:
                pull = GPIO.PUD_UP
            elif config.pull == PullUpDown.DOWN:
                pull = GPIO.PUD_DOWN
            
            if config.mode == PinMode.INPUT:
                GPIO.setup(config.pin, GPIO.IN, pull_up_down=pull)
            
            elif config.mode == PinMode.OUTPUT:
                GPIO.setup(config.pin, GPIO.OUT, initial=config.initial)
            
            elif config.mode == PinMode.PWM:
                GPIO.setup(config.pin, GPIO.OUT)
                pwm = GPIO.PWM(config.pin, config.pwm_frequency)
                self._pwm_objects[config.pin] = pwm
            
            self._pins[config.pin] = config
            logger.info(f"Pin {config.pin} configured as {config.mode.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup pin {config.pin}: {e}")
            return False
    
    def read(self, pin: int) -> Optional[bool]:
        """
        Read digital input from a pin.
        
        Args:
            pin: Pin number
            
        Returns:
            Pin state (True/False) or None if error
        """
        if not HAS_GPIO:
            return None
        
        try:
            return bool(GPIO.input(pin))
        except Exception as e:
            logger.error(f"Failed to read pin {pin}: {e}")
            return None
    
    def write(self, pin: int, state: bool) -> bool:
        """
        Write digital output to a pin.
        
        Args:
            pin: Pin number
            state: Output state (True=HIGH, False=LOW)
            
        Returns:
            True if successful
        """
        if not HAS_GPIO:
            return False
        
        try:
            GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
            return True
        except Exception as e:
            logger.error(f"Failed to write pin {pin}: {e}")
            return False
    
    def toggle(self, pin: int) -> Optional[bool]:
        """
        Toggle pin state.
        
        Args:
            pin: Pin number
            
        Returns:
            New state or None if error
        """
        current = self.read(pin)
        if current is not None:
            new_state = not current
            if self.write(pin, new_state):
                return new_state
        return None
    
    def pulse(self, pin: int, duration_ms: float = 100, state: bool = True):
        """
        Send a pulse on a pin.
        
        Args:
            pin: Pin number
            duration_ms: Pulse duration in milliseconds
            state: Pulse state (default HIGH)
        """
        self.write(pin, state)
        time.sleep(duration_ms / 1000)
        self.write(pin, not state)
    
    def add_event_callback(
        self, 
        pin: int, 
        edge: EdgeDetect,
        callback: Callable[[GPIOEvent], None],
        bounce_time_ms: int = 200
    ) -> bool:
        """
        Add interrupt callback for pin state change.
        
        Args:
            pin: Pin number
            edge: Edge detection mode
            callback: Function to call on event
            bounce_time_ms: Debounce time
            
        Returns:
            True if successful
        """
        if not HAS_GPIO:
            return False
        
        try:
            # Map edge to GPIO constant
            gpio_edge = {
                EdgeDetect.RISING: GPIO.RISING,
                EdgeDetect.FALLING: GPIO.FALLING,
                EdgeDetect.BOTH: GPIO.BOTH,
            }[edge]
            
            # Store callback
            if pin not in self._callbacks:
                self._callbacks[pin] = []
            self._callbacks[pin].append(callback)
            
            # Wrapper that creates GPIOEvent
            def event_wrapper(channel):
                event = GPIOEvent(
                    pin=channel,
                    state=bool(GPIO.input(channel)),
                    timestamp=time.time(),
                    edge=edge
                )
                for cb in self._callbacks.get(channel, []):
                    try:
                        cb(event)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            
            GPIO.add_event_detect(
                pin,
                gpio_edge,
                callback=event_wrapper,
                bouncetime=bounce_time_ms
            )
            
            logger.info(f"Added callback for pin {pin} on {edge.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add callback: {e}")
            return False
    
    def remove_event_callback(self, pin: int) -> bool:
        """Remove event detection from pin."""
        if not HAS_GPIO:
            return False
        
        try:
            GPIO.remove_event_detect(pin)
            self._callbacks.pop(pin, None)
            return True
        except Exception:
            return False
    
    def wait_for_edge(self, pin: int, edge: EdgeDetect, timeout_ms: int = -1) -> bool:
        """
        Wait for edge on pin (blocking).
        
        Args:
            pin: Pin number
            edge: Edge to wait for
            timeout_ms: Timeout in ms (-1 for infinite)
            
        Returns:
            True if edge detected, False if timeout
        """
        if not HAS_GPIO:
            return False
        
        gpio_edge = {
            EdgeDetect.RISING: GPIO.RISING,
            EdgeDetect.FALLING: GPIO.FALLING,
            EdgeDetect.BOTH: GPIO.BOTH,
        }[edge]
        
        timeout = None if timeout_ms < 0 else timeout_ms
        result = GPIO.wait_for_edge(pin, gpio_edge, timeout=timeout)
        return result is not None
    
    def cleanup(self, pin: int = None):
        """
        Cleanup GPIO resources.
        
        Args:
            pin: Specific pin to cleanup, or None for all
        """
        if not HAS_GPIO:
            return
        
        try:
            if pin is not None:
                # Stop PWM if active
                if pin in self._pwm_objects:
                    self._pwm_objects[pin].stop()
                    del self._pwm_objects[pin]
                
                GPIO.cleanup(pin)
                self._pins.pop(pin, None)
            else:
                # Cleanup all
                for pwm in self._pwm_objects.values():
                    pwm.stop()
                self._pwm_objects.clear()
                
                GPIO.cleanup()
                self._pins.clear()
                self._callbacks.clear()
            
            logger.info(f"GPIO cleanup: {'all' if pin is None else f'pin {pin}'}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


class PWMController:
    """PWM output control."""
    
    def __init__(self, gpio: GPIOController, pin: int, frequency: int = 1000):
        """
        Initialize PWM on a pin.
        
        Args:
            gpio: GPIO controller instance
            pin: Pin number
            frequency: PWM frequency in Hz
        """
        self.gpio = gpio
        self.pin = pin
        self.frequency = frequency
        self._duty_cycle = 0
        self._running = False
        
        # Setup pin for PWM
        config = PinConfig(pin=pin, mode=PinMode.PWM, pwm_frequency=frequency)
        gpio.setup_pin(config)
    
    def start(self, duty_cycle: float = 0):
        """
        Start PWM output.
        
        Args:
            duty_cycle: Initial duty cycle (0-100)
        """
        if not HAS_GPIO:
            return
        
        pwm = self.gpio._pwm_objects.get(self.pin)
        if pwm:
            pwm.start(duty_cycle)
            self._duty_cycle = duty_cycle
            self._running = True
    
    def stop(self):
        """Stop PWM output."""
        if not HAS_GPIO:
            return
        
        pwm = self.gpio._pwm_objects.get(self.pin)
        if pwm:
            pwm.stop()
            self._running = False
    
    def set_duty_cycle(self, duty_cycle: float):
        """
        Set PWM duty cycle.
        
        Args:
            duty_cycle: Duty cycle (0-100)
        """
        if not HAS_GPIO:
            return
        
        duty_cycle = max(0, min(100, duty_cycle))
        pwm = self.gpio._pwm_objects.get(self.pin)
        if pwm:
            pwm.ChangeDutyCycle(duty_cycle)
            self._duty_cycle = duty_cycle
    
    def set_frequency(self, frequency: int):
        """
        Set PWM frequency.
        
        Args:
            frequency: Frequency in Hz
        """
        if not HAS_GPIO:
            return
        
        pwm = self.gpio._pwm_objects.get(self.pin)
        if pwm:
            pwm.ChangeFrequency(frequency)
            self.frequency = frequency
    
    def fade(self, target: float, duration_ms: float, steps: int = 50):
        """
        Fade duty cycle to target value.
        
        Args:
            target: Target duty cycle
            duration_ms: Fade duration
            steps: Number of steps
        """
        start = self._duty_cycle
        step_delay = duration_ms / steps / 1000
        step_size = (target - start) / steps
        
        for i in range(steps):
            self.set_duty_cycle(start + step_size * (i + 1))
            time.sleep(step_delay)


class I2CDevice:
    """I2C device interface."""
    
    def __init__(self, bus: int = 1, address: int = 0x00):
        """
        Initialize I2C device.
        
        Args:
            bus: I2C bus number (usually 1 on Pi)
            address: Device address
        """
        self.bus_num = bus
        self.address = address
        self._bus = None
        
        if HAS_I2C:
            try:
                self._bus = smbus.SMBus(bus)
                logger.info(f"I2C bus {bus} opened, device 0x{address:02X}")
            except Exception as e:
                logger.error(f"Failed to open I2C bus: {e}")
    
    def read_byte(self, register: int) -> Optional[int]:
        """Read single byte from register."""
        if not self._bus:
            return None
        try:
            return self._bus.read_byte_data(self.address, register)
        except Exception as e:
            logger.error(f"I2C read error: {e}")
            return None
    
    def write_byte(self, register: int, value: int) -> bool:
        """Write single byte to register."""
        if not self._bus:
            return False
        try:
            self._bus.write_byte_data(self.address, register, value)
            return True
        except Exception as e:
            logger.error(f"I2C write error: {e}")
            return False
    
    def read_word(self, register: int) -> Optional[int]:
        """Read 16-bit word from register."""
        if not self._bus:
            return None
        try:
            return self._bus.read_word_data(self.address, register)
        except Exception as e:
            logger.error(f"I2C read error: {e}")
            return None
    
    def write_word(self, register: int, value: int) -> bool:
        """Write 16-bit word to register."""
        if not self._bus:
            return False
        try:
            self._bus.write_word_data(self.address, register, value)
            return True
        except Exception as e:
            logger.error(f"I2C write error: {e}")
            return False
    
    def read_block(self, register: int, length: int) -> Optional[list[int]]:
        """Read block of data from register."""
        if not self._bus:
            return None
        try:
            return self._bus.read_i2c_block_data(self.address, register, length)
        except Exception as e:
            logger.error(f"I2C read error: {e}")
            return None
    
    def write_block(self, register: int, data: list[int]) -> bool:
        """Write block of data to register."""
        if not self._bus:
            return False
        try:
            self._bus.write_i2c_block_data(self.address, register, data)
            return True
        except Exception as e:
            logger.error(f"I2C write error: {e}")
            return False
    
    def close(self):
        """Close I2C bus."""
        if self._bus:
            self._bus.close()
            self._bus = None


class SPIDevice:
    """SPI device interface."""
    
    def __init__(self, bus: int = 0, device: int = 0, speed_hz: int = 1000000):
        """
        Initialize SPI device.
        
        Args:
            bus: SPI bus number
            device: SPI device/chip select
            speed_hz: Clock speed in Hz
        """
        self.bus = bus
        self.device = device
        self.speed_hz = speed_hz
        self._spi = None
        
        if HAS_SPI:
            try:
                self._spi = spidev.SpiDev()
                self._spi.open(bus, device)
                self._spi.max_speed_hz = speed_hz
                logger.info(f"SPI bus {bus} device {device} opened at {speed_hz}Hz")
            except Exception as e:
                logger.error(f"Failed to open SPI: {e}")
    
    def transfer(self, data: list[int]) -> Optional[list[int]]:
        """
        Transfer data (simultaneous read/write).
        
        Args:
            data: Data to send
            
        Returns:
            Data received
        """
        if not self._spi:
            return None
        try:
            return self._spi.xfer2(data)
        except Exception as e:
            logger.error(f"SPI transfer error: {e}")
            return None
    
    def read(self, length: int) -> Optional[list[int]]:
        """Read data from SPI."""
        if not self._spi:
            return None
        try:
            return self._spi.readbytes(length)
        except Exception as e:
            logger.error(f"SPI read error: {e}")
            return None
    
    def write(self, data: list[int]) -> bool:
        """Write data to SPI."""
        if not self._spi:
            return False
        try:
            self._spi.writebytes(data)
            return True
        except Exception as e:
            logger.error(f"SPI write error: {e}")
            return False
    
    def close(self):
        """Close SPI device."""
        if self._spi:
            self._spi.close()
            self._spi = None


# Common sensor classes
class DHT22Sensor:
    """DHT22 temperature/humidity sensor."""
    
    def __init__(self, gpio: GPIOController, pin: int):
        self.gpio = gpio
        self.pin = pin
        
        # Try to use Adafruit library
        try:
            import adafruit_dht
            import board
            self._dht = adafruit_dht.DHT22(getattr(board, f"D{pin}"))
        except ImportError:
            self._dht = None
            logger.warning("adafruit_dht not installed")
    
    def read(self) -> tuple[Optional[float], Optional[float]]:
        """
        Read temperature and humidity.
        
        Returns:
            Tuple of (temperature_c, humidity_percent)
        """
        if self._dht is None:
            return None, None
        
        try:
            return self._dht.temperature, self._dht.humidity
        except Exception as e:
            logger.error(f"DHT22 read error: {e}")
            return None, None


class ServoMotor:
    """Servo motor control."""
    
    def __init__(self, gpio: GPIOController, pin: int, min_pulse_ms: float = 0.5, max_pulse_ms: float = 2.5):
        self.gpio = gpio
        self.pin = pin
        self.min_pulse = min_pulse_ms
        self.max_pulse = max_pulse_ms
        
        # Servo typically uses 50Hz PWM
        self.pwm = PWMController(gpio, pin, frequency=50)
        self.pwm.start(0)
    
    def set_angle(self, angle: float):
        """
        Set servo angle.
        
        Args:
            angle: Angle in degrees (0-180)
        """
        angle = max(0, min(180, angle))
        
        # Convert angle to duty cycle
        # 50Hz = 20ms period
        # Typical servo: 0.5ms-2.5ms pulse = 0-180 degrees
        pulse_ms = self.min_pulse + (self.max_pulse - self.min_pulse) * (angle / 180)
        duty_cycle = (pulse_ms / 20) * 100
        
        self.pwm.set_duty_cycle(duty_cycle)
    
    def sweep(self, start: float, end: float, duration_ms: float, steps: int = 50):
        """Sweep servo from start to end angle."""
        step_delay = duration_ms / steps / 1000
        step_size = (end - start) / steps
        
        for i in range(steps + 1):
            self.set_angle(start + step_size * i)
            time.sleep(step_delay)
    
    def stop(self):
        """Stop servo (release hold)."""
        self.pwm.set_duty_cycle(0)


# Global controller instance
_gpio_controller: Optional[GPIOController] = None


def get_gpio() -> GPIOController:
    """Get global GPIO controller."""
    global _gpio_controller
    if _gpio_controller is None:
        _gpio_controller = GPIOController()
    return _gpio_controller


def cleanup_gpio():
    """Cleanup GPIO resources."""
    global _gpio_controller
    if _gpio_controller:
        _gpio_controller.cleanup()
        _gpio_controller = None
