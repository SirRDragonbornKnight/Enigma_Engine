"""
Constants for ForgeAI.

Centralizes magic numbers and values that were previously hardcoded
throughout the codebase.
"""

# =============================================================================
# FILE SIZE LIMITS
# =============================================================================
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB - max file size for reading
MAX_LOG_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB - max log file size

# =============================================================================
# NETWORK TIMEOUTS (in seconds)
# =============================================================================
DEFAULT_WEB_TIMEOUT = 10  # Default timeout for web requests
DEFAULT_API_TIMEOUT = 30  # Default timeout for API calls
HOME_ASSISTANT_TIMEOUT = 10  # Timeout for Home Assistant API

# =============================================================================
# IMAGE GENERATION DEFAULTS
# =============================================================================
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_IMAGE_SIZE = (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)
MAX_IMAGE_DIMENSION = 2048

# =============================================================================
# TEXT GENERATION DEFAULTS
# =============================================================================
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_MAX_GEN_TOKENS = 100
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.1

# =============================================================================
# TRAINING DEFAULTS
# =============================================================================
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_EPOCHS = 30
DEFAULT_WARMUP_STEPS = 100
DEFAULT_GRAD_CLIP = 1.0

# =============================================================================
# SEARCH/PAGINATION LIMITS
# =============================================================================
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 20
DEFAULT_PAGE_SIZE = 50

# =============================================================================
# HARDWARE THRESHOLDS (in MB)
# =============================================================================
MIN_RAM_FOR_SMALL_MODEL = 4096  # 4GB
MIN_RAM_FOR_MEDIUM_MODEL = 8192  # 8GB
MIN_VRAM_FOR_IMAGE_GEN = 4000  # 4GB VRAM for local image generation
MIN_VRAM_FOR_VIDEO_GEN = 8000  # 8GB VRAM for video generation

# =============================================================================
# GUI CONSTANTS
# =============================================================================
DEFAULT_FONT_SIZE = 13
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600

# =============================================================================
# ROBOT/IOT CONSTANTS
# =============================================================================
ARDUINO_BAUD_RATE = 9600
ARDUINO_RESET_DELAY = 2  # seconds to wait for Arduino reset
SERVO_MOVEMENT_DELAY = 0.5  # seconds to wait for servo movement
GPIO_PWM_FREQUENCY = 50  # Hz for servo control

# =============================================================================
# VOICE CONSTANTS
# =============================================================================
DEFAULT_VOICE_SAMPLE_RATE = 16000
VOICE_TRIGGER_TIMEOUT = 1.0  # seconds
VOICE_LISTENING_DURATION = 2.0  # seconds
