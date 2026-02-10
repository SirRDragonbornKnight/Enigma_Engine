"""
================================================================================
Config Validation - Validate configuration with dataclasses/pydantic.
================================================================================

Provides validated configuration classes:
- Type checking and coercion
- Default values with validation
- Nested configuration support
- Environment variable overrides
- Schema export

USAGE:
    from enigma_engine.config.validation import (
        ModelConfig, TrainingConfig, APIConfig, 
        load_config, validate_config
    )
    
    # Load and validate config from file
    config = load_config("config.yaml", ModelConfig)
    
    # Access validated fields
    print(config.model_size)  # Type-safe access
    
    # Create config programmatically
    training = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        epochs=10
    )
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigError(Exception):
    """Configuration validation error."""


class ModelSize(str, Enum):
    """Valid model sizes."""
    NANO = "nano"
    MICRO = "micro"
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XL = "xl"
    XXL = "xxl"
    OMEGA = "omega"


class DeviceType(str, Enum):
    """Valid device types."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class QuantizationType(str, Enum):
    """Quantization types."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"


def validate_range(value: int | float, min_val: int | float, max_val: int | float, name: str) -> None:
    """Validate a value is within range."""
    if not min_val <= value <= max_val:
        raise ConfigError(f"{name} must be between {min_val} and {max_val}, got {value}")


def validate_positive(value: int | float, name: str) -> None:
    """Validate a value is positive."""
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")


def validate_path_exists(path: str | Path, name: str, must_exist: bool = False) -> Path:
    """Validate a path."""
    p = Path(path)
    if must_exist and not p.exists():
        raise ConfigError(f"{name} path does not exist: {path}")
    return p


@dataclass
class BaseConfig:
    """Base class for validated configs."""
    
    def __post_init__(self):
        """Run validation after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Override to add custom validation."""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        data = self.to_dict()
        # Handle enums
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value
        return json.dumps(data, indent=indent, default=str)
    
    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create from dictionary."""
        # Filter to only valid fields
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    @classmethod
    def from_json(cls: type[T], json_str: str) -> T:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def from_env(cls: type[T], prefix: str = "FORGE") -> T:
        """
        Create from environment variables.
        
        Args:
            prefix: Environment variable prefix (e.g., FORGE_MODEL_SIZE)
        """
        data = {}
        for f in fields(cls):
            env_name = f"{prefix}_{f.name.upper()}"
            env_value = os.environ.get(env_name)
            
            if env_value is not None:
                # Type coercion
                if f.type == bool:
                    data[f.name] = env_value.lower() in ('true', '1', 'yes')
                elif f.type == int:
                    data[f.name] = int(env_value)
                elif f.type == float:
                    data[f.name] = float(env_value)
                else:
                    data[f.name] = env_value
        
        return cls(**data) if data else cls()
    
    def merge(self, other: dict[str, Any] | BaseConfig) -> BaseConfig:
        """Merge with another config or dict (other takes precedence)."""
        current = self.to_dict()
        other_dict = other.to_dict() if isinstance(other, BaseConfig) else other
        current.update({k: v for k, v in other_dict.items() if v is not None})
        return self.__class__.from_dict(current)


@dataclass
class ModelConfig(BaseConfig):
    """Model configuration."""
    
    model_size: str = "small"
    vocab_size: int = 50257
    context_length: int = 2048
    n_layers: int = 12
    n_heads: int = 12
    embed_dim: int = 768
    dropout: float = 0.1
    
    # Attention settings
    use_flash_attention: bool = True
    use_gqa: bool = False  # Grouped Query Attention
    n_kv_heads: int | None = None
    
    # Normalization
    rms_norm_eps: float = 1e-5
    
    # Device settings
    device: str = "auto"
    dtype: str = "float32"
    quantization: str = "none"
    
    def validate(self) -> None:
        validate_positive(self.vocab_size, "vocab_size")
        validate_positive(self.context_length, "context_length")
        validate_positive(self.n_layers, "n_layers")
        validate_positive(self.n_heads, "n_heads")
        validate_positive(self.embed_dim, "embed_dim")
        validate_range(self.dropout, 0.0, 1.0, "dropout")
        
        # Embed dim must be divisible by heads
        if self.embed_dim % self.n_heads != 0:
            raise ConfigError(f"embed_dim ({self.embed_dim}) must be divisible by n_heads ({self.n_heads})")
        
        # Validate model size
        valid_sizes = [s.value for s in ModelSize] + list(ModelSize.__members__.keys())
        if self.model_size.lower() not in [s.lower() for s in valid_sizes]:
            logger.warning(f"Non-standard model size: {self.model_size}")


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration."""
    
    # Core settings
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    max_steps: int | None = None
    
    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Learning rate schedule
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"
    min_lr: float = 1e-6
    
    # Gradient settings
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Data
    train_data_path: str | None = None
    val_data_path: str | None = None
    val_split: float = 0.1
    
    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 5
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_steps: int = 100
    eval_steps: int = 500
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False
    
    # Distributed
    gradient_checkpointing: bool = False
    
    def validate(self) -> None:
        validate_positive(self.learning_rate, "learning_rate")
        validate_positive(self.batch_size, "batch_size")
        validate_positive(self.epochs, "epochs")
        validate_range(self.weight_decay, 0.0, 1.0, "weight_decay")
        validate_range(self.val_split, 0.0, 1.0, "val_split")
        validate_positive(self.warmup_steps, "warmup_steps")
        validate_positive(self.gradient_accumulation_steps, "gradient_accumulation_steps")
        
        if self.min_lr >= self.learning_rate:
            raise ConfigError(f"min_lr ({self.min_lr}) must be less than learning_rate ({self.learning_rate})")


@dataclass
class InferenceConfig(BaseConfig):
    """Inference configuration."""
    
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    do_sample: bool = True
    num_beams: int = 1
    
    # Stop conditions
    stop_sequences: list[str] = field(default_factory=list)
    eos_token_id: int | None = None
    
    # Batching
    batch_size: int = 1
    use_cache: bool = True
    
    # Streaming
    stream: bool = False
    
    def validate(self) -> None:
        validate_positive(self.max_new_tokens, "max_new_tokens")
        validate_range(self.temperature, 0.0, 2.0, "temperature")
        validate_positive(self.top_k, "top_k")
        validate_range(self.top_p, 0.0, 1.0, "top_p")
        validate_range(self.repetition_penalty, 0.1, 10.0, "repetition_penalty")


@dataclass
class APIConfig(BaseConfig):
    """API server configuration."""
    
    host: str = "0.0.0.0"
    port: int = 8080
    
    # Security
    api_key: str | None = None
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Timeouts
    request_timeout: int = 300  # seconds
    
    # Workers
    workers: int = 1
    
    # SSL
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    
    def validate(self) -> None:
        validate_range(self.port, 1, 65535, "port")
        validate_positive(self.rate_limit_requests, "rate_limit_requests")
        validate_positive(self.rate_limit_window, "rate_limit_window")
        validate_positive(self.workers, "workers")


@dataclass
class MemoryConfig(BaseConfig):
    """Memory system configuration."""
    
    # Vector database
    vector_db_type: str = "faiss"  # faiss, simple, chroma
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Conversation storage
    conversation_storage: str = "json"  # json, sqlite
    max_conversations: int = 1000
    max_messages_per_conversation: int = 100
    
    # Entity memory
    entity_memory_enabled: bool = True
    max_entities: int = 10000
    
    # Paths
    memory_dir: str = "memory"
    
    def validate(self) -> None:
        validate_positive(self.embedding_dim, "embedding_dim")
        validate_positive(self.max_conversations, "max_conversations")


@dataclass
class VoiceConfig(BaseConfig):
    """Voice system configuration."""
    
    # TTS
    tts_engine: str = "pyttsx3"  # pyttsx3, elevenlabs, openai
    tts_voice: str | None = None
    tts_rate: int = 150
    
    # STT
    stt_engine: str = "whisper"  # whisper, vosk, google
    stt_model: str = "base"
    
    # Voice activity
    vad_enabled: bool = True
    silence_threshold: float = 0.03
    speech_timeout: float = 1.0
    
    # Audio
    sample_rate: int = 16000
    channels: int = 1
    
    def validate(self) -> None:
        validate_positive(self.tts_rate, "tts_rate")
        validate_positive(self.sample_rate, "sample_rate")
        validate_range(self.silence_threshold, 0.0, 1.0, "silence_threshold")


@dataclass
class ForgeConfig(BaseConfig):
    """Main Enigma AI Engine configuration."""
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    api: APIConfig = field(default_factory=APIConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    
    # Global settings
    debug: bool = False
    log_level: str = "INFO"
    data_dir: str = "data"
    models_dir: str = "models"
    
    def validate(self) -> None:
        if self.log_level.upper() not in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
            raise ConfigError(f"Invalid log_level: {self.log_level}")


def load_config(path: str | Path, config_class: type[T] = ForgeConfig) -> T:
    """
    Load configuration from a file.
    
    Args:
        path: Path to config file (JSON or YAML)
        config_class: Configuration class to use
        
    Returns:
        Validated configuration instance
    """
    path = Path(path)
    
    if not path.exists():
        logger.warning(f"Config file not found: {path}, using defaults")
        return config_class()
    
    with open(path, encoding='utf-8') as f:
        if path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                data = yaml.safe_load(f)
            except ImportError:
                raise ConfigError("PyYAML required for YAML config files")
        else:
            data = json.load(f)
    
    return config_class.from_dict(data)


def save_config(config: BaseConfig, path: str | Path) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = config.to_dict()
    
    with open(path, 'w', encoding='utf-8') as f:
        if path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                yaml.safe_dump(data, f, default_flow_style=False)
            except ImportError:
                raise ConfigError("PyYAML required for YAML config files")
        else:
            json.dump(data, f, indent=2, default=str)


def validate_config(config: dict[str, Any] | BaseConfig, config_class: type[T] = ForgeConfig) -> T:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration dict or instance
        config_class: Target configuration class
        
    Returns:
        Validated configuration
        
    Raises:
        ConfigError: If validation fails
    """
    if isinstance(config, BaseConfig):
        config = config.to_dict()
    
    return config_class.from_dict(config)


def get_config_schema(config_class: type[BaseConfig] = ForgeConfig) -> dict[str, Any]:
    """
    Get JSON schema for a config class.
    
    Args:
        config_class: Configuration class
        
    Returns:
        JSON Schema dictionary
    """
    schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    hints = get_type_hints(config_class)
    
    for f in fields(config_class):
        field_type = hints.get(f.name, Any)
        
        # Map Python types to JSON schema types
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        
        prop = {"type": type_map.get(field_type, "string")}
        
        if f.default is not None and f.default is not field(default=None).default:
            prop["default"] = f.default
        
        schema["properties"][f.name] = prop
        
        # Mark as required if no default
        if f.default is None or f.default is field(default=None).default:
            if f.default_factory is None or f.default_factory is field(default_factory=list).default_factory:
                schema["required"].append(f.name)
    
    return schema
