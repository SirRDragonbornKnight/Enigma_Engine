"""
Configuration Management for enigma_engine

Centralized configuration with:
- Environment variable support
- Config file loading (YAML, TOML, JSON)
- Type validation
- Default values
- Runtime updates

Usage:
    from enigma_engine.core.config_manager import ConfigManager
    
    config = ConfigManager.load('config.yaml')
    print(config.model.hidden_size)
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import toml
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False


T = TypeVar('T')


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: Optional[int] = None  # For GQA
    intermediate_size: Optional[int] = None  # FFN hidden dim
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    use_flash_attention: bool = False
    use_sliding_window: bool = False
    sliding_window_size: int = 4096
    tie_word_embeddings: bool = True
    
    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 10000
    max_epochs: Optional[int] = None
    
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    
    gradient_clip: float = 1.0
    mixed_precision: str = "fp16"  # fp16, bf16, fp32
    
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10
    
    seed: int = 42
    
    # Advanced
    use_gradient_checkpointing: bool = False
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16


@dataclass
class InferenceConfig:
    """Inference configuration."""
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    
    use_kv_cache: bool = True
    use_speculative_decoding: bool = False
    speculative_tokens: int = 5
    
    batch_size: int = 1
    use_continuous_batching: bool = False


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_minute: int = 60
    tokens_per_day: int = 1000000
    
    # Auth
    require_auth: bool = False
    api_key_file: Optional[str] = None
    
    # CORS
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    
    # Logging
    log_requests: bool = True
    log_file: Optional[str] = None


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    max_memory_gb: Optional[float] = None
    offload_to_cpu: bool = False
    use_paged_attention: bool = False
    kv_cache_dtype: str = "auto"
    
    # For multi-GPU
    device_map: str = "auto"
    max_split_size_mb: int = 512


@dataclass
class DataConfig:
    """Data processing configuration."""
    data_dir: str = "data"
    output_dir: str = "outputs"
    cache_dir: str = ".cache"
    
    max_length: int = 512
    truncation: bool = True
    padding: bool = True
    
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class ForgeConfig:
    """Master configuration for enigma_engine."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Meta
    project_name: str = "enigma_engine"
    experiment_name: str = "default"
    version: str = "1.0.0"


class ConfigManager:
    """
    Configuration management system.
    
    Features:
    - Load from YAML, TOML, or JSON
    - Environment variable overrides
    - Type validation
    - Nested config access
    """
    
    def __init__(self, config: Optional[ForgeConfig] = None):
        self.config = config or ForgeConfig()
        self._env_prefix = "FORGE_"
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ConfigManager':
        """Load configuration from file."""
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}. Using defaults.")
            return cls()
        
        ext = path.suffix.lower()
        
        if ext in ('.yaml', '.yml'):
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
            with open(path) as f:
                data = yaml.safe_load(f)
        
        elif ext == '.toml':
            if not TOML_AVAILABLE:
                raise ImportError("toml not installed. Install with: pip install toml")
            with open(path) as f:
                data = toml.load(f)
        
        elif ext == '.json':
            with open(path) as f:
                data = json.load(f)
        
        else:
            raise ValueError(f"Unsupported config format: {ext}")
        
        # Parse into config objects
        config = cls._parse_config(data)
        manager = cls(config)
        
        # Apply environment overrides
        manager._apply_env_overrides()
        
        return manager
    
    @classmethod
    def _parse_config(cls, data: dict[str, Any]) -> ForgeConfig:
        """Parse dictionary into ForgeConfig."""
        config = ForgeConfig()
        
        # Map sections to config objects
        section_map = {
            'model': (ModelConfig, 'model'),
            'training': (TrainingConfig, 'training'),
            'inference': (InferenceConfig, 'inference'),
            'server': (ServerConfig, 'server'),
            'memory': (MemoryConfig, 'memory'),
            'data': (DataConfig, 'data'),
        }
        
        for section, (config_cls, attr) in section_map.items():
            if section in data:
                section_config = cls._dict_to_dataclass(data[section], config_cls)
                setattr(config, attr, section_config)
        
        # Top-level fields
        for key in ('project_name', 'experiment_name', 'version'):
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    @classmethod
    def _dict_to_dataclass(cls, data: dict[str, Any], cls_type: type[T]) -> T:
        """Convert dictionary to dataclass, handling nested structures."""
        # Get field names
        field_names = {f.name for f in fields(cls_type)}
        
        # Filter to valid fields
        valid_data = {k: v for k, v in data.items() if k in field_names}
        
        return cls_type(**valid_data)
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Environment variables like FORGE_MODEL_HIDDEN_SIZE=1024
        for key, value in os.environ.items():
            if not key.startswith(self._env_prefix):
                continue
            
            # Parse key: FORGE_MODEL_HIDDEN_SIZE -> model.hidden_size
            parts = key[len(self._env_prefix):].lower().split('_')
            
            if len(parts) < 2:
                continue
            
            section = parts[0]
            field_name = '_'.join(parts[1:])
            
            # Get the section config
            section_config = getattr(self.config, section, None)
            if section_config is None:
                continue
            
            # Check if field exists
            if not hasattr(section_config, field_name):
                continue
            
            # Get field type and convert
            current_value = getattr(section_config, field_name)
            converted = self._convert_value(value, type(current_value))
            
            setattr(section_config, field_name, converted)
            logger.debug(f"Applied env override: {section}.{field_name} = {converted}")
    
    def _convert_value(self, value: str, target_type: type) -> Any:
        """Convert string value to target type."""
        if target_type == bool:
            return value.lower() in ('true', '1', 'yes')
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == list:
            return value.split(',')
        else:
            return value
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file."""
        path = Path(path)
        ext = path.suffix.lower()
        
        data = self.to_dict()
        
        if ext in ('.yaml', '.yml'):
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML not installed")
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        
        elif ext == '.toml':
            if not TOML_AVAILABLE:
                raise ImportError("toml not installed")
            with open(path, 'w') as f:
                toml.dump(data, f)
        
        elif ext == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported config format: {ext}")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'project_name': self.config.project_name,
            'experiment_name': self.config.experiment_name,
            'version': self.config.version,
            'model': asdict(self.config.model),
            'training': asdict(self.config.training),
            'inference': asdict(self.config.inference),
            'server': asdict(self.config.server),
            'memory': asdict(self.config.memory),
            'data': asdict(self.config.data),
        }
    
    def update(self, **kwargs):
        """Update configuration values using dot notation."""
        for key, value in kwargs.items():
            parts = key.split('.')
            
            if len(parts) == 1:
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            elif len(parts) == 2:
                section, field = parts
                section_config = getattr(self.config, section, None)
                if section_config and hasattr(section_config, field):
                    setattr(section_config, field, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        parts = key.split('.')
        
        try:
            if len(parts) == 1:
                return getattr(self.config, key, default)
            elif len(parts) == 2:
                section, field = parts
                section_config = getattr(self.config, section)
                return getattr(section_config, field, default)
        except AttributeError:
            return default
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to config sections."""
        if name in ('config', '_env_prefix'):
            return super().__getattribute__(name)
        return getattr(self.config, name)


def load_config(path: Optional[str] = None) -> ConfigManager:
    """
    Load configuration from file or use defaults.
    
    Searches for config files in order:
    1. Explicit path if provided
    2. forge_config.yaml
    3. config.yaml
    4. config.json
    """
    if path:
        return ConfigManager.load(path)
    
    search_paths = [
        'forge_config.yaml',
        'forge_config.yml',
        'config.yaml',
        'config.yml',
        'config.json',
    ]
    
    for p in search_paths:
        if Path(p).exists():
            logger.info(f"Loading config from: {p}")
            return ConfigManager.load(p)
    
    logger.info("No config file found. Using defaults.")
    return ConfigManager()


# Preset configurations
PRESETS = {
    'nano': ModelConfig(
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        vocab_size=32000
    ),
    'small': ModelConfig(
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        vocab_size=50257
    ),
    'medium': ModelConfig(
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        vocab_size=50257
    ),
    'large': ModelConfig(
        hidden_size=2048,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        vocab_size=50257
    ),
    'xl': ModelConfig(
        hidden_size=4096,
        num_layers=48,
        num_heads=32,
        num_kv_heads=8,
        vocab_size=100000
    )
}


def get_preset(name: str) -> ModelConfig:
    """Get a preset model configuration."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
