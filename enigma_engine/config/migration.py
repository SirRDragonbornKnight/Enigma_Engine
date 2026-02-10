"""
Configuration Migration Utility

Handles schema changes and migrations between Enigma AI Engine versions.
Automatically updates user configurations when upgrading.

Usage:
    from enigma_engine.config.migration import ConfigMigrator, migrate_config
    
    # Automatic migration
    config = migrate_config(old_config_dict, from_version="1.0.0")
    
    # Or with migrator instance
    migrator = ConfigMigrator()
    config = migrator.migrate(old_config, "1.0.0", "2.0.0")
"""

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Current config version
CURRENT_CONFIG_VERSION = "2.0.0"


@dataclass
class MigrationStep:
    """Represents a single migration step."""
    from_version: str
    to_version: str
    description: str
    migrate_fn: Callable[[dict[str, Any]], dict[str, Any]]


class ConfigMigrator:
    """
    Handles configuration migration between Enigma AI Engine versions.
    
    Migrations are defined as steps that transform config from one version
    to the next. The migrator chains these steps to upgrade configs across
    multiple versions.
    """
    
    def __init__(self):
        self._migrations: list[MigrationStep] = []
        self._register_migrations()
    
    def _register_migrations(self):
        """Register all known migration steps."""
        
        # 1.0.0 -> 1.1.0: Added module system
        self._migrations.append(MigrationStep(
            from_version="1.0.0",
            to_version="1.1.0",
            description="Add module system configuration",
            migrate_fn=self._migrate_1_0_to_1_1,
        ))
        
        # 1.1.0 -> 1.2.0: Renamed settings
        self._migrations.append(MigrationStep(
            from_version="1.1.0",
            to_version="1.2.0",
            description="Rename model_type to model_size",
            migrate_fn=self._migrate_1_1_to_1_2,
        ))
        
        # 1.2.0 -> 1.3.0: Added voice settings
        self._migrations.append(MigrationStep(
            from_version="1.2.0",
            to_version="1.3.0",
            description="Add voice configuration section",
            migrate_fn=self._migrate_1_2_to_1_3,
        ))
        
        # 1.3.0 -> 2.0.0: Major restructure
        self._migrations.append(MigrationStep(
            from_version="1.3.0",
            to_version="2.0.0",
            description="Restructure config for v2 module system",
            migrate_fn=self._migrate_1_3_to_2_0,
        ))
    
    # === Migration Functions ===
    
    def _migrate_1_0_to_1_1(self, config: dict[str, Any]) -> dict[str, Any]:
        """Add module system config section."""
        if "modules" not in config:
            config["modules"] = {
                "auto_load": ["model", "tokenizer", "inference"],
                "disabled": [],
            }
        return config
    
    def _migrate_1_1_to_1_2(self, config: dict[str, Any]) -> dict[str, Any]:
        """Rename model_type to model_size."""
        if "model_type" in config:
            config["model_size"] = config.pop("model_type")
        
        # Standardize size names
        size_map = {
            "tiny": "tiny",
            "base": "small",
            "normal": "small",
            "mid": "medium",
            "big": "large",
        }
        if config.get("model_size") in size_map:
            config["model_size"] = size_map[config["model_size"]]
        
        return config
    
    def _migrate_1_2_to_1_3(self, config: dict[str, Any]) -> dict[str, Any]:
        """Add voice configuration section."""
        if "voice" not in config:
            config["voice"] = {
                "enabled": False,
                "input_device": "default",
                "output_device": "default",
                "tts_engine": "pyttsx3",
                "stt_engine": "vosk",
            }
        
        # Move old voice settings if present
        if "voice_enabled" in config:
            config["voice"]["enabled"] = config.pop("voice_enabled")
        if "tts_provider" in config:
            config["voice"]["tts_engine"] = config.pop("tts_provider")
        
        return config
    
    def _migrate_1_3_to_2_0(self, config: dict[str, Any]) -> dict[str, Any]:
        """Major restructure for v2."""
        # Create new structure
        new_config = {
            "version": "2.0.0",
            "core": {},
            "modules": config.get("modules", {}),
            "gui": {},
            "voice": config.get("voice", {}),
            "paths": {},
        }
        
        # Move model settings to core
        core_keys = ["model_size", "vocab_size", "device", "dtype", "use_flash_attention"]
        for key in core_keys:
            if key in config:
                new_config["core"][key] = config[key]
        
        # Set defaults
        new_config["core"].setdefault("model_size", "small")
        new_config["core"].setdefault("device", "auto")
        
        # Move GUI settings
        gui_keys = ["theme", "font_size", "window_geometry", "gui_mode"]
        for key in gui_keys:
            if key in config:
                new_config["gui"][key] = config[key]
        
        # Move paths
        path_keys = ["models_dir", "data_dir", "output_dir", "logs_dir"]
        for key in path_keys:
            if key in config:
                new_config["paths"][key] = config[key]
        
        # Preserve any unrecognized keys in 'custom' section
        known_keys = set(core_keys + gui_keys + path_keys + 
                        ["modules", "voice", "version"])
        custom = {}
        for key, value in config.items():
            if key not in known_keys:
                custom[key] = value
        if custom:
            new_config["custom"] = custom
        
        return new_config
    
    # === Public API ===
    
    def get_version_path(self, from_ver: str, to_ver: str) -> list[MigrationStep]:
        """
        Find the migration path between two versions.
        
        Returns list of migration steps in order, or empty list if no path.
        """
        # Build version graph
        version_order = ["1.0.0", "1.1.0", "1.2.0", "1.3.0", "2.0.0"]
        
        try:
            from_idx = version_order.index(from_ver)
            to_idx = version_order.index(to_ver)
        except ValueError:
            return []
        
        if from_idx >= to_idx:
            return []  # Already at or past target version
        
        # Collect migrations between versions
        path = []
        for migration in self._migrations:
            if migration.from_version in version_order[from_idx:to_idx]:
                path.append(migration)
        
        return path
    
    def migrate(
        self,
        config: dict[str, Any],
        from_version: str,
        to_version: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Migrate a configuration from one version to another.
        
        Args:
            config: The configuration dictionary to migrate
            from_version: The current version of the config
            to_version: Target version (defaults to current version)
            
        Returns:
            Migrated configuration dictionary
        """
        to_version = to_version or CURRENT_CONFIG_VERSION
        
        path = self.get_version_path(from_version, to_version)
        if not path:
            logger.info(f"No migration needed from {from_version} to {to_version}")
            return config
        
        logger.info(f"Migrating config from {from_version} to {to_version}")
        logger.info(f"Migration path: {len(path)} steps")
        
        for step in path:
            logger.info(f"  {step.from_version} -> {step.to_version}: {step.description}")
            try:
                config = step.migrate_fn(config.copy())
            except Exception as e:
                logger.error(f"Migration step failed: {e}")
                raise RuntimeError(f"Failed to migrate {step.from_version} -> {step.to_version}: {e}")
        
        # Update version
        config["version"] = to_version
        
        return config
    
    def needs_migration(self, config: dict[str, Any]) -> bool:
        """Check if a config needs migration."""
        version = config.get("version", "1.0.0")
        return version != CURRENT_CONFIG_VERSION
    
    def detect_version(self, config: dict[str, Any]) -> str:
        """
        Detect the version of a config based on its structure.
        
        Useful when version field is missing.
        """
        # If version is present, use it
        if "version" in config:
            return config["version"]
        
        # Detect based on structure
        if "core" in config and "paths" in config:
            return "2.0.0"
        if "voice" in config and isinstance(config["voice"], dict):
            return "1.3.0"
        if "model_size" in config:
            return "1.2.0"
        if "modules" in config:
            return "1.1.0"
        
        return "1.0.0"


def backup_config(config_path: Path) -> Path:
    """
    Create a backup of the config file before migration.
    
    Returns path to backup file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.with_suffix(f".backup_{timestamp}.json")
    
    shutil.copy2(config_path, backup_path)
    logger.info(f"Config backed up to {backup_path}")
    
    return backup_path


def migrate_config(
    config: dict[str, Any],
    from_version: Optional[str] = None,
    to_version: Optional[str] = None
) -> dict[str, Any]:
    """
    Convenience function to migrate a configuration.
    
    Args:
        config: Configuration dictionary
        from_version: Source version (auto-detected if not provided)
        to_version: Target version (defaults to current)
        
    Returns:
        Migrated configuration
    """
    migrator = ConfigMigrator()
    
    if from_version is None:
        from_version = migrator.detect_version(config)
    
    return migrator.migrate(config, from_version, to_version)


def migrate_config_file(
    config_path: Path,
    create_backup: bool = True
) -> dict[str, Any]:
    """
    Migrate a configuration file in place.
    
    Args:
        config_path: Path to the JSON config file
        create_backup: Whether to create a backup first
        
    Returns:
        Migrated configuration
    """
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    migrator = ConfigMigrator()
    from_version = migrator.detect_version(config)
    
    if not migrator.needs_migration(config):
        logger.info("Config is already at current version")
        return config
    
    # Backup
    if create_backup:
        backup_config(config_path)
    
    # Migrate
    config = migrator.migrate(config, from_version)
    
    # Save
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Config migrated from {from_version} to {CURRENT_CONFIG_VERSION}")
    
    return config


def get_default_config() -> dict[str, Any]:
    """Get the default configuration for current version."""
    return {
        "version": CURRENT_CONFIG_VERSION,
        "core": {
            "model_size": "small",
            "vocab_size": 8000,
            "device": "auto",
            "dtype": "float32",
        },
        "modules": {
            "auto_load": ["model", "tokenizer", "inference", "memory"],
            "disabled": [],
        },
        "gui": {
            "theme": "dark",
            "font_size": 12,
            "gui_mode": "standard",
        },
        "voice": {
            "enabled": False,
            "input_device": "default",
            "output_device": "default",
            "tts_engine": "pyttsx3",
            "stt_engine": "vosk",
        },
        "paths": {
            "models_dir": "models",
            "data_dir": "data",
            "output_dir": "outputs",
            "logs_dir": "logs",
        },
    }
