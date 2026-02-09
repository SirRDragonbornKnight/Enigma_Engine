"""
Avatar Performance Modes for Enigma AI Engine

Provides configurable performance levels for avatar system.
Lower modes reduce CPU/GPU usage for lighter systems.

Usage:
    from enigma_engine.avatar.performance import (
        PerformanceMode, AvatarPerformanceConfig,
        get_performance_config, set_performance_mode
    )
    
    # Set lightweight mode for low-end system
    set_performance_mode(PerformanceMode.MINIMAL)
    
    # Or use preset
    config = get_performance_config()
    config.apply_preset("raspberry_pi")
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PerformanceMode(Enum):
    """Avatar performance presets."""
    FULL = "full"           # All features, full quality
    HIGH = "high"           # Most features, slight optimizations  
    BALANCED = "balanced"   # Good balance of quality and performance
    LIGHT = "light"         # Reduced features for laptops
    MINIMAL = "minimal"     # Bare minimum for very low-end systems
    STATIC = "static"       # No animation at all, just static image


@dataclass
class AvatarPerformanceConfig:
    """
    Detailed performance configuration for avatar system.
    
    All values can be tuned individually or use presets.
    """
    # Core Settings
    mode: PerformanceMode = PerformanceMode.BALANCED
    
    # Animation Settings
    animation_enabled: bool = True
    animation_fps: int = 30          # Frame rate for animations (5-60)
    idle_animation: bool = True      # Breathing, blinking, micro-movements
    smooth_movement: bool = True     # Smooth bone transitions
    physics_enabled: bool = True     # Hair/cloth physics
    
    # Bone Control Settings  
    bone_control_enabled: bool = True
    max_active_bones: int = 50       # Limit simultaneous bone updates
    bone_update_rate: float = 0.033  # Seconds between bone updates (30fps)
    interpolation_steps: int = 5     # Steps for smooth movement
    
    # Visual Quality
    texture_quality: str = "high"    # low/medium/high/ultra
    shadow_enabled: bool = True
    reflection_enabled: bool = True
    antialiasing: bool = True
    post_processing: bool = True
    
    # Expression/Emotion
    expression_enabled: bool = True
    lip_sync_enabled: bool = True
    lip_sync_quality: str = "high"   # low/medium/high
    emotion_detection: bool = True
    blendshape_smoothing: bool = True
    
    # Background Processing
    background_processing: bool = True  # Process when minimized
    gpu_acceleration: bool = True
    multithreading: bool = True
    max_threads: int = 4
    
    # Memory Limits
    texture_cache_mb: int = 256
    animation_cache_mb: int = 64
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "animation_enabled": self.animation_enabled,
            "animation_fps": self.animation_fps,
            "idle_animation": self.idle_animation,
            "smooth_movement": self.smooth_movement,
            "physics_enabled": self.physics_enabled,
            "bone_control_enabled": self.bone_control_enabled,
            "max_active_bones": self.max_active_bones,
            "bone_update_rate": self.bone_update_rate,
            "interpolation_steps": self.interpolation_steps,
            "texture_quality": self.texture_quality,
            "shadow_enabled": self.shadow_enabled,
            "reflection_enabled": self.reflection_enabled,
            "antialiasing": self.antialiasing,
            "post_processing": self.post_processing,
            "expression_enabled": self.expression_enabled,
            "lip_sync_enabled": self.lip_sync_enabled,
            "lip_sync_quality": self.lip_sync_quality,
            "emotion_detection": self.emotion_detection,
            "blendshape_smoothing": self.blendshape_smoothing,
            "background_processing": self.background_processing,
            "gpu_acceleration": self.gpu_acceleration,
            "multithreading": self.multithreading,
            "max_threads": self.max_threads,
            "texture_cache_mb": self.texture_cache_mb,
            "animation_cache_mb": self.animation_cache_mb,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AvatarPerformanceConfig':
        """Create from dictionary."""
        mode_str = data.pop("mode", "balanced")
        mode = PerformanceMode(mode_str)
        return cls(mode=mode, **data)
    
    def apply_preset(self, preset: str):
        """
        Apply a preset configuration.
        
        Presets:
            full - Everything enabled, max quality
            high - High quality with minor optimizations
            balanced - Good balance (default)
            light - For laptops/tablets
            minimal - For Raspberry Pi / very low-end
            static - No animations (lowest CPU)
            raspberry_pi - Optimized for RPi
            laptop_battery - Battery saving mode
        """
        presets = {
            "full": {
                "mode": PerformanceMode.FULL,
                "animation_fps": 60,
                "max_active_bones": 100,
                "interpolation_steps": 10,
                "texture_quality": "ultra",
                "lip_sync_quality": "high",
                "texture_cache_mb": 512,
                "animation_cache_mb": 128,
            },
            "high": {
                "mode": PerformanceMode.HIGH,
                "animation_fps": 45,
                "max_active_bones": 75,
                "interpolation_steps": 8,
                "texture_quality": "high",
            },
            "balanced": {
                "mode": PerformanceMode.BALANCED,
                "animation_fps": 30,
                "max_active_bones": 50,
                "interpolation_steps": 5,
                "texture_quality": "medium",
            },
            "light": {
                "mode": PerformanceMode.LIGHT,
                "animation_fps": 20,
                "idle_animation": False,
                "physics_enabled": False,
                "max_active_bones": 25,
                "interpolation_steps": 3,
                "texture_quality": "low",
                "shadow_enabled": False,
                "reflection_enabled": False,
                "antialiasing": False,
                "post_processing": False,
                "lip_sync_quality": "low",
                "emotion_detection": False,
                "blendshape_smoothing": False,
                "texture_cache_mb": 128,
                "animation_cache_mb": 32,
            },
            "minimal": {
                "mode": PerformanceMode.MINIMAL,
                "animation_fps": 10,
                "idle_animation": False,
                "smooth_movement": False,
                "physics_enabled": False,
                "max_active_bones": 10,
                "bone_update_rate": 0.1,
                "interpolation_steps": 1,
                "texture_quality": "low",
                "shadow_enabled": False,
                "reflection_enabled": False,
                "antialiasing": False,
                "post_processing": False,
                "lip_sync_enabled": False,
                "lip_sync_quality": "low",
                "emotion_detection": False,
                "blendshape_smoothing": False,
                "background_processing": False,
                "gpu_acceleration": False,
                "multithreading": False,
                "max_threads": 1,
                "texture_cache_mb": 64,
                "animation_cache_mb": 16,
            },
            "static": {
                "mode": PerformanceMode.STATIC,
                "animation_enabled": False,
                "animation_fps": 1,
                "idle_animation": False,
                "smooth_movement": False,
                "physics_enabled": False,
                "bone_control_enabled": False,
                "max_active_bones": 0,
                "expression_enabled": False,
                "lip_sync_enabled": False,
                "emotion_detection": False,
                "background_processing": False,
                "gpu_acceleration": False,
                "multithreading": False,
                "texture_cache_mb": 32,
                "animation_cache_mb": 0,
            },
            "raspberry_pi": {
                "mode": PerformanceMode.MINIMAL,
                "animation_fps": 15,
                "idle_animation": False,
                "physics_enabled": False,
                "max_active_bones": 15,
                "bone_update_rate": 0.067,  # 15fps
                "interpolation_steps": 2,
                "texture_quality": "low",
                "shadow_enabled": False,
                "reflection_enabled": False,
                "antialiasing": False,
                "post_processing": False,
                "lip_sync_quality": "low",
                "emotion_detection": False,
                "gpu_acceleration": False,
                "multithreading": True,
                "max_threads": 2,
                "texture_cache_mb": 64,
                "animation_cache_mb": 16,
            },
            "laptop_battery": {
                "mode": PerformanceMode.LIGHT,
                "animation_fps": 15,
                "idle_animation": False,
                "physics_enabled": False,
                "max_active_bones": 20,
                "interpolation_steps": 2,
                "texture_quality": "low",
                "shadow_enabled": False,
                "reflection_enabled": False,
                "post_processing": False,
                "lip_sync_quality": "low",
                "background_processing": False,
                "texture_cache_mb": 128,
            },
        }
        
        if preset not in presets:
            logger.warning(f"Unknown preset: {preset}, using balanced")
            preset = "balanced"
        
        settings = presets[preset]
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logger.info(f"Applied avatar performance preset: {preset}")
    
    def estimate_resource_usage(self) -> Dict[str, str]:
        """
        Estimate resource usage based on current settings.
        
        Returns:
            Dictionary with CPU, GPU, Memory estimates
        """
        # Base scores
        cpu_score = 0
        gpu_score = 0
        memory_score = self.texture_cache_mb + self.animation_cache_mb
        
        # Animation impact
        if self.animation_enabled:
            cpu_score += self.animation_fps * 0.5
            if self.smooth_movement:
                cpu_score += 10
            if self.physics_enabled:
                cpu_score += 20
                gpu_score += 15
        
        # Bone control impact
        if self.bone_control_enabled:
            cpu_score += self.max_active_bones * 0.3
            cpu_score += self.interpolation_steps * 2
        
        # Visual quality impact
        quality_mult = {"low": 0.5, "medium": 1.0, "high": 1.5, "ultra": 2.0}
        gpu_score += 20 * quality_mult.get(self.texture_quality, 1.0)
        
        if self.shadow_enabled:
            gpu_score += 15
        if self.reflection_enabled:
            gpu_score += 10
        if self.antialiasing:
            gpu_score += 8
        if self.post_processing:
            gpu_score += 12
        
        # Expression/lip sync impact
        if self.expression_enabled:
            cpu_score += 5
            if self.lip_sync_enabled:
                cpu_score += 10 * quality_mult.get(self.lip_sync_quality, 1.0)
            if self.emotion_detection:
                cpu_score += 15
        
        # Threading impact (reduces effective CPU per core)
        if self.multithreading and self.max_threads > 1:
            cpu_score = cpu_score / min(self.max_threads, 4) * 1.2
        
        # Classify
        def classify(score, thresholds):
            if score < thresholds[0]:
                return "Very Low"
            elif score < thresholds[1]:
                return "Low"
            elif score < thresholds[2]:
                return "Medium"
            elif score < thresholds[3]:
                return "High"
            else:
                return "Very High"
        
        return {
            "cpu": classify(cpu_score, [10, 25, 50, 80]),
            "gpu": classify(gpu_score, [15, 30, 60, 90]),
            "memory": f"{memory_score} MB",
            "mode": self.mode.value,
            "recommended_for": self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        """Get system recommendation based on settings."""
        if self.mode == PerformanceMode.STATIC:
            return "Any system (no animations)"
        elif self.mode == PerformanceMode.MINIMAL:
            return "Raspberry Pi, low-end laptops"
        elif self.mode == PerformanceMode.LIGHT:
            return "Laptops, integrated graphics"
        elif self.mode == PerformanceMode.BALANCED:
            return "Desktop with dedicated GPU"
        elif self.mode == PerformanceMode.HIGH:
            return "Gaming PC, workstation"
        else:
            return "High-end gaming PC"


# Global configuration
_performance_config: Optional[AvatarPerformanceConfig] = None
_config_path: Optional[Path] = None


def get_performance_config() -> AvatarPerformanceConfig:
    """Get or create global performance configuration."""
    global _performance_config, _config_path
    
    if _performance_config is None:
        from ..config import CONFIG
        _config_path = Path(CONFIG.get("data_dir", "data")) / "avatar_performance.json"
        
        # Try to load saved config
        if _config_path.exists():
            try:
                with open(_config_path) as f:
                    data = json.load(f)
                _performance_config = AvatarPerformanceConfig.from_dict(data)
                logger.info(f"Loaded avatar performance config: {_performance_config.mode.value}")
            except Exception as e:
                logger.warning(f"Could not load performance config: {e}")
                _performance_config = AvatarPerformanceConfig()
        else:
            _performance_config = AvatarPerformanceConfig()
    
    return _performance_config


def save_performance_config():
    """Save current performance configuration."""
    global _performance_config, _config_path
    
    if _performance_config is None or _config_path is None:
        return
    
    try:
        _config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_config_path, 'w') as f:
            json.dump(_performance_config.to_dict(), f, indent=2)
        logger.debug("Saved avatar performance config")
    except Exception as e:
        logger.error(f"Could not save performance config: {e}")


def set_performance_mode(mode: PerformanceMode):
    """
    Set avatar performance mode.
    
    Args:
        mode: Performance mode preset
    """
    config = get_performance_config()
    config.apply_preset(mode.value)
    save_performance_config()
    
    logger.info(f"Avatar performance mode set to: {mode.value}")


def auto_detect_performance_mode() -> PerformanceMode:
    """
    Auto-detect recommended performance mode based on system capabilities.
    
    Returns:
        Recommended PerformanceMode
    """
    import platform
    
    try:
        import psutil
        
        # Check RAM
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        # Check CPU cores
        cpu_count = psutil.cpu_count(logical=False) or 1
        
        # Simple heuristic
        if ram_gb < 2 or cpu_count == 1:
            return PerformanceMode.MINIMAL
        elif ram_gb < 4 or cpu_count <= 2:
            return PerformanceMode.LIGHT
        elif ram_gb < 8:
            return PerformanceMode.BALANCED
        elif ram_gb < 16:
            return PerformanceMode.HIGH
        else:
            return PerformanceMode.FULL
            
    except ImportError:
        # psutil not available, check platform
        if platform.machine() in ['armv7l', 'aarch64']:
            return PerformanceMode.MINIMAL  # Likely RPi
        else:
            return PerformanceMode.BALANCED
    except Exception as e:
        logger.warning(f"Could not detect system capabilities: {e}")
        return PerformanceMode.BALANCED
