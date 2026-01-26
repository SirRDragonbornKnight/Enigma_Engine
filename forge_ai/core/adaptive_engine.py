"""
================================================================================
Adaptive Engine - Smart inference that adapts to context.
================================================================================

This module provides a smart wrapper around ForgeEngine that automatically
adjusts its behavior based on:

1. Gaming Mode: Reduces resources when games are detected
2. Device Capabilities: Uses optimal settings for hardware
3. Distributed Mode: Offloads to more powerful devices when available
4. Power Saving: Reduces consumption on battery or embedded devices

USAGE:
    from forge_ai.core.adaptive_engine import AdaptiveEngine
    
    # Create with automatic detection
    engine = AdaptiveEngine()
    
    # Generate - automatically adapts to current context
    response = engine.generate("Hello world")
    
    # Check current mode
    print(engine.get_status())

INTEGRATION:
    AdaptiveEngine wraps ForgeEngine and adds these layers:
    
    ┌──────────────────────────────────────────────────────────────────────┐
    │                      AdaptiveEngine                                   │
    │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐                 │
    │  │ Gaming Mode │  │ Distributed │  │ Device       │                 │
    │  │  (context)  │  │   (network) │  │  Profiles    │                 │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘                 │
    │         │                │                 │                          │
    │         └────────────────┴─────────────────┘                          │
    │                          │                                            │
    │                    ┌─────┴─────┐                                     │
    │                    │ForgeEngine│                                     │
    │                    └───────────┘                                     │
    └──────────────────────────────────────────────────────────────────────┘
"""

import logging
import time
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any, Generator, List
from pathlib import Path

logger = logging.getLogger(__name__)


class AdaptiveMode(Enum):
    """Current adaptive mode."""
    FULL = auto()           # Full power, all features
    GAMING = auto()         # Reduced resources for gaming
    DISTRIBUTED = auto()    # Offloading to remote server
    LOW_POWER = auto()      # Battery/embedded mode
    OFFLINE = auto()        # Builtin fallbacks only


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive behavior."""
    enable_gaming_mode: bool = True
    enable_distributed: bool = True
    enable_low_power: bool = True
    enable_builtin_fallback: bool = True
    
    # Gaming mode settings
    gaming_cpu_only: bool = True
    gaming_max_tokens: int = 50
    
    # Distributed settings
    server_url: Optional[str] = None
    offload_threshold_ms: int = 5000  # Offload if local would take longer
    
    # Low power settings
    low_power_max_tokens: int = 30
    low_power_cpu_only: bool = True


class AdaptiveEngine:
    """
    Smart inference engine that adapts to context.
    
    This wraps ForgeEngine and automatically:
    - Detects and respects gaming mode
    - Offloads to distributed servers when beneficial
    - Uses low-power mode on constrained devices
    - Falls back to builtins when models unavailable
    """
    
    def __init__(
        self,
        config: AdaptiveConfig = None,
        model_path: Optional[Path] = None,
        **engine_kwargs
    ):
        """
        Initialize adaptive engine.
        
        Args:
            config: Adaptive behavior configuration
            model_path: Path to model (passed to ForgeEngine)
            **engine_kwargs: Additional args for ForgeEngine
        """
        self.config = config or AdaptiveConfig()
        self._model_path = model_path
        self._engine_kwargs = engine_kwargs
        
        # Lazy-loaded components
        self._forge_engine = None
        self._low_power_engine = None
        self._gaming_mode = None
        self._distributed = None
        
        # Current state
        self._current_mode = AdaptiveMode.FULL
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "total_generations": 0,
            "local_generations": 0,
            "distributed_generations": 0,
            "fallback_generations": 0,
            "gaming_mode_activations": 0,
        }
        
        # Initialize gaming mode detection
        if self.config.enable_gaming_mode:
            self._init_gaming_mode()
        
        # Initialize distributed connection
        if self.config.enable_distributed and self.config.server_url:
            self._init_distributed()
    
    def _init_gaming_mode(self):
        """Initialize gaming mode detection."""
        try:
            from .gaming_mode import get_gaming_mode
            self._gaming_mode = get_gaming_mode()
            self._gaming_mode.on_game_start(self._on_game_start)
            self._gaming_mode.on_game_end(self._on_game_end)
            self._gaming_mode.enable()
            logger.info("Gaming mode detection enabled")
        except ImportError as e:
            logger.warning(f"Gaming mode not available: {e}")
    
    def _init_distributed(self):
        """Initialize distributed connection."""
        try:
            from ..comms.distributed import DistributedNode, NodeRole
            self._distributed = DistributedNode(
                "adaptive_client",
                role=NodeRole.INFERENCE_CLIENT,
            )
            if self.config.server_url:
                if self._distributed.connect(self.config.server_url):
                    logger.info(f"Connected to distributed server: {self.config.server_url}")
                else:
                    logger.warning(f"Could not connect to: {self.config.server_url}")
        except ImportError as e:
            logger.warning(f"Distributed mode not available: {e}")
    
    def _on_game_start(self, game: str, profile):
        """Handle game start event."""
        logger.info(f"Game detected: {game}, switching to gaming mode")
        self._stats["gaming_mode_activations"] += 1
        self._current_mode = AdaptiveMode.GAMING
    
    def _on_game_end(self, game: str):
        """Handle game end event."""
        logger.info(f"Game ended: {game}, restoring full mode")
        self._current_mode = AdaptiveMode.FULL
    
    @property
    def forge_engine(self):
        """Get or create ForgeEngine (lazy loaded)."""
        if self._forge_engine is None:
            try:
                from .inference import ForgeEngine
                kwargs = self._engine_kwargs.copy()
                if self._model_path:
                    kwargs["model_path"] = self._model_path
                self._forge_engine = ForgeEngine(**kwargs)
                logger.info("ForgeEngine loaded")
            except Exception as e:
                logger.error(f"Could not load ForgeEngine: {e}")
        return self._forge_engine
    
    @property
    def low_power_engine(self):
        """Get or create low-power engine."""
        if self._low_power_engine is None:
            try:
                from .low_power_inference import LowPowerEngine
                self._low_power_engine = LowPowerEngine()
                logger.info("LowPowerEngine loaded")
            except Exception as e:
                logger.warning(f"LowPowerEngine not available: {e}")
        return self._low_power_engine
    
    def _should_use_distributed(self) -> bool:
        """Check if we should offload to distributed server."""
        if not self.config.enable_distributed:
            return False
        if not self._distributed:
            return False
        if not self._distributed.peers:
            return False
        
        # Check if any peer has inference capability
        for peer in self._distributed.peers.values():
            if "inference" in peer.capabilities:
                return True
        
        return False
    
    def _should_use_low_power(self) -> bool:
        """Check if we should use low-power mode."""
        if not self.config.enable_low_power:
            return False
        
        try:
            from .device_profiles import get_device_profiler, DeviceClass
            profiler = get_device_profiler()
            device_class = profiler.classify()
            
            return device_class in {
                DeviceClass.EMBEDDED,
                DeviceClass.MOBILE,
                DeviceClass.LAPTOP_LOW,
            }
        except ImportError:
            return False
    
    def _get_effective_mode(self) -> AdaptiveMode:
        """Determine the effective mode based on current context."""
        # Gaming mode takes priority
        if self._gaming_mode and self._gaming_mode.active_game:
            return AdaptiveMode.GAMING
        
        # Check for distributed capability
        if self._should_use_distributed():
            return AdaptiveMode.DISTRIBUTED
        
        # Check for low-power device
        if self._should_use_low_power():
            return AdaptiveMode.LOW_POWER
        
        return AdaptiveMode.FULL
    
    def _get_adjusted_params(self, mode: AdaptiveMode, **kwargs) -> Dict[str, Any]:
        """Adjust generation parameters based on mode."""
        params = kwargs.copy()
        
        if mode == AdaptiveMode.GAMING:
            # Reduce generation for gaming
            params.setdefault("max_gen", self.config.gaming_max_tokens)
            if self.config.gaming_cpu_only:
                params["device"] = "cpu"
        
        elif mode == AdaptiveMode.LOW_POWER:
            # Minimal generation for low-power
            params.setdefault("max_gen", self.config.low_power_max_tokens)
            if self.config.low_power_cpu_only:
                params["device"] = "cpu"
            params.setdefault("temperature", 0.5)  # More focused
        
        return params
    
    def generate(
        self,
        prompt: str,
        max_gen: int = 100,
        temperature: float = 0.8,
        **kwargs
    ) -> str:
        """
        Generate text, automatically adapting to current context.
        
        Args:
            prompt: Input text
            max_gen: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        with self._lock:
            self._stats["total_generations"] += 1
            
            # Determine current mode
            mode = self._get_effective_mode()
            self._current_mode = mode
            
            # Adjust parameters based on mode
            params = self._get_adjusted_params(
                mode,
                max_gen=max_gen,
                temperature=temperature,
                **kwargs
            )
            
            # Route to appropriate engine
            try:
                if mode == AdaptiveMode.DISTRIBUTED:
                    return self._generate_distributed(prompt, params)
                elif mode == AdaptiveMode.LOW_POWER:
                    return self._generate_low_power(prompt, params)
                elif mode == AdaptiveMode.GAMING:
                    return self._generate_gaming(prompt, params)
                else:
                    return self._generate_local(prompt, params)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                return self._generate_fallback(prompt, params)
    
    def _generate_local(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate using local ForgeEngine."""
        self._stats["local_generations"] += 1
        
        engine = self.forge_engine
        if engine is None:
            return self._generate_fallback(prompt, params)
        
        return engine.generate(prompt, **params)
    
    def _generate_distributed(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate using distributed server."""
        self._stats["distributed_generations"] += 1
        
        try:
            result = self._distributed.generate(
                prompt,
                max_tokens=params.get("max_gen", 100),
                temperature=params.get("temperature", 0.8),
            )
            return result
        except Exception as e:
            logger.warning(f"Distributed generation failed: {e}, falling back to local")
            return self._generate_local(prompt, params)
    
    def _generate_low_power(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate using low-power engine."""
        engine = self.low_power_engine
        if engine:
            return engine.generate(prompt, **params)
        
        # Fall back to local with adjusted params
        return self._generate_local(prompt, params)
    
    def _generate_gaming(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate in gaming mode (minimal resources)."""
        # Check if we can offload
        if self._should_use_distributed():
            return self._generate_distributed(prompt, params)
        
        # Use low-power if available
        if self.low_power_engine:
            return self._generate_low_power(prompt, params)
        
        # Use local with gaming params
        return self._generate_local(prompt, params)
    
    def _generate_fallback(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate using builtin fallback."""
        self._stats["fallback_generations"] += 1
        
        if not self.config.enable_builtin_fallback:
            raise RuntimeError("No generation engine available")
        
        try:
            from ..builtin import BuiltinChat
            chat = BuiltinChat()
            chat.load()
            return chat.respond(prompt)
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return "I'm sorry, I couldn't generate a response right now."
    
    def stream_generate(
        self,
        prompt: str,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream generate text token by token.
        
        Note: Streaming may not be available in all modes.
        """
        mode = self._get_effective_mode()
        
        # Only local engine supports streaming
        if mode == AdaptiveMode.FULL:
            engine = self.forge_engine
            if engine and hasattr(engine, "stream_generate"):
                yield from engine.stream_generate(prompt, **kwargs)
                return
        
        # Fall back to non-streaming
        result = self.generate(prompt, **kwargs)
        yield result
    
    def chat(self, message: str, **kwargs) -> str:
        """
        Chat with conversation history.
        
        Args:
            message: User message
            **kwargs: Additional parameters
            
        Returns:
            AI response
        """
        mode = self._get_effective_mode()
        
        if mode == AdaptiveMode.FULL:
            engine = self.forge_engine
            if engine and hasattr(engine, "chat"):
                return engine.chat(message, **kwargs)
        
        # Fall back to basic generation
        return self.generate(f"User: {message}\nAssistant:", **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            "mode": self._current_mode.name,
            "gaming_active": self._gaming_mode.active_game if self._gaming_mode else None,
            "distributed_connected": bool(self._distributed and self._distributed.peers),
            "forge_engine_loaded": self._forge_engine is not None,
            "low_power_available": self._low_power_engine is not None,
            "stats": self._stats.copy(),
        }
    
    def set_server(self, url: str) -> bool:
        """
        Set distributed server URL.
        
        Args:
            url: Server URL (e.g., "192.168.1.100:5000")
            
        Returns:
            True if connected successfully
        """
        self.config.server_url = url
        
        if self._distributed:
            return self._distributed.connect(url)
        
        self._init_distributed()
        return bool(self._distributed and self._distributed.peers)
    
    def enable_gaming_mode(self, enable: bool = True):
        """Enable or disable gaming mode detection."""
        self.config.enable_gaming_mode = enable
        
        if enable and not self._gaming_mode:
            self._init_gaming_mode()
        elif not enable and self._gaming_mode:
            self._gaming_mode.disable()


# Convenience function
def get_adaptive_engine(**kwargs) -> AdaptiveEngine:
    """Get or create the global adaptive engine."""
    global _adaptive_engine
    if '_adaptive_engine' not in globals() or _adaptive_engine is None:
        _adaptive_engine = AdaptiveEngine(**kwargs)
    return _adaptive_engine


__all__ = [
    'AdaptiveEngine',
    'AdaptiveConfig',
    'AdaptiveMode',
    'get_adaptive_engine',
]
