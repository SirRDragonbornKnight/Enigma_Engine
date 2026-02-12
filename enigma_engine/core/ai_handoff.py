"""
AI Handoff System - Coordination for Multi-AI Task Delegation

When the main chat AI needs to delegate a task to another AI (image gen, code gen, etc.),
this system handles the handoff to:
1. Pause the main AI to free up memory and compute
2. Let the specialized AI run with full resources
3. Resume the main AI after completion

This is especially important for single-GPU setups where multiple AIs
competing for VRAM causes slowdowns.

Usage:
    from enigma_engine.core.ai_handoff import AIHandoffManager
    
    handoff = AIHandoffManager()
    
    # When chat AI needs to delegate to image AI
    with handoff.delegate_task("image_gen", free_memory=True):
        result = image_generator.generate("a cat")
    
    # Main AI automatically resumes after context exits

See Also:
    - enigma_engine/core/tool_router.py - Routes requests, uses handoff
    - enigma_engine/core/inference.py - Main chat AI, can pause during handoff
    - SUGGESTIONS.md - "AI Prompt Handoff" improvement
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HandoffState(Enum):
    """State of the AI handoff process."""
    IDLE = "idle"              # No handoff in progress
    PAUSING = "pausing"        # Main AI is being paused
    DELEGATED = "delegated"    # Task delegated to specialized AI
    RESUMING = "resuming"      # Specialized AI done, resuming main AI
    ERROR = "error"            # Error during handoff


@dataclass
class HandoffConfig:
    """Configuration for AI handoff behavior."""
    
    # Whether to aggressively free GPU memory when delegating
    free_gpu_memory: bool = True
    
    # Whether to move the main model to CPU during handoff
    offload_main_model: bool = False
    
    # Timeout for the delegated task (seconds)
    delegation_timeout: float = 300.0  # 5 minutes
    
    # Whether to show status messages
    verbose: bool = True
    
    # Callback when handoff state changes
    on_state_change: Optional[Callable[[HandoffState], None]] = None


@dataclass
class DelegationContext:
    """Context for a delegated task."""
    task_type: str              # e.g., "image_gen", "code_gen"
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    result: Any = None
    
    def duration(self) -> float:
        """Get duration of the task."""
        end = self.completed_at or time.time()
        return end - self.started_at


class AIHandoffManager:
    """
    Manages AI-to-AI task delegation with memory optimization.
    
    The handoff manager ensures that when the main chat AI delegates
    a task to a specialized AI (image, code, video, etc.), the main
    AI pauses to free up GPU memory and compute resources.
    
    This is critical for single-GPU setups where memory is limited.
    
    Thread Safety:
        All methods are thread-safe. Only one delegation can happen at a time.
    
    Example:
        >>> handoff = AIHandoffManager()
        >>> 
        >>> # Check if we can delegate
        >>> if handoff.can_delegate():
        >>>     with handoff.delegate_task("image_gen"):
        >>>         result = generate_image("a sunset")
        >>> 
        >>> # Get handoff statistics
        >>> stats = handoff.get_statistics()
    """
    
    _instance: Optional['AIHandoffManager'] = None
    _lock = threading.Lock()
    
    def __init__(self, config: Optional[HandoffConfig] = None):
        """
        Initialize the handoff manager.
        
        Args:
            config: Configuration for handoff behavior
        """
        self.config = config or HandoffConfig()
        self._state = HandoffState.IDLE
        self._state_lock = threading.Lock()
        self._current_delegation: Optional[DelegationContext] = None
        
        # Track statistics
        self._total_delegations = 0
        self._successful_delegations = 0
        self._total_delegation_time = 0.0
        
        # Track paused models
        self._paused_models: Dict[int, Any] = {}
        self._paused_model_devices: Dict[int, str] = {}
        
        # History of delegations
        self._history: List[DelegationContext] = []
        self._max_history = 100
    
    @classmethod
    def get_instance(cls) -> 'AIHandoffManager':
        """Get or create singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = AIHandoffManager()
            return cls._instance
    
    @property
    def state(self) -> HandoffState:
        """Get current handoff state."""
        with self._state_lock:
            return self._state
    
    def _set_state(self, new_state: HandoffState) -> None:
        """Set state and notify callback."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            
        if self.config.verbose:
            logger.info(f"[Handoff] State: {old_state.value} -> {new_state.value}")
        
        if self.config.on_state_change:
            try:
                self.config.on_state_change(new_state)
            except Exception as e:
                logger.warning(f"Handoff state callback error: {e}")
    
    def can_delegate(self) -> bool:
        """Check if a delegation can start."""
        return self.state == HandoffState.IDLE
    
    def is_delegating(self) -> bool:
        """Check if currently delegating."""
        return self.state == HandoffState.DELEGATED
    
    @contextmanager
    def delegate_task(
        self,
        task_type: str,
        pause_model: Optional[Any] = None,
        free_memory: bool = True
    ):
        """
        Context manager for delegating a task to a specialized AI.
        
        This pauses the main AI (if provided), frees GPU memory,
        and yields control to the specialized AI.
        
        Args:
            task_type: Type of task (e.g., "image_gen", "code_gen", "video_gen")
            pause_model: The main model to pause (optional)
            free_memory: Whether to free GPU memory during delegation
        
        Yields:
            DelegationContext for tracking the delegation
        
        Example:
            with handoff.delegate_task("image_gen", pause_model=chat_model):
                image = image_ai.generate("a cat")
        """
        if not self.can_delegate():
            logger.warning(f"[Handoff] Cannot delegate - already in state: {self.state.value}")
            yield None
            return
        
        context = DelegationContext(task_type=task_type)
        self._current_delegation = context
        
        try:
            # Phase 1: Pause main AI
            self._set_state(HandoffState.PAUSING)
            self._pause_main_ai(pause_model, free_memory)
            
            # Phase 2: Delegated - specialized AI runs
            self._set_state(HandoffState.DELEGATED)
            
            if self.config.verbose:
                logger.info(f"[Handoff] Delegated to: {task_type}")
            
            # Yield control to specialized AI
            yield context
            
            context.success = True
            
        except Exception as e:
            context.success = False
            context.error = str(e)
            self._set_state(HandoffState.ERROR)
            logger.error(f"[Handoff] Error during delegation: {e}")
            raise
        
        finally:
            # Phase 3: Resume main AI
            context.completed_at = time.time()
            self._set_state(HandoffState.RESUMING)
            self._resume_main_ai()
            
            # Update statistics
            self._total_delegations += 1
            if context.success:
                self._successful_delegations += 1
            self._total_delegation_time += context.duration()
            
            # Store history
            self._history.append(context)
            while len(self._history) > self._max_history:
                self._history.pop(0)
            
            self._current_delegation = None
            self._set_state(HandoffState.IDLE)
            
            if self.config.verbose:
                logger.info(f"[Handoff] Completed in {context.duration():.2f}s")
    
    def _pause_main_ai(self, model: Optional[Any], free_memory: bool) -> None:
        """
        Pause the main AI to free resources for the specialized AI.
        
        Actions:
        1. Move model to CPU (if configured)
        2. Clear GPU cache
        3. Force garbage collection
        """
        if model is not None:
            model_id = id(model)
            
            if self.config.offload_main_model:
                # Move model to CPU to free GPU memory
                try:
                    if hasattr(model, 'device'):
                        self._paused_model_devices[model_id] = str(model.device)
                    if hasattr(model, 'cpu'):
                        model.cpu()
                        logger.debug("[Handoff] Moved main model to CPU")
                except Exception as e:
                    logger.warning(f"[Handoff] Could not move model to CPU: {e}")
            
            self._paused_models[model_id] = model
        
        if free_memory and self.config.free_gpu_memory:
            self._free_gpu_memory()
    
    def _resume_main_ai(self) -> None:
        """
        Resume the main AI after the specialized AI completes.
        
        Actions:
        1. Move model back to GPU (if it was offloaded)
        2. Clear references to paused state
        """
        # Restore models to original devices
        for model_id, device in self._paused_model_devices.items():
            model = self._paused_models.get(model_id)
            if model is not None:
                try:
                    if hasattr(model, 'to') and device and 'cuda' in device:
                        model.to(device)
                        logger.debug(f"[Handoff] Restored model to {device}")
                except Exception as e:
                    logger.warning(f"[Handoff] Could not restore model to {device}: {e}")
        
        self._paused_models.clear()
        self._paused_model_devices.clear()
    
    def _free_gpu_memory(self) -> None:
        """Aggressively free GPU memory."""
        try:
            import torch
            if torch.cuda.is_available():
                # Free PyTorch cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Get memory stats
                if self.config.verbose:
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    logger.debug(f"[Handoff] GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[Handoff] Could not free GPU memory: {e}")
        
        # Force garbage collection
        gc.collect()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get handoff statistics."""
        return {
            "total_delegations": self._total_delegations,
            "successful_delegations": self._successful_delegations,
            "success_rate": (
                self._successful_delegations / self._total_delegations
                if self._total_delegations > 0 else 0.0
            ),
            "total_delegation_time": self._total_delegation_time,
            "average_delegation_time": (
                self._total_delegation_time / self._total_delegations
                if self._total_delegations > 0 else 0.0
            ),
            "current_state": self.state.value,
            "recent_delegations": len(self._history),
        }
    
    def get_recent_delegations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent delegation history."""
        return [
            {
                "task_type": ctx.task_type,
                "duration": ctx.duration(),
                "success": ctx.success,
                "error": ctx.error,
            }
            for ctx in self._history[-limit:]
        ]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_handoff_manager: Optional[AIHandoffManager] = None


def get_handoff_manager() -> AIHandoffManager:
    """Get the global handoff manager instance."""
    global _handoff_manager
    if _handoff_manager is None:
        _handoff_manager = AIHandoffManager.get_instance()
    return _handoff_manager


@contextmanager
def delegate_to_ai(
    task_type: str,
    pause_model: Optional[Any] = None,
    free_memory: bool = True
):
    """
    Convenience function for delegating a task to another AI.
    
    This is the main entry point for AI handoff. Use this when
    the main chat AI needs to delegate to a specialized AI.
    
    Args:
        task_type: Type of specialized AI ("image_gen", "code_gen", etc.)
        pause_model: Main model to pause during delegation
        free_memory: Whether to free GPU memory
    
    Example:
        from enigma_engine.core.ai_handoff import delegate_to_ai
        
        def chat_with_image(prompt):
            # Main AI detects image request
            if "draw" in prompt.lower():
                with delegate_to_ai("image_gen", pause_model=self.model):
                    # Image AI gets full GPU
                    result = image_ai.generate(prompt)
                return f"Here's your image: {result}"
            else:
                return main_ai.generate(prompt)
    """
    manager = get_handoff_manager()
    with manager.delegate_task(task_type, pause_model, free_memory) as ctx:
        yield ctx


def is_handoff_active() -> bool:
    """Check if a handoff is currently active."""
    return get_handoff_manager().is_delegating()


def get_handoff_stats() -> Dict[str, Any]:
    """Get handoff statistics."""
    return get_handoff_manager().get_statistics()
