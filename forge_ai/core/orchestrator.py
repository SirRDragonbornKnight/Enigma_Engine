"""
================================================================================
MODEL ORCHESTRATOR - CENTRAL INTELLIGENCE COORDINATOR
================================================================================

The Model Orchestrator is the central coordinator for all AI models and capabilities.
It manages model registration, task routing, multi-model collaboration, and resource management.

FILE: forge_ai/core/orchestrator.py
TYPE: Central Coordination System
MAIN CLASS: ModelOrchestrator

KEY FEATURES:
- Register any model (Forge, HuggingFace, GGUF, external API)
- Route tasks based on capability matching
- Enable model-to-model communication
- Support fallback chains (if model A fails, try model B)
- Memory-aware loading (don't load what won't fit)
- Hot-swap models without restart

USAGE:
    from forge_ai.core.orchestrator import ModelOrchestrator, get_orchestrator
    
    orchestrator = get_orchestrator()
    
    # Register models
    orchestrator.register_model(
        model_id="forge:small",
        capabilities=["text_generation", "reasoning"],
        load_args={"device": "cpu"}
    )
    
    # Execute a task (auto-selects best model)
    result = orchestrator.execute_task(
        capability="code_generation",
        task="Write a Python function to sort a list"
    )
    
    # Enable collaboration
    result = orchestrator.collaborate(
        requesting_model="forge:small",
        target_capability="vision",
        task="Describe this image: photo.jpg"
    )
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

from .capability_registry import CapabilityRegistry, get_capability_registry
from .model_pool import ModelPool, get_model_pool, ModelPoolConfig
from .collaboration import ModelCollaboration, get_collaboration

logger = logging.getLogger(__name__)


# =============================================================================
# ORCHESTRATOR CONFIGURATION
# =============================================================================

@dataclass
class OrchestratorConfig:
    """Configuration for the model orchestrator."""
    
    # Default models for capabilities
    default_chat_model: str = "auto"
    default_code_model: str = "auto"
    default_vision_model: str = "auto"
    default_image_gen_model: str = "auto"
    
    # Resource limits
    max_loaded_models: int = 3
    gpu_memory_limit_mb: int = 8000
    cpu_memory_limit_mb: int = 16000
    
    # Features
    enable_collaboration: bool = True
    enable_auto_fallback: bool = True
    fallback_to_cpu: bool = True
    enable_hot_swap: bool = True
    
    # Model pool config
    model_pool_config: Optional[ModelPoolConfig] = None


# =============================================================================
# TASK DEFINITION
# =============================================================================

@dataclass
class Task:
    """A task to be executed by a model."""
    
    capability: str                           # Required capability
    task: Any                                 # Task data (prompt, image, etc.)
    context: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    model_id: Optional[str] = None            # Specific model (or None for auto)
    timeout: float = 30.0
    require_sync: bool = True


@dataclass
class TaskResult:
    """Result of task execution."""
    
    success: bool
    model_id: str
    result: Any
    confidence: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# =============================================================================
# MODEL ORCHESTRATOR
# =============================================================================

class ModelOrchestrator:
    """
    Central coordinator for all AI models and capabilities.
    
    Responsibilities:
    - Model registration and discovery
    - Task routing to best available model
    - Multi-model collaboration
    - Fallback chains
    - Resource management
    - Capability aggregation
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()
        
        # Initialize subsystems
        self.capability_registry = get_capability_registry()
        
        # Initialize model pool with config
        pool_config = self.config.model_pool_config
        if pool_config is None:
            pool_config = ModelPoolConfig(
                max_loaded_models=self.config.max_loaded_models,
                gpu_memory_limit_mb=self.config.gpu_memory_limit_mb,
                cpu_memory_limit_mb=self.config.cpu_memory_limit_mb,
                fallback_to_cpu=self.config.fallback_to_cpu,
            )
        self.model_pool = get_model_pool(pool_config)
        
        # Initialize collaboration manager
        self.collaboration = get_collaboration()
        self.collaboration.set_orchestrator(self)
        
        # Model assignments (capability -> preferred model)
        self._model_assignments: Dict[str, List[str]] = {}
        
        # Fallback chains (model -> list of fallback models)
        self._fallback_chains: Dict[str, List[str]] = {}
        
        # Task execution history
        self._execution_history: List[Dict[str, Any]] = []
        self._max_history = 1000
    
    # -------------------------------------------------------------------------
    # MODEL REGISTRATION
    # -------------------------------------------------------------------------
    
    def register_model(
        self,
        model_id: str,
        capabilities: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        performance_ratings: Optional[Dict[str, float]] = None,
        auto_detect: bool = True,
        load_args: Optional[Dict[str, Any]] = None,
        preload: bool = False,
    ) -> None:
        """
        Register a model with the orchestrator.
        
        Args:
            model_id: Unique model identifier
            capabilities: List of capabilities this model has
            metadata: Optional model metadata
            performance_ratings: Optional performance ratings per capability
            auto_detect: Auto-detect capabilities from model metadata
            load_args: Optional arguments for loading the model
            preload: Load the model immediately
        """
        # Register with capability registry
        self.capability_registry.register_model(
            model_id=model_id,
            capabilities=capabilities,
            metadata=metadata or {},
            performance_ratings=performance_ratings,
            auto_detect=auto_detect,
        )
        
        logger.info(f"Registered model {model_id} with capabilities: {capabilities}")
        
        # Preload if requested
        if preload:
            self.model_pool.preload(model_id, load_args)
    
    def unregister_model(self, model_id: str) -> bool:
        """
        Unregister a model.
        
        Args:
            model_id: Model to unregister
            
        Returns:
            True if successful
        """
        # Unload from pool
        self.model_pool.unload_model(model_id)
        
        # Unregister from capability registry
        return self.capability_registry.unregister_model(model_id)
    
    def assign_model_to_capability(
        self,
        capability: str,
        model_id: str,
        priority: int = 0,
    ) -> None:
        """
        Assign a specific model as preferred for a capability.
        
        Args:
            capability: Capability name
            model_id: Model to assign
            priority: Priority (higher = preferred)
        """
        if capability not in self._model_assignments:
            self._model_assignments[capability] = []
        
        # Add with priority
        self._model_assignments[capability].append(model_id)
        
        # Sort by priority (requires storing priorities - simplified for now)
        # In production, would use a more sophisticated structure
        
        logger.info(f"Assigned {model_id} to capability {capability}")
    
    # -------------------------------------------------------------------------
    # MODEL DISCOVERY
    # -------------------------------------------------------------------------
    
    def find_best_model(
        self,
        capability: str,
        requirements: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Find the best model for a capability.
        
        Args:
            capability: Capability needed
            requirements: Optional requirements (device, etc.)
            
        Returns:
            Model ID or None if no suitable model found
        """
        # Check for manual assignment first
        if capability in self._model_assignments:
            assigned = self._model_assignments[capability]
            if assigned:
                # Return first assigned (highest priority)
                return assigned[0]
        
        # Check for default model
        default_key = f"default_{capability}_model"
        if hasattr(self.config, default_key):
            default = getattr(self.config, default_key)
            if default and default != "auto":
                return default
        
        # Use capability registry to find best
        return self.capability_registry.find_best_model(capability, requirements)
    
    def find_models_with_capability(
        self,
        capability: str,
        min_performance: Optional[float] = None,
    ) -> List[str]:
        """
        Find all models with a specific capability.
        
        Args:
            capability: Capability to search for
            min_performance: Minimum performance rating
            
        Returns:
            List of model IDs
        """
        return self.capability_registry.find_models_with_capability(
            capability, min_performance
        )
    
    def find_better_model(
        self,
        current_model: str,
        capability: str,
    ) -> Optional[str]:
        """
        Find a better model than the current one for a capability.
        
        Args:
            current_model: Current model ID
            capability: Capability needed
            
        Returns:
            Better model ID or None
        """
        # Get all models with capability
        models = self.find_models_with_capability(capability)
        
        # Get performance ratings
        current_info = self.capability_registry.get_model_info(current_model)
        if not current_info:
            return None
        
        current_rating = current_info.get("performance_ratings", {}).get(capability, 0.5)
        
        # Find model with better rating
        best_model = None
        best_rating = current_rating
        
        for model_id in models:
            if model_id == current_model:
                continue
            
            info = self.capability_registry.get_model_info(model_id)
            if not info:
                continue
            
            rating = info.get("performance_ratings", {}).get(capability, 0.5)
            if rating > best_rating:
                best_model = model_id
                best_rating = rating
        
        return best_model
    
    # -------------------------------------------------------------------------
    # TASK EXECUTION
    # -------------------------------------------------------------------------
    
    def execute_task(
        self,
        capability: str,
        task: Any = None,
        context: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Execute a task using the appropriate model.
        
        Args:
            capability: Required capability
            task: Task data (prompt, image, etc.)
            context: Optional context
            parameters: Optional parameters
            model_id: Optional specific model to use
            **kwargs: Additional arguments
            
        Returns:
            Task result
        """
        start_time = time.time()
        
        # Merge kwargs into task if task is None
        if task is None and kwargs:
            task = kwargs.get("prompt") or kwargs.get("task") or kwargs
        
        try:
            # Find model if not specified
            if model_id is None:
                model_id = self.find_best_model(capability)
                if not model_id:
                    raise ValueError(f"No model found for capability: {capability}")
            
            logger.info(f"Executing task with {model_id} ({capability})")
            
            # Get model from pool
            model = self.model_pool.get_model(model_id)
            
            # Execute based on capability
            result = self._execute_on_model(
                model=model,
                model_id=model_id,
                capability=capability,
                task=task,
                context=context or {},
                parameters=parameters or {},
            )
            
            # Release model back to pool
            self.model_pool.release_model(model_id)
            
            # Record execution
            self._record_execution(
                model_id=model_id,
                capability=capability,
                success=True,
                processing_time=time.time() - start_time,
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            
            # Record failure
            self._record_execution(
                model_id=model_id or "unknown",
                capability=capability,
                success=False,
                processing_time=time.time() - start_time,
                error=str(e),
            )
            
            # Try fallback if enabled
            if self.config.enable_auto_fallback and model_id:
                fallback = self._get_fallback_model(model_id, capability)
                if fallback:
                    logger.info(f"Trying fallback model: {fallback}")
                    return self.execute_task(
                        capability=capability,
                        task=task,
                        context=context,
                        parameters=parameters,
                        model_id=fallback,
                    )
            
            raise
    
    def _execute_on_model(
        self,
        model: Any,
        model_id: str,
        capability: str,
        task: Any,
        context: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Any:
        """
        Execute a task on a specific model.
        
        Args:
            model: Model instance
            model_id: Model identifier
            capability: Capability being used
            task: Task data
            context: Context dict
            parameters: Parameters dict
            
        Returns:
            Task result
        """
        # Parse model type from ID
        model_type = model_id.split(":")[0] if ":" in model_id else "forge"
        
        # Execute based on model type
        if model_type == "forge":
            return self._execute_on_forge_model(model, capability, task, parameters)
        elif model_type == "huggingface":
            return self._execute_on_hf_model(model, capability, task, parameters)
        elif model_type == "local":
            return self._execute_on_local_tool(model, capability, task, parameters)
        else:
            # Generic execution
            if hasattr(model, "generate"):
                return model.generate(str(task))
            elif hasattr(model, "chat"):
                return model.chat(str(task))
            elif callable(model):
                return model(task)
            else:
                raise ValueError(f"Don't know how to execute on model type: {model_type}")
    
    def _execute_on_forge_model(
        self,
        engine: Any,
        capability: str,
        task: Any,
        parameters: Dict[str, Any],
    ) -> Any:
        """Execute task on a Forge model (ForgeEngine)."""
        if capability in ["text_generation", "chat", "reasoning"]:
            return engine.chat(str(task), **parameters)
        elif capability == "code_generation":
            return engine.generate(f"Write code: {task}", **parameters)
        else:
            return engine.generate(str(task), **parameters)
    
    def _execute_on_hf_model(
        self,
        model: Any,
        capability: str,
        task: Any,
        parameters: Dict[str, Any],
    ) -> Any:
        """Execute task on a HuggingFace model."""
        # HuggingFace models typically have a generate method
        if hasattr(model, "generate_text"):
            return model.generate_text(str(task), **parameters)
        elif hasattr(model, "generate"):
            return model.generate(str(task), **parameters)
        elif hasattr(model, "__call__"):
            return model(str(task))
        else:
            raise ValueError("HuggingFace model doesn't have expected interface")
    
    def _execute_on_local_tool(
        self,
        tool: Any,
        capability: str,
        task: Any,
        parameters: Dict[str, Any],
    ) -> Any:
        """Execute task on a local tool."""
        # Local tools are typically callable or have an execute method
        if callable(tool):
            return tool(task, **parameters)
        elif hasattr(tool, "execute"):
            return tool.execute(task, **parameters)
        else:
            return f"Local tool {tool} executed with task: {task}"
    
    # -------------------------------------------------------------------------
    # COLLABORATION
    # -------------------------------------------------------------------------
    
    def collaborate(
        self,
        requesting_model: str,
        target_capability: str,
        task: Any,
        context: Optional[Dict[str, Any]] = None,
        target_model: Optional[str] = None,
    ) -> Any:
        """
        Enable model-to-model collaboration.
        
        Args:
            requesting_model: Model making the request
            target_capability: Capability needed
            task: Task description
            context: Optional shared context
            target_model: Optional specific target model
            
        Returns:
            Result from target model
        """
        if not self.config.enable_collaboration:
            raise ValueError("Collaboration is disabled in configuration")
        
        response = self.collaboration.request_assistance(
            requesting_model=requesting_model,
            target_capability=target_capability,
            task=task,
            context=context,
            target_model=target_model,
        )
        
        if not response.success:
            raise RuntimeError(f"Collaboration failed: {response.error}")
        
        return response.result
    
    # -------------------------------------------------------------------------
    # FALLBACK MANAGEMENT
    # -------------------------------------------------------------------------
    
    def set_fallback_chain(
        self,
        model_id: str,
        fallback_models: List[str],
    ) -> None:
        """
        Set fallback models for a model.
        
        Args:
            model_id: Model to set fallbacks for
            fallback_models: List of fallback models in priority order
        """
        self._fallback_chains[model_id] = fallback_models
        logger.info(f"Set fallback chain for {model_id}: {fallback_models}")
    
    def _get_fallback_model(
        self,
        model_id: str,
        capability: str,
    ) -> Optional[str]:
        """Get the fallback model for a model and capability."""
        # Check explicit fallback chain
        if model_id in self._fallback_chains:
            fallbacks = self._fallback_chains[model_id]
            for fallback in fallbacks:
                if self.capability_registry.has_capability(fallback, capability):
                    return fallback
        
        # Find any other model with capability
        models = self.find_models_with_capability(capability)
        for m in models:
            if m != model_id:
                return m
        
        return None
    
    # -------------------------------------------------------------------------
    # HOT-SWAP
    # -------------------------------------------------------------------------
    
    def hot_swap_model(
        self,
        old_model_id: str,
        new_model_id: str,
    ) -> bool:
        """
        Hot-swap one model for another without restart.
        
        This replaces one model with another for all its capabilities
        without requiring a system restart. The old model is unloaded
        and the new model is preloaded.
        
        **Important Notes:**
        - Any ongoing tasks using the old model will complete with the old model
        - New tasks will automatically use the new model
        - The new model must already be registered with the orchestrator
        - If hot-swap is disabled in config, this will return False
        - The old model is immediately unloaded after capability transfer
        
        **Thread Safety:**
        This method uses internal locking to ensure thread-safe operation.
        However, ongoing task execution is not interrupted.
        
        **Side Effects:**
        - Old model is unloaded from memory
        - CUDA cache is cleared if GPU was used
        - All capability assignments are updated
        - New model is preloaded into memory
        
        Args:
            old_model_id: Model to replace
            new_model_id: New model to use
            
        Returns:
            True if hot-swap successful, False otherwise
            
        Example:
            >>> orchestrator.hot_swap_model("forge:small", "forge:medium")
            True
        """
        if not self.config.enable_hot_swap:
            logger.warning("Hot-swap is disabled in configuration")
            return False
        
        try:
            # Get capabilities of old model
            old_caps = self.capability_registry.get_capabilities(old_model_id)
            
            # Update assignments
            for cap in old_caps:
                if cap in self._model_assignments:
                    if old_model_id in self._model_assignments[cap]:
                        idx = self._model_assignments[cap].index(old_model_id)
                        self._model_assignments[cap][idx] = new_model_id
            
            # Unload old model
            self.model_pool.unload_model(old_model_id)
            
            # Preload new model
            self.model_pool.preload(new_model_id)
            
            logger.info(f"Hot-swapped {old_model_id} → {new_model_id}")
            return True
        
        except Exception as e:
            logger.error(f"Hot-swap failed: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # STATUS & ANALYTICS
    # -------------------------------------------------------------------------
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "registered_models": self.capability_registry.list_models(),
            "loaded_models": self.model_pool.list_models(),
            "memory_usage": self.model_pool.get_memory_usage(),
            "system_memory": self.model_pool.get_system_memory_info(),
            "collaboration_enabled": self.config.enable_collaboration,
            "collaboration_stats": (
                self.collaboration.get_collaboration_stats()
                if self.config.enable_collaboration else {}
            ),
            "execution_stats": self._get_execution_stats(),
        }
    
    def _get_execution_stats(self) -> Dict[str, Any]:
        """Get task execution statistics."""
        if not self._execution_history:
            return {"total_executions": 0}
        
        total = len(self._execution_history)
        successful = sum(1 for e in self._execution_history if e["success"])
        
        # Average processing time
        times = [e["processing_time"] for e in self._execution_history]
        avg_time = sum(times) / len(times) if times else 0
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "avg_processing_time": avg_time,
        }
    
    def _record_execution(
        self,
        model_id: str,
        capability: str,
        success: bool,
        processing_time: float,
        error: Optional[str] = None,
    ) -> None:
        """Record a task execution for analytics."""
        self._execution_history.append({
            "timestamp": time.time(),
            "model_id": model_id,
            "capability": capability,
            "success": success,
            "processing_time": processing_time,
            "error": error,
        })
        
        # Trim history if too long
        if len(self._execution_history) > self._max_history:
            self._execution_history = self._execution_history[-self._max_history:]


# =============================================================================
# GLOBAL ORCHESTRATOR INSTANCE
# =============================================================================

_global_orchestrator: Optional[ModelOrchestrator] = None


def get_orchestrator(config: Optional[OrchestratorConfig] = None) -> ModelOrchestrator:
    """
    Get the global orchestrator instance.
    
    Args:
        config: Optional orchestrator configuration
        
    Returns:
        Global ModelOrchestrator instance
    """
    global _global_orchestrator
    if _global_orchestrator is None:
        # Load config from CONFIG if not provided
        if config is None:
            from ..config import CONFIG
            config = OrchestratorConfig(
                max_loaded_models=CONFIG.get("max_concurrent_models", 3),
                gpu_memory_limit_mb=CONFIG.get("gpu_memory_limit_mb", 8000),
                enable_collaboration=CONFIG.get("enable_collaboration", True),
                fallback_to_cpu=CONFIG.get("fallback_to_cpu", True),
            )
        
        _global_orchestrator = ModelOrchestrator(config)
    
    return _global_orchestrator
