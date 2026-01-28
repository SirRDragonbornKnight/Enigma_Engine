"""
================================================================================
MODEL COLLABORATION - MODEL-TO-MODEL COMMUNICATION
================================================================================

Enables AI models to work together and hand off tasks to each other.

Example flows:
- Chat model detects code question → asks Code model → returns answer
- Chat model sees image reference → asks Vision model → incorporates description
- Small model on Pi → asks large model on PC for complex reasoning
- Confidence-based handoff (if confidence < threshold, ask another model)

FILE: forge_ai/core/collaboration.py
TYPE: Multi-Model Coordination
MAIN CLASS: ModelCollaboration

USAGE:
    from forge_ai.core.collaboration import ModelCollaboration, get_collaboration
    
    collab = get_collaboration()
    
    # Model A asks Model B for help
    response = collab.request_assistance(
        requesting_model="forge:small",
        target_capability="code_generation",
        task="Write a Python function to sort a list",
        context={"conversation": [...]}
    )
    
    # Confidence-based handoff
    response = collab.smart_handoff(
        model_id="forge:small",
        task="Explain quantum computing",
        confidence_threshold=0.7
    )
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# COLLABORATION TYPES
# =============================================================================

class CollaborationType(Enum):
    """Types of model collaboration."""
    REQUEST_RESPONSE = "request_response"  # Simple request/response
    CONFIDENCE_HANDOFF = "confidence_handoff"  # Hand off if confidence low
    PIPELINE = "pipeline"  # Multi-stage processing
    CONSENSUS = "consensus"  # Multiple models vote
    SPECIALIST = "specialist"  # Route to specialist model


# =============================================================================
# COLLABORATION REQUEST
# =============================================================================

@dataclass
class CollaborationRequest:
    """A request for inter-model collaboration."""
    
    requesting_model: str                     # ID of model making request
    target_capability: str                    # Capability needed
    target_model: Optional[str] = None        # Specific model (or None for auto)
    task: str = ""                            # Task description
    context: Dict[str, Any] = field(default_factory=dict)  # Shared context
    parameters: Dict[str, Any] = field(default_factory=dict)  # Task parameters
    collaboration_type: CollaborationType = CollaborationType.REQUEST_RESPONSE
    timeout_seconds: float = 30.0
    require_sync: bool = True                 # Wait for response vs async


@dataclass
class CollaborationResponse:
    """Response from a collaboration request."""
    
    success: bool
    responding_model: str
    result: Any
    confidence: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# =============================================================================
# MODEL COLLABORATION
# =============================================================================

class ModelCollaboration:
    """
    Manages collaboration between AI models.
    
    Features:
    - Request/response protocol between models
    - Context sharing (conversation history)
    - Confidence-based handoff
    - Async collaboration (don't block)
    - Pipeline processing (task flows through multiple models)
    """
    
    def __init__(self):
        """Initialize the collaboration manager."""
        self._orchestrator = None  # Set by orchestrator
        self._collaboration_history: List[Dict[str, Any]] = []
        self._max_history = 1000
    
    def set_orchestrator(self, orchestrator: Any) -> None:
        """
        Set the orchestrator instance.
        
        Args:
            orchestrator: ModelOrchestrator instance
        """
        self._orchestrator = orchestrator
    
    # -------------------------------------------------------------------------
    # REQUEST/RESPONSE COLLABORATION
    # -------------------------------------------------------------------------
    
    def request_assistance(
        self,
        requesting_model: str,
        target_capability: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        target_model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> CollaborationResponse:
        """
        Request assistance from another model.
        
        Args:
            requesting_model: ID of model making request
            target_capability: Capability needed (e.g., "code_generation")
            task: Task description
            context: Optional shared context
            target_model: Optional specific model to use
            parameters: Optional task parameters
            
        Returns:
            CollaborationResponse with result
        """
        if not self._orchestrator:
            return CollaborationResponse(
                success=False,
                responding_model="none",
                result=None,
                error="No orchestrator configured"
            )
        
        start_time = time.time()
        
        try:
            # Find appropriate model if not specified
            if target_model is None:
                target_model = self._orchestrator.find_best_model(target_capability)
                if not target_model:
                    return CollaborationResponse(
                        success=False,
                        responding_model="none",
                        result=None,
                        error=f"No model found for capability: {target_capability}"
                    )
            
            # Prepare request
            request_data = {
                "task": task,
                "context": context or {},
                "parameters": parameters or {},
            }
            
            # Execute task through orchestrator
            result = self._orchestrator.execute_task(
                model_id=target_model,
                capability=target_capability,
                **request_data
            )
            
            # Record collaboration
            self._record_collaboration(
                requesting_model=requesting_model,
                responding_model=target_model,
                capability=target_capability,
                success=True,
            )
            
            # Extract confidence from result if available
            confidence = self._extract_confidence(result)
            
            return CollaborationResponse(
                success=True,
                responding_model=target_model,
                result=result,
                confidence=confidence,
                processing_time=time.time() - start_time,
            )
        
        except Exception as e:
            logger.error(f"Collaboration request failed: {e}")
            return CollaborationResponse(
                success=False,
                responding_model=target_model or "unknown",
                result=None,
                error=str(e),
                processing_time=time.time() - start_time,
            )
    
    def _extract_confidence(self, result: Any) -> float:
        """
        Extract confidence score from model result.
        
        Looks for confidence in various formats:
        - Direct 'confidence' key in dict result
        - 'metadata.confidence' nested in dict
        - Object with confidence attribute
        - Logprobs-based calculation
        
        Returns:
            Float confidence between 0.0 and 1.0
        """
        if result is None:
            return 0.0
        
        # Dict result with confidence key
        if isinstance(result, dict):
            if 'confidence' in result:
                return float(result['confidence'])
            if 'metadata' in result and isinstance(result['metadata'], dict):
                if 'confidence' in result['metadata']:
                    return float(result['metadata']['confidence'])
            # Try logprobs-based confidence
            if 'logprobs' in result and result['logprobs']:
                import math
                # Average probability of tokens
                probs = [math.exp(lp) for lp in result['logprobs'] if lp is not None]
                if probs:
                    return sum(probs) / len(probs)
            # Perplexity to confidence conversion
            if 'perplexity' in result:
                # Lower perplexity = higher confidence
                ppl = float(result['perplexity'])
                return max(0.0, min(1.0, 1.0 / (1.0 + math.log(ppl) / 10)))
        
        # Object with confidence attribute
        if hasattr(result, 'confidence'):
            return float(result.confidence)
        
        # Default confidence
        return 1.0
    
    # -------------------------------------------------------------------------
    # CONFIDENCE-BASED HANDOFF
    # -------------------------------------------------------------------------
    
    def smart_handoff(
        self,
        model_id: str,
        capability: str,
        task: str,
        confidence_threshold: float = 0.7,
        context: Optional[Dict[str, Any]] = None,
        max_handoffs: int = 2,
    ) -> CollaborationResponse:
        """
        Execute task with confidence-based handoff to other models.
        
        If the model's confidence is below threshold, hand off to
        a more capable model.
        
        Args:
            model_id: Initial model to try
            capability: Capability needed
            task: Task description
            confidence_threshold: Minimum confidence to accept result
            context: Optional shared context
            max_handoffs: Maximum number of handoffs to attempt
            
        Returns:
            CollaborationResponse with result
        """
        if not self._orchestrator:
            return CollaborationResponse(
                success=False,
                responding_model="none",
                result=None,
                error="No orchestrator configured"
            )
        
        handoff_count = 0
        current_model = model_id
        
        while handoff_count <= max_handoffs:
            try:
                # Execute task
                result = self._orchestrator.execute_task(
                    model_id=current_model,
                    capability=capability,
                    task=task,
                    context=context or {},
                )
                
                # Check confidence (if available in result)
                confidence = self._extract_confidence(result)
                
                if confidence >= confidence_threshold:
                    # Success! Confidence is high enough
                    logger.info(
                        f"Task completed by {current_model} "
                        f"with confidence {confidence:.2f}"
                    )
                    return CollaborationResponse(
                        success=True,
                        responding_model=current_model,
                        result=result,
                        confidence=confidence,
                        metadata={"handoff_count": handoff_count},
                    )
                
                # Confidence too low, hand off to better model
                logger.info(
                    f"Confidence {confidence:.2f} below threshold "
                    f"{confidence_threshold:.2f}, handing off..."
                )
                
                # Find next best model
                next_model = self._orchestrator.find_better_model(
                    current_model=current_model,
                    capability=capability,
                )
                
                if not next_model or next_model == current_model:
                    # No better model available, return current result
                    logger.warning(
                        f"No better model available for handoff, "
                        f"using result from {current_model}"
                    )
                    return CollaborationResponse(
                        success=True,
                        responding_model=current_model,
                        result=result,
                        confidence=confidence,
                        metadata={
                            "handoff_count": handoff_count,
                            "note": "Low confidence but no better model available"
                        },
                    )
                
                current_model = next_model
                handoff_count += 1
            
            except Exception as e:
                logger.error(f"Smart handoff failed: {e}")
                return CollaborationResponse(
                    success=False,
                    responding_model=current_model,
                    result=None,
                    error=str(e),
                )
        
        # Max handoffs reached
        return CollaborationResponse(
            success=False,
            responding_model=current_model,
            result=None,
            error=f"Maximum handoffs ({max_handoffs}) reached",
        )
    
    # -------------------------------------------------------------------------
    # PIPELINE COLLABORATION
    # -------------------------------------------------------------------------
    
    def execute_pipeline(
        self,
        stages: List[Dict[str, Any]],
        initial_input: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> CollaborationResponse:
        """
        Execute a multi-stage pipeline where output of one model
        feeds into the next.
        
        Args:
            stages: List of stage definitions, each with:
                - capability: Capability needed
                - model_id: Optional specific model
                - transform: Optional function to transform output
            initial_input: Initial input data
            context: Optional shared context
            
        Returns:
            CollaborationResponse with final result
        """
        if not self._orchestrator:
            return CollaborationResponse(
                success=False,
                responding_model="none",
                result=None,
                error="No orchestrator configured"
            )
        
        current_input = initial_input
        pipeline_history = []
        
        try:
            for i, stage in enumerate(stages):
                capability = stage["capability"]
                model_id = stage.get("model_id")
                transform = stage.get("transform")
                
                # Find model if not specified
                if not model_id:
                    model_id = self._orchestrator.find_best_model(capability)
                    if not model_id:
                        return CollaborationResponse(
                            success=False,
                            responding_model="none",
                            result=None,
                            error=f"No model found for stage {i} capability: {capability}",
                        )
                
                # Execute stage
                logger.info(f"Pipeline stage {i+1}/{len(stages)}: {model_id} ({capability})")
                result = self._orchestrator.execute_task(
                    model_id=model_id,
                    capability=capability,
                    task=current_input,
                    context=context or {},
                )
                
                # Record stage
                pipeline_history.append({
                    "stage": i,
                    "model_id": model_id,
                    "capability": capability,
                    "input": current_input,
                    "output": result,
                })
                
                # Transform output for next stage if needed
                if transform:
                    current_input = transform(result)
                else:
                    current_input = result
            
            # Pipeline complete
            return CollaborationResponse(
                success=True,
                responding_model=f"pipeline:{len(stages)}stages",
                result=current_input,
                metadata={"pipeline_history": pipeline_history},
            )
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return CollaborationResponse(
                success=False,
                responding_model="pipeline",
                result=None,
                error=str(e),
                metadata={"pipeline_history": pipeline_history},
            )
    
    # -------------------------------------------------------------------------
    # CONSENSUS COLLABORATION
    # -------------------------------------------------------------------------
    
    def consensus(
        self,
        capability: str,
        task: str,
        num_models: int = 3,
        context: Optional[Dict[str, Any]] = None,
        voting_strategy: str = "majority",
    ) -> CollaborationResponse:
        """
        Get consensus from multiple models.
        
        Args:
            capability: Capability needed
            task: Task description
            num_models: Number of models to consult
            context: Optional shared context
            voting_strategy: "majority", "unanimous", or "best"
            
        Returns:
            CollaborationResponse with consensus result
        """
        if not self._orchestrator:
            return CollaborationResponse(
                success=False,
                responding_model="none",
                result=None,
                error="No orchestrator configured"
            )
        
        try:
            # Find models with capability
            models = self._orchestrator.find_models_with_capability(capability)
            if len(models) < num_models:
                logger.warning(
                    f"Only {len(models)} models available for consensus, "
                    f"requested {num_models}"
                )
                num_models = len(models)
            
            if num_models == 0:
                return CollaborationResponse(
                    success=False,
                    responding_model="none",
                    result=None,
                    error=f"No models found for capability: {capability}",
                )
            
            # Get responses from multiple models
            responses = []
            for model_id in models[:num_models]:
                try:
                    result = self._orchestrator.execute_task(
                        model_id=model_id,
                        capability=capability,
                        task=task,
                        context=context or {},
                    )
                    responses.append({
                        "model_id": model_id,
                        "result": result,
                        "confidence": self._extract_confidence(result),
                    })
                except Exception as e:
                    logger.warning(f"Model {model_id} failed in consensus: {e}")
            
            if not responses:
                return CollaborationResponse(
                    success=False,
                    responding_model="consensus",
                    result=None,
                    error="All models failed to respond",
                )
            
            # Apply voting strategy
            if voting_strategy == "majority":
                # Simple majority (most common result)
                from collections import Counter
                results = [r["result"] for r in responses]
                most_common = Counter(results).most_common(1)[0][0]
                consensus_result = most_common
            
            elif voting_strategy == "best":
                # Use result from model with highest confidence
                best = max(responses, key=lambda x: x["confidence"])
                consensus_result = best["result"]
            
            else:  # unanimous
                # All models must agree
                results = [r["result"] for r in responses]
                if len(set(results)) == 1:
                    consensus_result = results[0]
                else:
                    return CollaborationResponse(
                        success=False,
                        responding_model="consensus",
                        result=None,
                        error="No unanimous consensus reached",
                        metadata={"responses": responses},
                    )
            
            return CollaborationResponse(
                success=True,
                responding_model=f"consensus:{num_models}models",
                result=consensus_result,
                metadata={
                    "responses": responses,
                    "voting_strategy": voting_strategy,
                },
            )
        
        except Exception as e:
            logger.error(f"Consensus failed: {e}")
            return CollaborationResponse(
                success=False,
                responding_model="consensus",
                result=None,
                error=str(e),
            )
    
    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------
    
    def _extract_confidence(self, result: Any) -> float:
        """Extract confidence score from result if available."""
        if isinstance(result, dict):
            if "confidence" in result:
                return float(result["confidence"])
            if "score" in result:
                return float(result["score"])
        
        # Default confidence
        return 0.8
    
    def _record_collaboration(
        self,
        requesting_model: str,
        responding_model: str,
        capability: str,
        success: bool,
    ) -> None:
        """Record a collaboration event for analytics."""
        self._collaboration_history.append({
            "timestamp": time.time(),
            "requesting_model": requesting_model,
            "responding_model": responding_model,
            "capability": capability,
            "success": success,
        })
        
        # Trim history if too long
        if len(self._collaboration_history) > self._max_history:
            self._collaboration_history = self._collaboration_history[-self._max_history:]
    
    # -------------------------------------------------------------------------
    # ANALYTICS
    # -------------------------------------------------------------------------
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get statistics about model collaborations."""
        if not self._collaboration_history:
            return {"total_collaborations": 0}
        
        total = len(self._collaboration_history)
        successful = sum(1 for c in self._collaboration_history if c["success"])
        
        # Most common collaborations
        from collections import Counter
        pairs = [
            (c["requesting_model"], c["responding_model"])
            for c in self._collaboration_history
        ]
        common_pairs = Counter(pairs).most_common(5)
        
        return {
            "total_collaborations": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0,
            "common_collaborations": [
                {
                    "requesting": pair[0][0],
                    "responding": pair[0][1],
                    "count": pair[1],
                }
                for pair in common_pairs
            ],
        }


# =============================================================================
# GLOBAL COLLABORATION INSTANCE
# =============================================================================

_global_collaboration: Optional[ModelCollaboration] = None


def get_collaboration() -> ModelCollaboration:
    """
    Get the global collaboration manager instance.
    
    Returns:
        Global ModelCollaboration instance
    """
    global _global_collaboration
    if _global_collaboration is None:
        _global_collaboration = ModelCollaboration()
    return _global_collaboration
