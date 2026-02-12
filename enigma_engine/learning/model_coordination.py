"""
Multi-Model Coordination System
===============================

Coordinates multiple specialized models working together:
- Task delegation to appropriate specialist models
- Ensemble responses from multiple models
- Model consensus and voting
- Cascading inference (fallback chains)
- Parallel model execution

Usage:
    from enigma_engine.learning.model_coordination import ModelCoordinator
    
    coordinator = ModelCoordinator()
    
    # Register specialized models
    coordinator.register_model("code", code_model, specialties=["python", "javascript"])
    coordinator.register_model("creative", creative_model, specialties=["stories", "poetry"])
    coordinator.register_model("general", general_model, is_fallback=True)
    
    # Coordinated inference
    response = coordinator.generate("Write a Python function")
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

logger = logging.getLogger(__name__)


class CoordinationStrategy(Enum):
    """How to coordinate multiple models."""
    ROUTE = "route"  # Route to best model for task
    ENSEMBLE = "ensemble"  # Combine outputs from multiple models
    CASCADE = "cascade"  # Try models in order until success
    PARALLEL = "parallel"  # Run all in parallel, return first/best
    CONSENSUS = "consensus"  # Models vote on best response
    SPECIALIZE = "specialize"  # Each model handles part of the task


class ModelRole(Enum):
    """Role a model plays in coordination."""
    SPECIALIST = "specialist"  # Good at specific tasks
    GENERALIST = "generalist"  # Handles broad range
    FALLBACK = "fallback"  # Used when others fail
    CRITIC = "critic"  # Evaluates other models' outputs
    ROUTER = "router"  # Decides which model to use


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    model: Any  # The actual model object
    role: ModelRole = ModelRole.GENERALIST
    specialties: List[str] = field(default_factory=list)
    priority: int = 0  # Higher = preferred
    max_tokens: int = 2048
    is_available: bool = True
    
    # Performance tracking
    total_calls: int = 0
    successful_calls: int = 0
    total_latency: float = 0.0
    avg_quality_score: float = 0.5
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.5
        return self.successful_calls / self.total_calls
    
    def avg_latency(self) -> float:
        """Calculate average latency."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency / self.total_calls
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "role": self.role.value,
            "specialties": self.specialties,
            "priority": self.priority,
            "max_tokens": self.max_tokens,
            "is_available": self.is_available,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "total_latency": self.total_latency,
            "avg_quality_score": self.avg_quality_score,
        }


@dataclass
class GenerationRequest:
    """A request for coordinated generation."""
    prompt: str
    context: List[Dict[str, str]] = field(default_factory=list)
    max_tokens: int = 256
    temperature: float = 0.7
    strategy: CoordinationStrategy = CoordinationStrategy.ROUTE
    required_models: Optional[List[str]] = None
    excluded_models: Optional[List[str]] = None
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result from coordinated generation."""
    response: str
    model_used: str
    strategy_used: CoordinationStrategy
    latency: float
    confidence: float = 1.0
    alternatives: List[Tuple[str, str]] = field(default_factory=list)  # (model, response)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskClassifier:
    """
    Classifies tasks to determine which model(s) should handle them.
    
    Uses keyword matching and learned patterns.
    """
    
    def __init__(self):
        # Keyword mappings
        self.specialty_keywords = {
            "code": ["code", "function", "class", "debug", "programming", "python", "javascript", 
                     "algorithm", "compile", "syntax", "variable", "loop", "array", "api"],
            "creative": ["story", "poem", "creative", "imagine", "fiction", "write", "character",
                        "narrative", "plot", "setting", "metaphor", "rhyme"],
            "math": ["calculate", "math", "equation", "solve", "formula", "derivative", "integral",
                    "statistics", "probability", "algebra", "geometry"],
            "analysis": ["analyze", "compare", "evaluate", "review", "assess", "critique",
                        "summarize", "explain", "interpret"],
            "conversation": ["chat", "talk", "hello", "hi", "how are you", "what's up"],
        }
    
    def classify(self, text: str) -> List[Tuple[str, float]]:
        """
        Classify text into specialties with confidence scores.
        
        Returns sorted list of (specialty, confidence) tuples.
        """
        text_lower = text.lower()
        scores = {}
        
        for specialty, keywords in self.specialty_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[specialty] = score / len(keywords)
        
        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_scores:
            return [("general", 1.0)]
        
        return sorted_scores


class EnsembleCombiner:
    """
    Combines outputs from multiple models into a single response.
    """
    
    @staticmethod
    def majority_vote(responses: List[str]) -> str:
        """Simple majority voting (for shorter responses)."""
        if not responses:
            return ""
        if len(responses) == 1:
            return responses[0]
        
        # Hash-based voting for exact matches
        vote_counts: Dict[str, int] = {}
        for response in responses:
            # Normalize for comparison
            normalized = response.strip().lower()
            key = hashlib.md5(normalized.encode()).hexdigest()
            vote_counts[key] = vote_counts.get(key, 0) + 1
        
        # Find majority
        best_key = max(vote_counts, key=vote_counts.get)
        
        # Return original (non-normalized) response
        for response in responses:
            if hashlib.md5(response.strip().lower().encode()).hexdigest() == best_key:
                return response
        
        return responses[0]
    
    @staticmethod
    def longest_response(responses: List[str]) -> str:
        """Return the most detailed (longest) response."""
        if not responses:
            return ""
        return max(responses, key=len)
    
    @staticmethod
    def merge_responses(responses: List[str], separator: str = "\n\n") -> str:
        """Merge all responses with separator."""
        return separator.join(r.strip() for r in responses if r.strip())
    
    @staticmethod
    def weighted_combine(
        responses: List[Tuple[str, float]],  # (response, weight)
    ) -> str:
        """Combine responses weighted by model quality scores."""
        if not responses:
            return ""
        
        # For now, return highest weighted response
        # In future, could do actual text blending
        return max(responses, key=lambda x: x[1])[0]


class ModelCoordinator:
    """
    Coordinates multiple models for inference.
    
    Example:
        coordinator = ModelCoordinator()
        
        # Register models
        coordinator.register_model("code_expert", code_model, 
                                   specialties=["python", "code"])
        coordinator.register_model("general", general_model, is_fallback=True)
        
        # Generate with automatic routing
        result = coordinator.generate("Write a Python function to sort a list")
        print(result.response)
        print(f"Model used: {result.model_used}")
    """
    
    def __init__(
        self,
        default_strategy: CoordinationStrategy = CoordinationStrategy.ROUTE,
        max_parallel: int = 4,
    ):
        """
        Initialize coordinator.
        
        Args:
            default_strategy: Default coordination strategy
            max_parallel: Maximum parallel model calls
        """
        self.default_strategy = default_strategy
        self.max_parallel = max_parallel
        
        self.models: Dict[str, ModelInfo] = {}
        self.classifier = TaskClassifier()
        self.combiner = EnsembleCombiner()
        
        self._executor = ThreadPoolExecutor(max_workers=max_parallel)
        self._lock = threading.Lock()
    
    def register_model(
        self,
        name: str,
        model: Any,
        role: ModelRole = ModelRole.GENERALIST,
        specialties: Optional[List[str]] = None,
        priority: int = 0,
        is_fallback: bool = False,
        max_tokens: int = 2048,
    ) -> None:
        """
        Register a model with the coordinator.
        
        Args:
            name: Unique model name
            model: Model object (must have generate() method)
            role: Model's role in coordination
            specialties: List of task types this model excels at
            priority: Higher priority models are preferred
            is_fallback: Use as fallback when others fail
            max_tokens: Maximum tokens this model can handle
        """
        if is_fallback:
            role = ModelRole.FALLBACK
        
        info = ModelInfo(
            name=name,
            model=model,
            role=role,
            specialties=specialties or [],
            priority=priority,
            max_tokens=max_tokens,
        )
        
        with self._lock:
            self.models[name] = info
        
        logger.info(f"Registered model '{name}' with role {role.value}, "
                   f"specialties: {specialties or ['general']}")
    
    def unregister_model(self, name: str) -> bool:
        """Remove a model from the coordinator."""
        with self._lock:
            if name in self.models:
                del self.models[name]
                return True
        return False
    
    def _select_model(self, request: GenerationRequest) -> List[ModelInfo]:
        """
        Select best model(s) for the request.
        
        Returns list of models in order of preference.
        """
        available = [m for m in self.models.values() if m.is_available]
        
        # Apply filters
        if request.required_models:
            available = [m for m in available if m.name in request.required_models]
        if request.excluded_models:
            available = [m for m in available if m.name not in request.excluded_models]
        
        if not available:
            logger.warning("No models available for request")
            return []
        
        # Classify the task
        task_types = self.classifier.classify(request.prompt)
        primary_task = task_types[0][0] if task_types else "general"
        
        # Score models
        scored = []
        for model in available:
            score = model.priority * 10
            
            # Specialty match bonus
            if primary_task in model.specialties:
                score += 50
            elif any(t in model.specialties for t, _ in task_types):
                score += 25
            
            # Performance bonuses
            score += model.success_rate() * 20
            score += model.avg_quality_score * 10
            
            # Penalize fallback models unless needed
            if model.role == ModelRole.FALLBACK:
                score -= 30
            
            scored.append((score, model))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [m for _, m in scored]
    
    def _call_model(
        self,
        model_info: ModelInfo,
        request: GenerationRequest,
    ) -> Tuple[Optional[str], float]:
        """
        Call a single model.
        
        Returns (response, latency) or (None, 0) on failure.
        """
        start_time = time.time()
        
        try:
            model = model_info.model
            
            # Build prompt with context
            if request.context:
                context_str = "\n".join(
                    f"{m['role'].title()}: {m['content']}"
                    for m in request.context[-5:]  # Last 5 messages
                )
                full_prompt = f"{context_str}\nUser: {request.prompt}\nAssistant:"
            else:
                full_prompt = request.prompt
            
            # Call model
            if hasattr(model, 'generate'):
                response = model.generate(
                    full_prompt,
                    max_tokens=min(request.max_tokens, model_info.max_tokens),
                    temperature=request.temperature,
                )
            elif hasattr(model, 'chat'):
                messages = request.context + [{"role": "user", "content": request.prompt}]
                response = model.chat(messages, max_tokens=request.max_tokens)
            elif callable(model):
                response = model(full_prompt)
            else:
                raise ValueError(f"Model {model_info.name} has no generate() or chat() method")
            
            latency = time.time() - start_time
            
            # Update stats
            with self._lock:
                model_info.total_calls += 1
                model_info.successful_calls += 1
                model_info.total_latency += latency
            
            return response, latency
            
        except Exception as e:
            logger.error(f"Model {model_info.name} failed: {e}")
            latency = time.time() - start_time
            
            with self._lock:
                model_info.total_calls += 1
                model_info.total_latency += latency
            
            return None, latency
    
    def generate(
        self,
        prompt: str,
        strategy: Optional[CoordinationStrategy] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate response using coordinated models.
        
        Args:
            prompt: Input prompt
            strategy: Coordination strategy (uses default if None)
            **kwargs: Additional request parameters
        
        Returns:
            GenerationResult with response and metadata
        """
        request = GenerationRequest(
            prompt=prompt,
            strategy=strategy or self.default_strategy,
            **kwargs
        )
        
        if request.strategy == CoordinationStrategy.ROUTE:
            return self._route_generate(request)
        elif request.strategy == CoordinationStrategy.ENSEMBLE:
            return self._ensemble_generate(request)
        elif request.strategy == CoordinationStrategy.CASCADE:
            return self._cascade_generate(request)
        elif request.strategy == CoordinationStrategy.PARALLEL:
            return self._parallel_generate(request)
        elif request.strategy == CoordinationStrategy.CONSENSUS:
            return self._consensus_generate(request)
        else:
            return self._route_generate(request)
    
    def _route_generate(self, request: GenerationRequest) -> GenerationResult:
        """Route to single best model."""
        models = self._select_model(request)
        
        if not models:
            return GenerationResult(
                response="No models available.",
                model_used="none",
                strategy_used=CoordinationStrategy.ROUTE,
                latency=0.0,
                confidence=0.0,
            )
        
        # Try best model
        best_model = models[0]
        response, latency = self._call_model(best_model, request)
        
        if response is None:
            # Try fallback
            for fallback in models[1:]:
                response, latency = self._call_model(fallback, request)
                if response:
                    return GenerationResult(
                        response=response,
                        model_used=fallback.name,
                        strategy_used=CoordinationStrategy.ROUTE,
                        latency=latency,
                        confidence=0.8,
                        metadata={"fallback": True},
                    )
            
            return GenerationResult(
                response="All models failed.",
                model_used="none",
                strategy_used=CoordinationStrategy.ROUTE,
                latency=latency,
                confidence=0.0,
            )
        
        return GenerationResult(
            response=response,
            model_used=best_model.name,
            strategy_used=CoordinationStrategy.ROUTE,
            latency=latency,
            confidence=1.0,
        )
    
    def _ensemble_generate(self, request: GenerationRequest) -> GenerationResult:
        """Get responses from multiple models and combine."""
        models = self._select_model(request)[:self.max_parallel]
        
        if not models:
            return GenerationResult(
                response="No models available.",
                model_used="none",
                strategy_used=CoordinationStrategy.ENSEMBLE,
                latency=0.0,
                confidence=0.0,
            )
        
        # Call all models in parallel
        start_time = time.time()
        futures = {
            self._executor.submit(self._call_model, m, request): m
            for m in models
        }
        
        responses = []
        alternatives = []
        
        for future in as_completed(futures, timeout=request.timeout):
            model = futures[future]
            try:
                response, _ = future.result()
                if response:
                    responses.append((response, model.avg_quality_score))
                    alternatives.append((model.name, response))
            except Exception as e:
                logger.error(f"Ensemble call failed: {e}")
        
        total_latency = time.time() - start_time
        
        if not responses:
            return GenerationResult(
                response="All models failed.",
                model_used="none",
                strategy_used=CoordinationStrategy.ENSEMBLE,
                latency=total_latency,
                confidence=0.0,
            )
        
        # Combine responses
        combined = self.combiner.weighted_combine(responses)
        
        return GenerationResult(
            response=combined,
            model_used=f"ensemble({len(responses)})",
            strategy_used=CoordinationStrategy.ENSEMBLE,
            latency=total_latency,
            confidence=len(responses) / len(models),
            alternatives=alternatives,
        )
    
    def _cascade_generate(self, request: GenerationRequest) -> GenerationResult:
        """Try models in order until one succeeds."""
        models = self._select_model(request)
        
        total_latency = 0.0
        attempts = []
        
        for model in models:
            response, latency = self._call_model(model, request)
            total_latency += latency
            
            if response:
                return GenerationResult(
                    response=response,
                    model_used=model.name,
                    strategy_used=CoordinationStrategy.CASCADE,
                    latency=total_latency,
                    confidence=1.0,
                    metadata={"attempts": len(attempts) + 1},
                )
            
            attempts.append(model.name)
        
        return GenerationResult(
            response="All models failed.",
            model_used="none",
            strategy_used=CoordinationStrategy.CASCADE,
            latency=total_latency,
            confidence=0.0,
            metadata={"failed_models": attempts},
        )
    
    def _parallel_generate(self, request: GenerationRequest) -> GenerationResult:
        """Run all models in parallel, return first success."""
        models = self._select_model(request)[:self.max_parallel]
        
        start_time = time.time()
        futures = {
            self._executor.submit(self._call_model, m, request): m
            for m in models
        }
        
        # Return first successful response
        for future in as_completed(futures, timeout=request.timeout):
            model = futures[future]
            try:
                response, latency = future.result()
                if response:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    
                    return GenerationResult(
                        response=response,
                        model_used=model.name,
                        strategy_used=CoordinationStrategy.PARALLEL,
                        latency=time.time() - start_time,
                        confidence=1.0,
                    )
            except Exception:
                pass  # Intentionally silent
        
        return GenerationResult(
            response="No model responded in time.",
            model_used="none",
            strategy_used=CoordinationStrategy.PARALLEL,
            latency=time.time() - start_time,
            confidence=0.0,
        )
    
    def _consensus_generate(self, request: GenerationRequest) -> GenerationResult:
        """Get responses and have models vote on best one."""
        # First, get responses from multiple models
        ensemble_result = self._ensemble_generate(request)
        
        if not ensemble_result.alternatives or len(ensemble_result.alternatives) < 2:
            return ensemble_result
        
        # Find critic model if available
        critic = next(
            (m for m in self.models.values() if m.role == ModelRole.CRITIC),
            None
        )
        
        if critic:
            # Have critic evaluate responses
            eval_prompt = f"""Evaluate these responses to: "{request.prompt}"

Responses:
{chr(10).join(f'{i+1}. {alt[1][:200]}...' for i, alt in enumerate(ensemble_result.alternatives))}

Which response is best? Reply with just the number."""
            
            eval_result, _ = self._call_model(critic, GenerationRequest(prompt=eval_prompt))
            
            if eval_result:
                try:
                    best_idx = int(eval_result.strip()) - 1
                    if 0 <= best_idx < len(ensemble_result.alternatives):
                        best_model, best_response = ensemble_result.alternatives[best_idx]
                        
                        return GenerationResult(
                            response=best_response,
                            model_used=best_model,
                            strategy_used=CoordinationStrategy.CONSENSUS,
                            latency=ensemble_result.latency,
                            confidence=0.9,
                            alternatives=ensemble_result.alternatives,
                        )
                except ValueError:
                    pass  # Intentionally silent
        
        # Fall back to majority vote
        responses = [alt[1] for alt in ensemble_result.alternatives]
        best = self.combiner.majority_vote(responses)
        
        return GenerationResult(
            response=best,
            model_used="consensus",
            strategy_used=CoordinationStrategy.CONSENSUS,
            latency=ensemble_result.latency,
            confidence=0.7,
            alternatives=ensemble_result.alternatives,
        )
    
    def update_quality_score(self, model_name: str, score: float) -> None:
        """
        Update a model's quality score based on feedback.
        
        Args:
            model_name: Name of the model
            score: Quality score (0-1)
        """
        if model_name not in self.models:
            return
        
        model = self.models[model_name]
        # Exponential moving average
        alpha = 0.1
        model.avg_quality_score = alpha * score + (1 - alpha) * model.avg_quality_score
    
    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all registered models."""
        return {
            name: {
                **info.to_dict(),
                "success_rate": info.success_rate(),
                "avg_latency": info.avg_latency(),
            }
            for name, info in self.models.items()
        }
    
    def set_model_availability(self, model_name: str, available: bool) -> None:
        """Set whether a model is available for use."""
        if model_name in self.models:
            self.models[model_name].is_available = available


# Convenience function
def create_coordinator() -> ModelCoordinator:
    """Create a default model coordinator."""
    return ModelCoordinator()


# Global singleton
_coordinator: Optional[ModelCoordinator] = None

def get_coordinator() -> ModelCoordinator:
    """Get global model coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = ModelCoordinator()
    return _coordinator


# Export public API
__all__ = [
    'ModelCoordinator',
    'ModelInfo',
    'ModelRole',
    'CoordinationStrategy',
    'GenerationRequest',
    'GenerationResult',
    'TaskClassifier',
    'EnsembleCombiner',
    'create_coordinator',
    'get_coordinator',
]
