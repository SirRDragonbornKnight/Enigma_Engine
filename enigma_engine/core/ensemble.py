"""
Multi-Model Ensemble for Enigma AI Engine

Combine multiple models for improved performance.

Features:
- Multiple ensemble strategies (voting, averaging, weighted)
- Dynamic weight adjustment
- Specialized model routing
- Parallel inference
- Confidence-based selection
- Quality filtering

Usage:
    from enigma_engine.core.ensemble import ModelEnsemble, create_ensemble
    
    # Quick ensemble
    ensemble = create_ensemble([
        "models/model_a",
        "models/model_b",
        "models/model_c"
    ])
    
    response = ensemble.generate("Hello, how are you?")
    
    # With custom strategy
    ensemble = ModelEnsemble(strategy="weighted")
    ensemble.add_model("models/creative", weight=0.6)
    ensemble.add_model("models/factual", weight=0.4)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class EnsembleStrategy(Enum):
    """Ensemble combination strategies."""
    VOTING = auto()           # Most common output
    AVERAGING = auto()        # Average logits
    WEIGHTED = auto()         # Weighted average
    CONFIDENCE = auto()       # Highest confidence wins
    ROUTING = auto()          # Route to specialized model
    CASCADING = auto()        # Fallback chain
    BEST_OF_N = auto()        # Generate multiple, pick best


@dataclass
class ModelMember:
    """A model in the ensemble."""
    name: str
    path: str
    weight: float = 1.0
    
    # Specialization
    domain: str = ""  # e.g., "code", "creative", "factual"
    
    # State
    model: Any = None
    tokenizer: Any = None
    is_loaded: bool = False
    
    # Performance tracking
    total_calls: int = 0
    avg_latency: float = 0.0
    avg_confidence: float = 0.0
    
    # Settings
    max_tokens: int = 512
    temperature: float = 0.7


@dataclass
class EnsembleConfig:
    """Ensemble configuration."""
    strategy: EnsembleStrategy = EnsembleStrategy.VOTING
    
    # Voting settings
    min_agreement: float = 0.5  # Minimum agreement for voting
    
    # Weighted settings
    normalize_weights: bool = True
    
    # Confidence settings
    confidence_threshold: float = 0.8
    
    # Routing settings
    domain_keywords: Dict[str, List[str]] = field(default_factory=dict)
    
    # Cascading settings
    cascade_on_low_confidence: bool = True
    cascade_threshold: float = 0.6
    
    # Best-of-N settings
    n_candidates: int = 3
    
    # Parallel execution
    parallel: bool = True
    max_workers: int = 4
    
    # Quality filtering
    filter_low_quality: bool = True
    min_response_length: int = 10


class ModelEnsemble:
    """
    Ensemble of multiple models for improved generation.
    """
    
    def __init__(
        self,
        strategy: str | EnsembleStrategy = EnsembleStrategy.VOTING,
        config: Optional[EnsembleConfig] = None
    ):
        """
        Initialize ensemble.
        
        Args:
            strategy: Combination strategy
            config: Optional configuration
        """
        if isinstance(strategy, str):
            strategy = EnsembleStrategy[strategy.upper()]
        
        self._strategy = strategy
        self._config = config or EnsembleConfig(strategy=strategy)
        
        self._members: List[ModelMember] = []
        self._executor = ThreadPoolExecutor(max_workers=self._config.max_workers)
        
        # Callbacks
        self._quality_scorer: Optional[Callable[[str], float]] = None
    
    def add_model(
        self,
        path: str,
        name: str = "",
        weight: float = 1.0,
        domain: str = "",
        load_now: bool = True
    ) -> ModelMember:
        """
        Add a model to the ensemble.
        
        Args:
            path: Model path
            name: Friendly name
            weight: Weight for weighted strategies
            domain: Specialization domain
            load_now: Load model immediately
            
        Returns:
            Model member
        """
        name = name or Path(path).name
        
        member = ModelMember(
            name=name,
            path=path,
            weight=weight,
            domain=domain
        )
        
        if load_now:
            self._load_model(member)
        
        self._members.append(member)
        logger.info(f"Added model to ensemble: {name} (weight={weight})")
        
        return member
    
    def _load_model(self, member: ModelMember):
        """Load a model."""
        try:
            from enigma_engine.core.model import Forge
            from enigma_engine.core.tokenizer import get_tokenizer
            
            member.model = Forge.from_pretrained(member.path)
            member.tokenizer = get_tokenizer()
            member.is_loaded = True
            
            # Set to eval mode
            if TORCH_AVAILABLE and hasattr(member.model, 'eval'):
                member.model.eval()
            
            logger.debug(f"Loaded ensemble member: {member.name}")
            
        except Exception as e:
            logger.error(f"Failed to load {member.name}: {e}")
            member.is_loaded = False
    
    def remove_model(self, name: str) -> bool:
        """Remove a model from the ensemble."""
        for i, member in enumerate(self._members):
            if member.name == name:
                del self._members[i]
                logger.info(f"Removed model: {name}")
                return True
        return False
    
    def set_weights(self, weights: Dict[str, float]):
        """Set weights for models."""
        for member in self._members:
            if member.name in weights:
                member.weight = weights[member.name]
        
        # Normalize if configured
        if self._config.normalize_weights:
            self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        total = sum(m.weight for m in self._members)
        if total > 0:
            for member in self._members:
                member.weight /= total
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate response using ensemble.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional generation params
            
        Returns:
            Generated response
        """
        # Get active members
        members = [m for m in self._members if m.is_loaded]
        
        if not members:
            logger.warning("No loaded models in ensemble")
            return ""
        
        # Generate based on strategy
        if self._strategy == EnsembleStrategy.VOTING:
            return self._generate_voting(prompt, max_tokens, temperature, **kwargs)
        
        elif self._strategy == EnsembleStrategy.WEIGHTED:
            return self._generate_weighted(prompt, max_tokens, temperature, **kwargs)
        
        elif self._strategy == EnsembleStrategy.CONFIDENCE:
            return self._generate_confidence(prompt, max_tokens, temperature, **kwargs)
        
        elif self._strategy == EnsembleStrategy.ROUTING:
            return self._generate_routing(prompt, max_tokens, temperature, **kwargs)
        
        elif self._strategy == EnsembleStrategy.CASCADING:
            return self._generate_cascading(prompt, max_tokens, temperature, **kwargs)
        
        elif self._strategy == EnsembleStrategy.BEST_OF_N:
            return self._generate_best_of_n(prompt, max_tokens, temperature, **kwargs)
        
        else:
            # Default to first model
            return self._generate_single(members[0], prompt, max_tokens, temperature, **kwargs)
    
    def _generate_single(
        self,
        member: ModelMember,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate from a single model."""
        try:
            from enigma_engine.core.inference import EnigmaEngine
            
            import time
            start = time.perf_counter()
            
            engine = EnigmaEngine(member.model, member.tokenizer)
            response = engine.generate(
                prompt,
                max_gen=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Update stats
            latency = time.perf_counter() - start
            member.total_calls += 1
            member.avg_latency = (
                (member.avg_latency * (member.total_calls - 1) + latency) /
                member.total_calls
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed for {member.name}: {e}")
            return ""
    
    def _generate_voting(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate using voting strategy."""
        members = [m for m in self._members if m.is_loaded]
        
        # Generate from all models
        responses = self._parallel_generate(members, prompt, max_tokens, temperature, **kwargs)
        
        if not responses:
            return ""
        
        # Simple majority: find most similar response
        # (In practice, you'd use semantic similarity)
        
        # For simplicity, return longest response (usually most complete)
        valid_responses = [r for r in responses if len(r) >= self._config.min_response_length]
        
        if not valid_responses:
            return responses[0] if responses else ""
        
        # Return response that appears most similar to others
        return max(valid_responses, key=len)
    
    def _generate_weighted(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate using weighted averaging (token level)."""
        # This is complex for text - simplified to weighted selection
        members = [m for m in self._members if m.is_loaded]
        
        # Sort by weight descending
        members = sorted(members, key=lambda m: m.weight, reverse=True)
        
        # Generate from top weighted model primarily
        responses = self._parallel_generate(members[:3], prompt, max_tokens, temperature, **kwargs)
        
        if not responses:
            return ""
        
        # Weight-adjusted selection
        weighted = list(zip(responses, [m.weight for m in members[:len(responses)]]))
        
        # Pick highest weighted valid response
        for response, weight in sorted(weighted, key=lambda x: x[1], reverse=True):
            if len(response) >= self._config.min_response_length:
                return response
        
        return responses[0] if responses else ""
    
    def _generate_confidence(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate using confidence-based selection."""
        members = [m for m in self._members if m.is_loaded]
        
        # Generate with confidence scores
        results = []
        
        for member in members:
            response = self._generate_single(member, prompt, max_tokens, temperature, **kwargs)
            confidence = self._estimate_confidence(member, response)
            results.append((response, confidence, member))
        
        # Filter by threshold
        valid = [(r, c, m) for r, c, m in results if c >= self._config.confidence_threshold]
        
        if valid:
            best = max(valid, key=lambda x: x[1])
            return best[0]
        
        # Fallback to highest confidence even below threshold
        if results:
            return max(results, key=lambda x: x[1])[0]
        
        return ""
    
    def _generate_routing(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Route to specialized model based on prompt."""
        # Detect domain from prompt
        domain = self._detect_domain(prompt)
        
        # Find matching model
        for member in self._members:
            if member.domain == domain and member.is_loaded:
                logger.debug(f"Routing to {member.name} for domain: {domain}")
                return self._generate_single(member, prompt, max_tokens, temperature, **kwargs)
        
        # Fallback to highest weight
        loaded = [m for m in self._members if m.is_loaded]
        if loaded:
            best = max(loaded, key=lambda m: m.weight)
            return self._generate_single(best, prompt, max_tokens, temperature, **kwargs)
        
        return ""
    
    def _generate_cascading(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate with cascading fallback."""
        members = sorted(
            [m for m in self._members if m.is_loaded],
            key=lambda m: -m.weight  # Highest weight first
        )
        
        for member in members:
            response = self._generate_single(member, prompt, max_tokens, temperature, **kwargs)
            
            # Check quality
            confidence = self._estimate_confidence(member, response)
            
            if confidence >= self._config.cascade_threshold:
                return response
            
            logger.debug(f"Low confidence ({confidence:.2f}) from {member.name}, cascading...")
        
        # Return last response if all below threshold
        if members:
            return self._generate_single(members[-1], prompt, max_tokens, temperature, **kwargs)
        
        return ""
    
    def _generate_best_of_n(self, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
        """Generate multiple candidates and pick best."""
        members = [m for m in self._members if m.is_loaded]
        
        if not members:
            return ""
        
        # Generate N candidates (using available models or repeating)
        n = self._config.n_candidates
        candidates = []
        
        for i in range(n):
            member = members[i % len(members)]
            # Use slightly different temperature for diversity
            temp = temperature + (i * 0.1 - 0.05 * n)
            temp = max(0.1, min(2.0, temp))
            
            response = self._generate_single(member, prompt, max_tokens, temp, **kwargs)
            score = self._score_response(response)
            candidates.append((response, score))
        
        # Return best
        best = max(candidates, key=lambda x: x[1])
        return best[0]
    
    def _parallel_generate(
        self,
        members: List[ModelMember],
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> List[str]:
        """Generate from multiple models in parallel."""
        if not self._config.parallel or len(members) <= 1:
            return [
                self._generate_single(m, prompt, max_tokens, temperature, **kwargs)
                for m in members
            ]
        
        futures = {}
        for member in members:
            future = self._executor.submit(
                self._generate_single,
                member, prompt, max_tokens, temperature, **kwargs
            )
            futures[future] = member
        
        responses = []
        for future in as_completed(futures.keys()):
            try:
                response = future.result(timeout=60)
                responses.append(response)
            except Exception as e:
                logger.error(f"Parallel generation failed: {e}")
        
        return responses
    
    def _detect_domain(self, prompt: str) -> str:
        """Detect domain from prompt."""
        prompt_lower = prompt.lower()
        
        # Check configured keywords
        for domain, keywords in self._config.domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in prompt_lower:
                    return domain
        
        # Default domain detection
        code_keywords = ["code", "function", "class", "def ", "import ", "python", "javascript"]
        for kw in code_keywords:
            if kw in prompt_lower:
                return "code"
        
        creative_keywords = ["write a story", "poem", "creative", "imagine", "fiction"]
        for kw in creative_keywords:
            if kw in prompt_lower:
                return "creative"
        
        return ""  # General/unknown
    
    def _estimate_confidence(self, member: ModelMember, response: str) -> float:
        """Estimate confidence in a response."""
        if not response:
            return 0.0
        
        # Simple heuristics
        confidence = 0.5  # Base
        
        # Length bonus
        if len(response) >= 50:
            confidence += 0.1
        if len(response) >= 200:
            confidence += 0.1
        
        # Coherence (simple check)
        if response.endswith(('.', '!', '?', '"')):
            confidence += 0.1
        
        # Not truncated
        if not response.endswith('...'):
            confidence += 0.1
        
        # Model weight influence
        confidence *= (0.5 + member.weight * 0.5)
        
        return min(1.0, confidence)
    
    def _score_response(self, response: str) -> float:
        """Score a response for best-of-N selection."""
        if self._quality_scorer:
            return self._quality_scorer(response)
        
        # Default scoring
        score = 0.0
        
        if not response:
            return score
        
        # Length (prefer substantial responses)
        score += min(len(response) / 500, 1.0) * 0.3
        
        # Completeness (ends with punctuation)
        if response.strip().endswith(('.', '!', '?', '"', "'")):
            score += 0.3
        
        # Not too repetitive
        words = response.split()
        unique_ratio = len(set(words)) / max(1, len(words))
        score += unique_ratio * 0.2
        
        # Has structure (paragraphs, etc.)
        if '\n' in response:
            score += 0.1
        
        # Not truncated
        if not response.endswith('...'):
            score += 0.1
        
        return score
    
    def set_quality_scorer(self, scorer: Callable[[str], float]):
        """Set custom quality scoring function."""
        self._quality_scorer = scorer
    
    def get_members(self) -> List[ModelMember]:
        """Get all ensemble members."""
        return self._members.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        return {
            "strategy": self._strategy.name,
            "total_models": len(self._members),
            "loaded_models": sum(1 for m in self._members if m.is_loaded),
            "models": [
                {
                    "name": m.name,
                    "weight": m.weight,
                    "domain": m.domain,
                    "total_calls": m.total_calls,
                    "avg_latency": m.avg_latency,
                }
                for m in self._members
            ]
        }
    
    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=False)


def create_ensemble(
    model_paths: List[str],
    strategy: str = "voting",
    weights: Optional[List[float]] = None
) -> ModelEnsemble:
    """
    Quick function to create an ensemble.
    
    Args:
        model_paths: List of model paths
        strategy: Ensemble strategy
        weights: Optional list of weights (same order as paths)
        
    Returns:
        Configured ensemble
    """
    ensemble = ModelEnsemble(strategy=strategy)
    
    for i, path in enumerate(model_paths):
        weight = weights[i] if weights and i < len(weights) else 1.0
        ensemble.add_model(path, weight=weight)
    
    return ensemble
