"""
API Training Enhancements for Enigma AI Engine

Additional features for the API training system:
- API key rotation on rate limit/error
- Cost estimation before generating data
- Training data quality scoring
- Deduplication utilities

Usage:
    from enigma_engine.core.api_training_utils import (
        APIKeyRotator,
        CostEstimator,
        QualityScorer,
        deduplicate_training_data
    )
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# API Key Rotation
# =============================================================================

@dataclass
class APIKeyStatus:
    """Status of an API key."""
    key: str
    provider: str
    is_active: bool = True
    rate_limited_until: float = 0.0  # Unix timestamp
    error_count: int = 0
    last_error: Optional[str] = None
    total_calls: int = 0
    total_tokens: int = 0


class APIKeyRotator:
    """
    Manages multiple API keys with automatic rotation on rate limits.
    
    Usage:
        rotator = APIKeyRotator()
        rotator.add_key("openai", "sk-key1")
        rotator.add_key("openai", "sk-key2")
        
        # Get next available key (rotates on rate limit)
        key = rotator.get_key("openai")
        
        # Mark key as rate limited
        rotator.mark_rate_limited("openai", "sk-key1", retry_after=60)
    """
    
    def __init__(self):
        self._keys: Dict[str, List[APIKeyStatus]] = {}
        self._current_index: Dict[str, int] = {}
    
    def add_key(self, provider: str, api_key: str) -> None:
        """Add an API key for a provider."""
        if provider not in self._keys:
            self._keys[provider] = []
            self._current_index[provider] = 0
        
        # Check if key already exists
        for status in self._keys[provider]:
            if status.key == api_key:
                return  # Already added
        
        self._keys[provider].append(APIKeyStatus(
            key=api_key,
            provider=provider
        ))
        logger.info(f"Added API key for {provider} (total: {len(self._keys[provider])})")
    
    def get_key(self, provider: str) -> Optional[str]:
        """
        Get the next available API key for a provider.
        Automatically skips rate-limited keys.
        
        Returns:
            API key or None if all keys are unavailable
        """
        if provider not in self._keys or not self._keys[provider]:
            return None
        
        keys = self._keys[provider]
        current_time = time.time()
        
        # Try each key starting from current index
        for i in range(len(keys)):
            idx = (self._current_index[provider] + i) % len(keys)
            status = keys[idx]
            
            # Skip if rate limited
            if status.rate_limited_until > current_time:
                continue
            
            # Skip if too many errors
            if status.error_count >= 5 and not status.is_active:
                continue
            
            # Found a valid key
            self._current_index[provider] = (idx + 1) % len(keys)
            status.total_calls += 1
            return status.key
        
        logger.warning(f"All API keys for {provider} are rate limited or errored")
        return None
    
    def mark_rate_limited(
        self, 
        provider: str, 
        api_key: str, 
        retry_after: int = 60
    ) -> None:
        """Mark a key as rate limited."""
        if provider not in self._keys:
            return
        
        for status in self._keys[provider]:
            if status.key == api_key:
                status.rate_limited_until = time.time() + retry_after
                logger.info(f"API key for {provider} rate limited for {retry_after}s")
                break
    
    def mark_error(self, provider: str, api_key: str, error: str) -> None:
        """Mark a key as having an error."""
        if provider not in self._keys:
            return
        
        for status in self._keys[provider]:
            if status.key == api_key:
                status.error_count += 1
                status.last_error = error
                if status.error_count >= 5:
                    status.is_active = False
                    logger.warning(f"API key for {provider} disabled after 5 errors")
                break
    
    def reset_errors(self, provider: str, api_key: str) -> None:
        """Reset error count for a key after successful use."""
        if provider not in self._keys:
            return
        
        for status in self._keys[provider]:
            if status.key == api_key:
                status.error_count = 0
                status.is_active = True
                break
    
    def get_status(self, provider: str) -> List[Dict[str, Any]]:
        """Get status of all keys for a provider."""
        if provider not in self._keys:
            return []
        
        current_time = time.time()
        return [
            {
                "key_preview": s.key[:8] + "..." if len(s.key) > 8 else s.key,
                "is_active": s.is_active,
                "is_rate_limited": s.rate_limited_until > current_time,
                "rate_limited_remaining": max(0, int(s.rate_limited_until - current_time)),
                "error_count": s.error_count,
                "total_calls": s.total_calls
            }
            for s in self._keys[provider]
        ]


# =============================================================================
# Cost Estimation
# =============================================================================

# Pricing per 1M tokens (input/output) as of 2024
API_PRICING = {
    "openai": {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    },
    "anthropic": {
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }
}


@dataclass
class CostEstimate:
    """Cost estimation result."""
    provider: str
    model: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    
    def __str__(self) -> str:
        return (
            f"Estimated cost for {self.provider}/{self.model}:\n"
            f"  Input: ~{self.estimated_input_tokens:,} tokens (${self.input_cost:.4f})\n"
            f"  Output: ~{self.estimated_output_tokens:,} tokens (${self.output_cost:.4f})\n"
            f"  Total: ${self.total_cost:.4f} {self.currency}"
        )


class CostEstimator:
    """
    Estimates API costs before generating training data.
    
    Usage:
        estimator = CostEstimator()
        estimate = estimator.estimate(
            provider="openai",
            model="gpt-4o",
            num_examples=100,
            avg_input_tokens=500,
            avg_output_tokens=200
        )
        print(estimate)  # Shows cost breakdown
    """
    
    def __init__(self, custom_pricing: Optional[Dict] = None):
        self.pricing = {**API_PRICING}
        if custom_pricing:
            for provider, models in custom_pricing.items():
                if provider not in self.pricing:
                    self.pricing[provider] = {}
                self.pricing[provider].update(models)
    
    def estimate(
        self,
        provider: str,
        model: str,
        num_examples: int,
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 300,
    ) -> CostEstimate:
        """
        Estimate the cost of generating training data.
        
        Args:
            provider: API provider (openai, anthropic)
            model: Model name
            num_examples: Number of examples to generate
            avg_input_tokens: Average tokens per input prompt
            avg_output_tokens: Average tokens per output response
            
        Returns:
            CostEstimate with breakdown
        """
        # Get pricing
        if provider not in self.pricing:
            logger.warning(f"Unknown provider {provider}, using estimates")
            input_price = 5.0  # Default assumption
            output_price = 15.0
        else:
            models = self.pricing[provider]
            # Find matching model
            model_lower = model.lower()
            pricing = None
            for m, p in models.items():
                if m.lower() in model_lower or model_lower in m.lower():
                    pricing = p
                    break
            
            if not pricing:
                logger.warning(f"Unknown model {model}, using most expensive pricing")
                pricing = max(models.values(), key=lambda p: p["input"])
            
            input_price = pricing["input"]
            output_price = pricing["output"]
        
        # Calculate totals
        total_input = num_examples * avg_input_tokens
        total_output = num_examples * avg_output_tokens
        
        # Cost per 1M tokens
        input_cost = (total_input / 1_000_000) * input_price
        output_cost = (total_output / 1_000_000) * output_price
        
        return CostEstimate(
            provider=provider,
            model=model,
            estimated_input_tokens=total_input,
            estimated_output_tokens=total_output,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost
        )
    
    def estimate_batch(
        self,
        provider: str,
        model: str,
        tasks: List[str],
        examples_per_task: int,
    ) -> CostEstimate:
        """
        Estimate cost for multiple training tasks.
        
        Args:
            provider: API provider
            model: Model name
            tasks: List of task names
            examples_per_task: Examples to generate per task
        """
        # Task-specific token estimates
        TOKEN_ESTIMATES = {
            "chat": (400, 200),
            "code": (600, 400),
            "vision": (500, 250),
            "avatar": (350, 150),
            "image_gen": (300, 150),
            "audio_gen": (300, 150),
            "video_gen": (400, 200),
            "3d_gen": (350, 200),
            "game": (500, 300),
            "robot": (450, 250),
            "math": (400, 300),
            "router": (300, 100),
        }
        
        total_input = 0
        total_output = 0
        
        for task in tasks:
            input_est, output_est = TOKEN_ESTIMATES.get(task, (400, 200))
            total_input += examples_per_task * input_est
            total_output += examples_per_task * output_est
        
        return self.estimate(
            provider, model, 1,
            avg_input_tokens=total_input,
            avg_output_tokens=total_output
        )


# =============================================================================
# Training Data Quality Scoring
# =============================================================================

@dataclass
class QualityScore:
    """Quality assessment for a training example."""
    score: float  # 0-1
    issues: List[str]
    metrics: Dict[str, float]


class QualityScorer:
    """
    Scores training data quality to filter low-quality examples.
    
    Usage:
        scorer = QualityScorer()
        score = scorer.score_example(input_text, output_text)
        if score.score < 0.5:
            print(f"Low quality: {score.issues}")
    """
    
    def __init__(self, min_length: int = 10, max_length: int = 10000):
        self.min_length = min_length
        self.max_length = max_length
    
    def score_example(self, input_text: str, output_text: str) -> QualityScore:
        """
        Score a single input/output training example.
        
        Returns:
            QualityScore with 0-1 score and identified issues
        """
        issues = []
        metrics = {}
        
        # Length checks
        if len(output_text) < self.min_length:
            issues.append("Output too short")
            metrics["output_length"] = 0.0
        elif len(output_text) > self.max_length:
            issues.append("Output too long")
            metrics["output_length"] = 0.5
        else:
            metrics["output_length"] = 1.0
        
        # Repetition check
        words = output_text.lower().split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            metrics["uniqueness"] = unique_ratio
            if unique_ratio < 0.3:
                issues.append("Too much repetition")
        else:
            metrics["uniqueness"] = 1.0
        
        # Format quality
        if output_text.strip() == input_text.strip():
            issues.append("Output is identical to input")
            metrics["format"] = 0.0
        elif not output_text.strip():
            issues.append("Empty output")
            metrics["format"] = 0.0
        else:
            metrics["format"] = 1.0
        
        # Coherence check (basic)
        if output_text.count("...") > 5:
            issues.append("Too many ellipses")
            metrics["coherence"] = 0.5
        elif output_text.count("  ") > 3:
            issues.append("Excessive whitespace")
            metrics["coherence"] = 0.7
        else:
            metrics["coherence"] = 1.0
        
        # Calculate overall score
        score = sum(metrics.values()) / len(metrics) if metrics else 0.0
        
        return QualityScore(score=score, issues=issues, metrics=metrics)
    
    def filter_dataset(
        self, 
        data: List[Tuple[str, str]], 
        min_score: float = 0.5
    ) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
        """
        Filter a dataset by quality score.
        
        Args:
            data: List of (input, output) tuples
            min_score: Minimum quality score to keep
            
        Returns:
            Tuple of (filtered_data, stats)
        """
        filtered = []
        stats = {"total": len(data), "kept": 0, "removed": 0, "issues": {}}
        
        for inp, out in data:
            score = self.score_example(inp, out)
            
            if score.score >= min_score:
                filtered.append((inp, out))
                stats["kept"] += 1
            else:
                stats["removed"] += 1
                for issue in score.issues:
                    stats["issues"][issue] = stats["issues"].get(issue, 0) + 1
        
        return filtered, stats


# =============================================================================
# Training Data Deduplication
# =============================================================================

def deduplicate_training_data(
    data: List[str],
    similarity_threshold: float = 0.9
) -> Tuple[List[str], int]:
    """
    Remove duplicate and near-duplicate training examples.
    
    Args:
        data: List of training examples
        similarity_threshold: How similar two items must be to be considered duplicates
        
    Returns:
        Tuple of (deduplicated_data, num_removed)
    """
    if not data:
        return [], 0
    
    # Use hash-based exact dedup first
    seen_hashes = set()
    unique = []
    
    for item in data:
        # Normalize for comparison
        normalized = ' '.join(item.lower().split())
        item_hash = hashlib.md5(normalized.encode()).hexdigest()
        
        if item_hash not in seen_hashes:
            seen_hashes.add(item_hash)
            unique.append(item)
    
    removed = len(data) - len(unique)
    logger.info(f"Deduplicated training data: {len(data)} -> {len(unique)} ({removed} removed)")
    
    return unique, removed


def deduplicate_qa_pairs(
    data: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], int]:
    """
    Remove duplicate Q&A pairs from training data.
    
    Args:
        data: List of (question, answer) tuples
        
    Returns:
        Tuple of (deduplicated_data, num_removed)
    """
    seen = set()
    unique = []
    
    for q, a in data:
        # Hash both question and answer
        key = hashlib.md5(f"{q.lower().strip()}||{a.lower().strip()}".encode()).hexdigest()
        
        if key not in seen:
            seen.add(key)
            unique.append((q, a))
    
    removed = len(data) - len(unique)
    return unique, removed


# =============================================================================
# Global Instances
# =============================================================================

_key_rotator: Optional[APIKeyRotator] = None
_cost_estimator: Optional[CostEstimator] = None


def get_key_rotator() -> APIKeyRotator:
    """Get or create global API key rotator."""
    global _key_rotator
    if _key_rotator is None:
        _key_rotator = APIKeyRotator()
    return _key_rotator


def get_cost_estimator() -> CostEstimator:
    """Get or create global cost estimator."""
    global _cost_estimator
    if _cost_estimator is None:
        _cost_estimator = CostEstimator()
    return _cost_estimator
