"""
Uncertainty Estimation for Enigma AI Engine

Help the AI know when it's unsure.

Features:
- Confidence scoring
- Entropy-based uncertainty
- Monte Carlo dropout
- Ensemble disagreement
- Calibration

Usage:
    from enigma_engine.core.uncertainty import UncertaintyEstimator, get_estimator
    
    estimator = get_estimator()
    
    # Get uncertainty for prediction
    result = estimator.estimate(model, "What is quantum entanglement?")
    print(f"Confidence: {result.confidence}")
    print(f"Should abstain: {result.should_abstain}")
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class UncertaintyMethod(Enum):
    """Methods for uncertainty estimation."""
    ENTROPY = "entropy"  # Token-level entropy
    MC_DROPOUT = "mc_dropout"  # Monte Carlo dropout
    ENSEMBLE = "ensemble"  # Multiple models
    TEMPERATURE = "temperature"  # Temperature scaling
    SEMANTIC = "semantic"  # Semantic similarity of outputs
    VERBALIZED = "verbalized"  # Model's own confidence


@dataclass
class UncertaintyResult:
    """Result of uncertainty estimation."""
    confidence: float  # 0-1, higher = more confident
    uncertainty: float  # 0-1, higher = more uncertain
    
    # Details
    method: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Decision support
    abstention_threshold: float = 0.7
    
    @property
    def should_abstain(self) -> bool:
        """Should model abstain from this prediction."""
        return self.uncertainty > self.abstention_threshold
    
    @property
    def confidence_level(self) -> str:
        """Human-readable confidence level."""
        if self.confidence >= 0.9:
            return "very high"
        elif self.confidence >= 0.75:
            return "high"
        elif self.confidence >= 0.5:
            return "medium"
        elif self.confidence >= 0.25:
            return "low"
        else:
            return "very low"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "confidence_level": self.confidence_level,
            "should_abstain": self.should_abstain,
            "method": self.method,
            "details": self.details
        }


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty estimation."""
    method: UncertaintyMethod = UncertaintyMethod.ENTROPY
    
    # MC Dropout
    mc_samples: int = 10
    dropout_rate: float = 0.1
    
    # Ensemble
    num_models: int = 3
    
    # Temperature
    temperature: float = 1.0
    
    # Calibration
    calibrate: bool = True
    
    # Thresholds
    abstention_threshold: float = 0.7


class EntropyEstimator:
    """Estimate uncertainty via entropy."""
    
    def estimate_token_entropy(self, logits: Any) -> float:
        """
        Compute token-level entropy.
        
        Args:
            logits: Logits tensor
            
        Returns:
            Normalized entropy (0-1)
        """
        import torch
        import torch.nn.functional as F
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Normalize by max possible entropy
        vocab_size = logits.shape[-1]
        max_entropy = math.log(vocab_size)
        
        normalized = entropy / max_entropy
        
        return normalized.mean().item()
    
    def estimate_sequence_entropy(self, logits_sequence: List[Any]) -> float:
        """Compute average entropy over sequence."""
        if not logits_sequence:
            return 0.5
        
        entropies = [self.estimate_token_entropy(l) for l in logits_sequence]
        return sum(entropies) / len(entropies)


class MCDropoutEstimator:
    """Monte Carlo dropout for uncertainty."""
    
    def __init__(self, num_samples: int = 10):
        self._num_samples = num_samples
    
    def enable_dropout(self, model: Any):
        """Enable dropout during inference."""
        for module in model.modules():
            if hasattr(module, 'training'):
                if 'dropout' in module.__class__.__name__.lower():
                    module.train()
    
    def disable_dropout(self, model: Any):
        """Disable dropout."""
        model.eval()
    
    def estimate(
        self,
        model: Any,
        input_ids: Any,
        tokenizer: Optional[Any] = None
    ) -> Tuple[float, List[str]]:
        """
        Estimate uncertainty via MC dropout.
        
        Returns:
            (uncertainty, generated_samples)
        """
        import torch
        
        samples = []
        
        for _ in range(self._num_samples):
            # Enable dropout
            self.enable_dropout(model)
            
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=True
                )
            
            if tokenizer:
                text = tokenizer.decode(output[0], skip_special_tokens=True)
                samples.append(text)
        
        # Disable dropout
        self.disable_dropout(model)
        
        # Compute variance in outputs
        if samples:
            unique_samples = set(samples)
            diversity = len(unique_samples) / len(samples)
            uncertainty = diversity  # More diverse = more uncertain
        else:
            uncertainty = 0.5
        
        return uncertainty, samples


class EnsembleEstimator:
    """Ensemble-based uncertainty via model disagreement."""
    
    def __init__(self, models: List[Any]):
        self._models = models
    
    def estimate(
        self,
        input_ids: Any,
        tokenizer: Optional[Any] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Estimate uncertainty via ensemble disagreement.
        
        Returns:
            (uncertainty, details)
        """
        import torch
        
        outputs = []
        
        for model in self._models:
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=50)
            
            if tokenizer:
                text = tokenizer.decode(output[0], skip_special_tokens=True)
                outputs.append(text)
        
        # Compute disagreement
        if outputs:
            unique_outputs = set(outputs)
            disagreement = (len(unique_outputs) - 1) / max(1, len(outputs) - 1)
        else:
            disagreement = 0.5
        
        return disagreement, {
            "outputs": outputs,
            "unique_count": len(set(outputs))
        }


class VerbalizedConfidence:
    """Extract confidence from model's own statements."""
    
    CONFIDENCE_PATTERNS = {
        "high": [
            r'i am (?:very )?(?:confident|certain|sure)',
            r'definitely',
            r'certainly',
            r'without doubt',
            r'clearly'
        ],
        "medium": [
            r'i (?:think|believe)',
            r'probably',
            r'likely',
            r'it seems'
        ],
        "low": [
            r'i\'m not (?:sure|certain)',
            r'might be',
            r'possibly',
            r'perhaps',
            r'unclear'
        ],
        "very_low": [
            r'i don\'t know',
            r'unsure',
            r'cannot determine',
            r'no idea',
            r'unable to'
        ]
    }
    
    CONFIDENCE_VALUES = {
        "high": 0.9,
        "medium": 0.6,
        "low": 0.3,
        "very_low": 0.1
    }
    
    def estimate(self, text: str) -> float:
        """
        Extract confidence from text.
        
        Args:
            text: Model's response
            
        Returns:
            Confidence score (0-1)
        """
        import re
        
        text_lower = text.lower()
        
        # Check patterns from low to high
        for level in ["very_low", "low", "medium", "high"]:
            for pattern in self.CONFIDENCE_PATTERNS[level]:
                if re.search(pattern, text_lower):
                    return self.CONFIDENCE_VALUES[level]
        
        # Default medium confidence
        return 0.5


class TemperatureCalibrator:
    """Calibrate confidence via temperature scaling."""
    
    def __init__(self, temperature: float = 1.0):
        self._temperature = temperature
    
    def fit(
        self,
        logits: List[Any],
        labels: List[int]
    ):
        """Fit temperature on validation data."""
        import torch
        import torch.nn.functional as F
        from torch.optim import LBFGS
        
        temp = torch.nn.Parameter(torch.ones(1) * self._temperature)
        
        def eval_loss():
            total_loss = 0
            for logit, label in zip(logits, labels):
                scaled = logit / temp
                loss = F.cross_entropy(scaled.unsqueeze(0), torch.tensor([label]))
                total_loss += loss
            return total_loss / len(logits)
        
        optimizer = LBFGS([temp], lr=0.01, max_iter=50)
        
        def closure():
            optimizer.zero_grad()
            loss = eval_loss()
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        self._temperature = temp.item()
        logger.info(f"Calibrated temperature: {self._temperature:.3f}")
    
    def calibrate(self, logits: Any) -> Any:
        """Apply temperature scaling."""
        return logits / self._temperature


class UncertaintyEstimator:
    """High-level uncertainty estimation interface."""
    
    def __init__(self, config: Optional[UncertaintyConfig] = None):
        """
        Initialize estimator.
        
        Args:
            config: Uncertainty configuration
        """
        self._config = config or UncertaintyConfig()
        
        self._entropy = EntropyEstimator()
        self._mc_dropout = MCDropoutEstimator(self._config.mc_samples)
        self._verbalized = VerbalizedConfidence()
        self._calibrator = TemperatureCalibrator(self._config.temperature)
    
    def estimate(
        self,
        model: Any,
        prompt: str,
        tokenizer: Optional[Any] = None,
        method: Optional[UncertaintyMethod] = None
    ) -> UncertaintyResult:
        """
        Estimate uncertainty for a prediction.
        
        Args:
            model: The model
            prompt: Input prompt
            tokenizer: Tokenizer
            method: Override method
            
        Returns:
            UncertaintyResult
        """
        method = method or self._config.method
        
        if method == UncertaintyMethod.ENTROPY:
            return self._estimate_entropy(model, prompt, tokenizer)
        elif method == UncertaintyMethod.MC_DROPOUT:
            return self._estimate_mc_dropout(model, prompt, tokenizer)
        elif method == UncertaintyMethod.VERBALIZED:
            return self._estimate_verbalized(model, prompt, tokenizer)
        else:
            return self._estimate_entropy(model, prompt, tokenizer)
    
    def _estimate_entropy(
        self,
        model: Any,
        prompt: str,
        tokenizer: Optional[Any]
    ) -> UncertaintyResult:
        """Entropy-based uncertainty."""
        import torch
        
        # Tokenize
        if tokenizer:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
        else:
            return UncertaintyResult(
                confidence=0.5,
                uncertainty=0.5,
                method="entropy"
            )
        
        # Get logits
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # Compute entropy
        entropy = self._entropy.estimate_token_entropy(logits[:, -1, :])
        
        # Convert to confidence
        confidence = 1 - entropy
        
        return UncertaintyResult(
            confidence=confidence,
            uncertainty=entropy,
            method="entropy",
            details={"raw_entropy": entropy},
            abstention_threshold=self._config.abstention_threshold
        )
    
    def _estimate_mc_dropout(
        self,
        model: Any,
        prompt: str,
        tokenizer: Optional[Any]
    ) -> UncertaintyResult:
        """MC dropout uncertainty."""
        import torch
        
        if tokenizer:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
        else:
            return UncertaintyResult(
                confidence=0.5,
                uncertainty=0.5,
                method="mc_dropout"
            )
        
        uncertainty, samples = self._mc_dropout.estimate(
            model, input_ids, tokenizer
        )
        
        return UncertaintyResult(
            confidence=1 - uncertainty,
            uncertainty=uncertainty,
            method="mc_dropout",
            details={
                "num_samples": len(samples),
                "unique_samples": len(set(samples))
            },
            abstention_threshold=self._config.abstention_threshold
        )
    
    def _estimate_verbalized(
        self,
        model: Any,
        prompt: str,
        tokenizer: Optional[Any]
    ) -> UncertaintyResult:
        """Verbalized confidence estimation."""
        # Generate response
        if hasattr(model, 'generate') and tokenizer:
            import torch
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=100)
            
            response = tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            response = ""
        
        confidence = self._verbalized.estimate(response)
        
        return UncertaintyResult(
            confidence=confidence,
            uncertainty=1 - confidence,
            method="verbalized",
            details={"response_preview": response[:100]},
            abstention_threshold=self._config.abstention_threshold
        )
    
    def add_abstention(
        self,
        response: str,
        uncertainty: UncertaintyResult
    ) -> str:
        """
        Add abstention notice if uncertain.
        
        Args:
            response: Model's response
            uncertainty: Uncertainty result
            
        Returns:
            Response with possible abstention notice
        """
        if uncertainty.should_abstain:
            notice = (
                f"\n\n[Note: I'm not very confident about this answer "
                f"(confidence: {uncertainty.confidence_level}). "
                f"Please verify independently.]"
            )
            return response + notice
        
        return response


# Global instance
_estimator: Optional[UncertaintyEstimator] = None


def get_estimator(
    config: Optional[UncertaintyConfig] = None
) -> UncertaintyEstimator:
    """Get or create global uncertainty estimator."""
    global _estimator
    if _estimator is None or config is not None:
        _estimator = UncertaintyEstimator(config)
    return _estimator
