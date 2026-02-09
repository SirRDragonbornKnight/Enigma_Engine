"""
Speculative Decoding for Enigma AI Engine

Speed up inference using small draft model to propose tokens.

Features:
- Draft and verify paradigm
- Configurable acceptance threshold
- Multi-token speculation
- Adaptive speculation length
- Fallback to standard decoding

Usage:
    from enigma_engine.core.speculative import SpeculativeDecoder, get_speculative_decoder
    
    decoder = get_speculative_decoder(
        draft_model=small_model,
        target_model=large_model
    )
    
    # Generate with speculation
    output = decoder.generate("Hello")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    # Speculation length
    num_speculative_tokens: int = 4  # Tokens to draft at once
    min_speculation: int = 1
    max_speculation: int = 8
    
    # Acceptance
    acceptance_threshold: float = 0.5  # Min probability to accept
    temperature: float = 1.0
    
    # Adaptive
    adaptive_length: bool = True  # Adjust speculation based on acceptance rate
    target_acceptance_rate: float = 0.8
    
    # Performance
    max_new_tokens: int = 256
    
    # Debug
    verbose: bool = False


@dataclass
class SpeculationStats:
    """Statistics for speculative decoding."""
    total_tokens_generated: int = 0
    tokens_accepted: int = 0
    tokens_rejected: int = 0
    speculation_rounds: int = 0
    
    draft_time: float = 0.0
    verify_time: float = 0.0
    total_time: float = 0.0
    
    @property
    def acceptance_rate(self) -> float:
        """Get acceptance rate."""
        total = self.tokens_accepted + self.tokens_rejected
        return self.tokens_accepted / total if total > 0 else 0.0
    
    @property
    def speedup_estimate(self) -> float:
        """Estimate speedup from speculation."""
        if self.total_tokens_generated == 0:
            return 1.0
        # Rough estimate: tokens/round indicates effectiveness
        if self.speculation_rounds == 0:
            return 1.0
        tokens_per_round = self.total_tokens_generated / self.speculation_rounds
        return min(5.0, max(1.0, tokens_per_round))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens_generated,
            "accepted": self.tokens_accepted,
            "rejected": self.tokens_rejected,
            "rounds": self.speculation_rounds,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "speedup_estimate": round(self.speedup_estimate, 2),
            "draft_time_ms": round(self.draft_time * 1000, 2),
            "verify_time_ms": round(self.verify_time * 1000, 2)
        }


class SpeculativeDecoder:
    """Speculative decoding implementation."""
    
    def __init__(
        self,
        draft_model: Any,
        target_model: Any,
        config: Optional[SpeculativeConfig] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize speculative decoder.
        
        Args:
            draft_model: Small, fast model for drafting
            target_model: Large, accurate model for verification
            config: Speculative decoding configuration
            tokenizer: Tokenizer for text conversion
        """
        self._draft = draft_model
        self._target = target_model
        self._config = config or SpeculativeConfig()
        self._tokenizer = tokenizer
        
        self._stats = SpeculationStats()
        self._current_speculation_length = self._config.num_speculative_tokens
    
    @property
    def stats(self) -> SpeculationStats:
        """Get current statistics."""
        return self._stats
    
    def reset_stats(self):
        """Reset statistics."""
        self._stats = SpeculationStats()
    
    def _get_logits(self, model: Any, input_ids: Any) -> Any:
        """Get logits from model."""
        try:
            import torch
            
            with torch.no_grad():
                outputs = model(input_ids)
                if hasattr(outputs, 'logits'):
                    return outputs.logits
                return outputs[0]
        except Exception as e:
            logger.error(f"Failed to get logits: {e}")
            raise
    
    def _sample_token(
        self,
        logits: Any,
        temperature: float = 1.0
    ) -> Tuple[int, float]:
        """Sample token from logits, return token and probability."""
        try:
            import torch
            import torch.nn.functional as F
            
            # Get last token logits
            if len(logits.shape) > 2:
                logits = logits[:, -1, :]
            elif len(logits.shape) == 2:
                logits = logits[-1, :]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample
            token = torch.multinomial(probs, 1).item()
            prob = probs[0, token].item() if len(probs.shape) > 1 else probs[token].item()
            
            return token, prob
            
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            return 0, 0.0
    
    def _draft_tokens(
        self,
        input_ids: Any,
        num_tokens: int
    ) -> Tuple[List[int], List[float]]:
        """
        Draft tokens using draft model.
        
        Returns:
            Tuple of (token_ids, probabilities)
        """
        import torch
        
        draft_tokens = []
        draft_probs = []
        
        current_ids = input_ids.clone()
        
        start_time = time.time()
        
        for _ in range(num_tokens):
            logits = self._get_logits(self._draft, current_ids)
            token, prob = self._sample_token(logits, self._config.temperature)
            
            draft_tokens.append(token)
            draft_probs.append(prob)
            
            # Append for next iteration
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[token]], device=current_ids.device)
            ], dim=-1)
        
        self._stats.draft_time += time.time() - start_time
        
        return draft_tokens, draft_probs
    
    def _verify_tokens(
        self,
        input_ids: Any,
        draft_tokens: List[int],
        draft_probs: List[float]
    ) -> Tuple[List[int], int]:
        """
        Verify draft tokens with target model.
        
        Returns:
            Tuple of (accepted_tokens, num_accepted)
        """
        import torch
        
        start_time = time.time()
        
        # Build full sequence with drafts
        draft_tensor = torch.tensor([draft_tokens], device=input_ids.device)
        full_seq = torch.cat([input_ids, draft_tensor], dim=-1)
        
        # Get target model logits for all positions
        target_logits = self._get_logits(self._target, full_seq)
        
        accepted_tokens = []
        num_accepted = 0
        
        # Check each draft token
        for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
            # Get target probability for this position
            pos = input_ids.shape[-1] + i - 1 if i > 0 else input_ids.shape[-1] - 1
            
            if pos >= 0 and pos < target_logits.shape[1]:
                import torch.nn.functional as F
                target_probs = F.softmax(target_logits[0, pos, :] / self._config.temperature, dim=-1)
                target_prob = target_probs[draft_token].item()
                
                # Acceptance criterion
                # Accept if target prob >= draft prob * threshold
                accept_threshold = draft_prob * self._config.acceptance_threshold
                
                if target_prob >= accept_threshold:
                    accepted_tokens.append(draft_token)
                    num_accepted += 1
                    self._stats.tokens_accepted += 1
                else:
                    # Rejection - sample from target instead
                    self._stats.tokens_rejected += 1
                    
                    # Sample correction token
                    correction_token, _ = self._sample_token(
                        target_logits[0, pos:pos+1, :],
                        self._config.temperature
                    )
                    accepted_tokens.append(correction_token)
                    break
        
        self._stats.verify_time += time.time() - start_time
        
        return accepted_tokens, num_accepted
    
    def _adjust_speculation_length(self, acceptance_rate: float):
        """Adaptively adjust speculation length."""
        if not self._config.adaptive_length:
            return
        
        if acceptance_rate > self._config.target_acceptance_rate:
            # Good acceptance, try more speculation
            self._current_speculation_length = min(
                self._current_speculation_length + 1,
                self._config.max_speculation
            )
        elif acceptance_rate < self._config.target_acceptance_rate * 0.5:
            # Poor acceptance, reduce speculation
            self._current_speculation_length = max(
                self._current_speculation_length - 1,
                self._config.min_speculation
            )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text with speculative decoding.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        import torch
        
        max_tokens = max_new_tokens or self._config.max_new_tokens
        
        # Reset stats for this generation
        self.reset_stats()
        start_time = time.time()
        
        # Tokenize prompt
        if self._tokenizer:
            input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        else:
            # Fallback: assume model has encode method
            input_ids = self._draft.encode(prompt)
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor([input_ids])
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
        
        # Move to same device as draft model
        device = next(self._draft.parameters()).device if hasattr(self._draft, 'parameters') else 'cpu'
        input_ids = input_ids.to(device)
        
        generated_tokens = []
        
        while len(generated_tokens) < max_tokens:
            self._stats.speculation_rounds += 1
            
            # Draft tokens
            draft_tokens, draft_probs = self._draft_tokens(
                input_ids,
                self._current_speculation_length
            )
            
            # Verify with target
            accepted, num_accepted = self._verify_tokens(
                input_ids,
                draft_tokens,
                draft_probs
            )
            
            # Add accepted tokens
            generated_tokens.extend(accepted)
            self._stats.total_tokens_generated += len(accepted)
            
            # Update input_ids
            accepted_tensor = torch.tensor([accepted], device=device)
            input_ids = torch.cat([input_ids, accepted_tensor], dim=-1)
            
            # Adjust speculation length
            if self._stats.speculation_rounds > 0:
                self._adjust_speculation_length(self._stats.acceptance_rate)
            
            # Check for EOS
            if self._tokenizer and hasattr(self._tokenizer, 'eos_token_id'):
                if accepted and accepted[-1] == self._tokenizer.eos_token_id:
                    break
            
            if self._config.verbose:
                logger.debug(f"Round {self._stats.speculation_rounds}: accepted {num_accepted}/{len(draft_tokens)}")
        
        self._stats.total_time = time.time() - start_time
        
        # Decode
        if self._tokenizer:
            output = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            output = str(generated_tokens)
        
        if self._config.verbose:
            logger.info(f"Speculative stats: {self._stats.to_dict()}")
        
        return output
    
    def generate_ids(
        self,
        input_ids: Any,
        max_new_tokens: Optional[int] = None
    ) -> Any:
        """
        Generate token IDs with speculative decoding.
        
        Args:
            input_ids: Input token IDs tensor
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated token IDs tensor
        """
        import torch
        
        max_tokens = max_new_tokens or self._config.max_new_tokens
        self.reset_stats()
        
        generated_tokens = []
        current_ids = input_ids
        
        while len(generated_tokens) < max_tokens:
            self._stats.speculation_rounds += 1
            
            draft_tokens, draft_probs = self._draft_tokens(
                current_ids,
                self._current_speculation_length
            )
            
            accepted, _ = self._verify_tokens(
                current_ids,
                draft_tokens,
                draft_probs
            )
            
            generated_tokens.extend(accepted)
            self._stats.total_tokens_generated += len(accepted)
            
            device = current_ids.device
            accepted_tensor = torch.tensor([accepted], device=device)
            current_ids = torch.cat([current_ids, accepted_tensor], dim=-1)
            
            if self._stats.speculation_rounds > 0:
                self._adjust_speculation_length(self._stats.acceptance_rate)
        
        return current_ids


class SimpleDraftModel:
    """Simple draft model using n-gram prediction or small LM."""
    
    def __init__(self, vocab_size: int = 50000):
        """
        Initialize simple draft model.
        
        Args:
            vocab_size: Vocabulary size
        """
        self._vocab_size = vocab_size
        self._ngrams: Dict[Tuple[int, ...], Dict[int, int]] = {}
    
    def train(self, sequences: List[List[int]], n: int = 3):
        """Train n-gram model on sequences."""
        for seq in sequences:
            for i in range(len(seq) - n):
                context = tuple(seq[i:i+n])
                next_token = seq[i+n]
                
                if context not in self._ngrams:
                    self._ngrams[context] = {}
                
                self._ngrams[context][next_token] = self._ngrams[context].get(next_token, 0) + 1
    
    def __call__(self, input_ids):
        """Generate logits for input."""
        import torch
        
        # Simple uniform distribution with boost for seen n-grams
        logits = torch.zeros(1, input_ids.shape[1], self._vocab_size)
        
        # Boost based on n-gram statistics
        if input_ids.shape[1] >= 3:
            context = tuple(input_ids[0, -3:].tolist())
            if context in self._ngrams:
                for token, count in self._ngrams[context].items():
                    if token < self._vocab_size:
                        logits[0, -1, token] = count
        
        return logits
    
    def parameters(self):
        """Dummy parameters for device detection."""
        import torch
        return iter([torch.zeros(1)])


# Global instance
_speculative_decoder: Optional[SpeculativeDecoder] = None


def get_speculative_decoder(
    draft_model: Optional[Any] = None,
    target_model: Optional[Any] = None,
    **kwargs
) -> Optional[SpeculativeDecoder]:
    """Get or create global speculative decoder."""
    global _speculative_decoder
    
    if draft_model is not None and target_model is not None:
        _speculative_decoder = SpeculativeDecoder(
            draft_model=draft_model,
            target_model=target_model,
            **kwargs
        )
    
    return _speculative_decoder
