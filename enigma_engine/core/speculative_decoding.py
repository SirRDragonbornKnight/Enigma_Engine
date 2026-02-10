"""
Speculative Decoding for enigma_engine

Speculative decoding uses a small "draft" model to generate candidate tokens,
then verifies them with the large "target" model in parallel.

Benefits:
- 2-3x speedup for autoregressive generation
- No quality loss (mathematically equivalent to target model)
- Works with any model pair

References:
- "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al.)
- "Accelerating Large Language Model Decoding with Speculative Sampling"
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    num_speculative_tokens: int = 5  # Number of tokens to speculate
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_retries: int = 3  # Max verification retries before falling back


class SpeculativeDecoder:
    """
    Speculative decoding for faster inference.
    
    Uses a small draft model to propose tokens, then verifies with
    the larger target model in a single forward pass.
    
    Usage:
        decoder = SpeculativeDecoder(
            target_model=large_model,
            draft_model=small_model,
            tokenizer=tokenizer
        )
        
        output = decoder.generate(
            prompt="Hello, how are you?",
            max_tokens=100
        )
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        tokenizer: Any,
        config: Optional[SpeculativeConfig] = None
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.config = config or SpeculativeConfig()
        
        # Put models in eval mode
        self.target_model.eval()
        self.draft_model.eval()
        
        # Statistics
        self.total_draft_tokens = 0
        self.accepted_tokens = 0
    
    @property
    def acceptance_rate(self) -> float:
        """Get the token acceptance rate."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens
    
    def reset_stats(self):
        """Reset acceptance statistics."""
        self.total_draft_tokens = 0
        self.accepted_tokens = 0
    
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_tokens: Optional[list[int]] = None
    ) -> str:
        """
        Generate text using speculative decoding.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            stop_tokens: Token IDs that stop generation
        
        Returns:
            Generated text
        """
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)])
        device = next(self.target_model.parameters()).device
        input_ids = input_ids.to(device)
        
        generated_ids = input_ids.clone()
        
        tokens_generated = 0
        while tokens_generated < max_tokens:
            # Draft phase: generate speculative tokens
            draft_ids, draft_probs = self._draft_tokens(
                generated_ids, temperature, top_p, top_k
            )
            
            # Verify phase: check draft tokens with target model
            accepted_ids, num_accepted = self._verify_tokens(
                generated_ids, draft_ids, draft_probs,
                temperature, top_p, top_k
            )
            
            # Update statistics
            self.total_draft_tokens += len(draft_ids[0])
            self.accepted_tokens += num_accepted
            
            # Append accepted tokens
            generated_ids = torch.cat([generated_ids, accepted_ids], dim=1)
            tokens_generated += accepted_ids.shape[1]
            
            # Check for stop tokens
            if stop_tokens:
                for stop_token in stop_tokens:
                    if stop_token in accepted_ids[0]:
                        # Truncate at stop token
                        stop_idx = (accepted_ids[0] == stop_token).nonzero()[0].item()
                        generated_ids = generated_ids[:, :generated_ids.shape[1] - accepted_ids.shape[1] + stop_idx + 1]
                        return self.tokenizer.decode(generated_ids[0].tolist())
        
        return self.tokenizer.decode(generated_ids[0].tolist())
    
    def _draft_tokens(
        self,
        input_ids: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate draft tokens using the small model."""
        draft_ids = []
        draft_probs = []
        
        current_ids = input_ids.clone()
        
        for _ in range(self.config.num_speculative_tokens):
            # Get draft model logits
            outputs = self.draft_model(current_ids)
            logits = outputs[:, -1, :] / temperature
            
            # Apply top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            draft_ids.append(next_token)
            draft_probs.append(probs)
            
            current_ids = torch.cat([current_ids, next_token], dim=1)
        
        draft_ids = torch.cat(draft_ids, dim=1)
        draft_probs = torch.stack(draft_probs, dim=1)
        
        return draft_ids, draft_probs
    
    def _verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> tuple[torch.Tensor, int]:
        """Verify draft tokens with target model."""
        # Concatenate input with draft tokens
        full_ids = torch.cat([input_ids, draft_ids], dim=1)
        
        # Get target model logits for all positions
        outputs = self.target_model(full_ids)
        
        # Extract logits for the draft positions
        start_pos = input_ids.shape[1] - 1
        target_logits = outputs[:, start_pos:start_pos + draft_ids.shape[1] + 1, :] / temperature
        
        # Apply sampling constraints
        if top_k > 0:
            for i in range(target_logits.shape[1]):
                indices_to_remove = target_logits[:, i, :] < torch.topk(target_logits[:, i, :], top_k)[0][..., -1, None]
                target_logits[:, i, :][indices_to_remove] = float('-inf')
        
        target_probs = F.softmax(target_logits, dim=-1)
        
        # Speculative sampling verification
        accepted_tokens = []
        num_accepted = 0
        
        for i in range(draft_ids.shape[1]):
            draft_token = draft_ids[0, i].item()
            
            # Get probabilities
            p_target = target_probs[0, i, draft_token].item()
            p_draft = draft_probs[0, i, draft_token].item()
            
            # Acceptance probability
            if p_draft > 0:
                accept_prob = min(1.0, p_target / p_draft)
            else:
                accept_prob = 1.0 if p_target > 0 else 0.0
            
            # Accept or reject
            if torch.rand(1).item() < accept_prob:
                accepted_tokens.append(draft_token)
                num_accepted += 1
            else:
                # Rejection: sample from adjusted distribution
                # p_adjusted = max(0, p_target - p_draft) / sum(max(0, p_target - p_draft))
                adjusted_probs = torch.clamp(
                    target_probs[0, i, :] - draft_probs[0, i, :] if i < draft_probs.shape[1] else target_probs[0, i, :],
                    min=0
                )
                
                if adjusted_probs.sum() > 0:
                    adjusted_probs = adjusted_probs / adjusted_probs.sum()
                    new_token = torch.multinomial(adjusted_probs.unsqueeze(0), num_samples=1)
                else:
                    new_token = torch.multinomial(target_probs[0, i, :].unsqueeze(0), num_samples=1)
                
                accepted_tokens.append(new_token.item())
                break
        
        # If all tokens accepted, sample one more from target
        if num_accepted == draft_ids.shape[1]:
            final_probs = target_probs[0, -1, :]
            final_token = torch.multinomial(final_probs.unsqueeze(0), num_samples=1)
            accepted_tokens.append(final_token.item())
        
        accepted_ids = torch.tensor([accepted_tokens], device=input_ids.device)
        
        return accepted_ids, num_accepted
    
    @torch.inference_mode()
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Streaming generation with speculative decoding.
        
        Yields tokens as they are accepted.
        """
        temperature = temperature or self.config.temperature
        
        input_ids = torch.tensor([self.tokenizer.encode(prompt)])
        device = next(self.target_model.parameters()).device
        input_ids = input_ids.to(device)
        
        generated_ids = input_ids.clone()
        tokens_generated = 0
        
        while tokens_generated < max_tokens:
            draft_ids, draft_probs = self._draft_tokens(
                generated_ids, temperature, 
                kwargs.get('top_p', 1.0),
                kwargs.get('top_k', 0)
            )
            
            accepted_ids, _ = self._verify_tokens(
                generated_ids, draft_ids, draft_probs,
                temperature,
                kwargs.get('top_p', 1.0),
                kwargs.get('top_k', 0)
            )
            
            generated_ids = torch.cat([generated_ids, accepted_ids], dim=1)
            tokens_generated += accepted_ids.shape[1]
            
            # Yield accepted tokens
            for token_id in accepted_ids[0]:
                yield self.tokenizer.decode([token_id.item()])


def create_speculative_decoder(
    target_model: nn.Module,
    draft_model: nn.Module,
    tokenizer: Any,
    num_speculative_tokens: int = 5
) -> SpeculativeDecoder:
    """
    Create a speculative decoder.
    
    Args:
        target_model: Large model for verification
        draft_model: Small model for drafting (should be ~10x smaller)
        tokenizer: Shared tokenizer
        num_speculative_tokens: Tokens to draft per iteration
    
    Returns:
        SpeculativeDecoder instance
    
    Example:
        # Load models
        target = load_model("forge-large")
        draft = load_model("forge-small")
        
        decoder = create_speculative_decoder(target, draft, tokenizer)
        output = decoder.generate("Once upon a time", max_tokens=100)
        print(f"Acceptance rate: {decoder.acceptance_rate:.1%}")
    """
    config = SpeculativeConfig(num_speculative_tokens=num_speculative_tokens)
    return SpeculativeDecoder(target_model, draft_model, tokenizer, config)
