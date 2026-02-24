"""
Generation and sampling methods for EnigmaEngine.

This module is split from ``inference.py`` to keep file sizes manageable.
The :class:`_GenerationMixin` is mixed into :class:`EnigmaEngine` so that
every method is available on the engine instance as before.

**Do not import from ``inference`` here** — that would create a circular
import.  All instance attributes (``self.model``, ``self.tokenizer``,
``self.device``, etc.) are set by ``EnigmaEngine.__init__`` which lives
in ``inference.py``.
"""
from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _GenerationMixin:
    """Generation, sampling, routing, and batch methods mixed into EnigmaEngine."""

    # =========================================================================
    # 🔀 ROUTING HELPERS — decide whether to use AI or direct tool dispatch
    # =========================================================================

    def _needs_ai_creativity(self, prompt: str) -> bool:
        """
        Check if the prompt requires AI creativity/context rather than direct execution.
        
        Returns True for ambiguous or creative requests that need AI interpretation.
        """
        prompt_lower = prompt.lower()
        
        # Phrases that indicate need for AI creativity
        creativity_indicators = [
            "what do you think",
            "surprise me",
            "something cool",
            "something interesting",
            "be creative",
            "your choice",
            "you decide",
            "like before",
            "like last time",
            "similar to",
            "in the style of",
            "mood",
            "feeling",
            "vibe",
            "not sure",
            "maybe",
            "suggest",
            "recommend",
            "what would",
            "how about",
            "can you think of",
        ]
        
        for indicator in creativity_indicators:
            if indicator in prompt_lower:
                return True
        
        # Very short prompts might be ambiguous
        words = prompt.split()
        if len(words) <= 2:
            # Single word commands are usually direct ("draw cat")
            # But single words alone are ambiguous
            if len(words) == 1 and words[0].lower() not in ['draw', 'paint', 'generate', 'create', 'make', 'speak', 'say']:
                return True
        
        return False
    
    def _try_direct_routing(self, intent: str, prompt: str) -> str | None:
        """
        Try to handle the request directly without main AI.
        
        Returns the response string if handled, None if should fall through to AI.
        """
        if intent == "image":
            logger.info("Direct routing to image generation")
            return self._direct_generation(prompt, "image", "generate_image")
        
        elif intent == "video":
            logger.info("Direct routing to video generation")
            return self._direct_generation(prompt, "video", "generate_video")
        
        elif intent == "audio":
            logger.info("Direct routing to audio/speech generation")
            return self._direct_generation(prompt, "audio", "speak_text")
        
        elif intent == "3d":
            logger.info("Direct routing to 3D generation")
            return self._direct_generation(prompt, "3D model", "generate_3d")
        
        elif intent == "gif":
            logger.info("Direct routing to GIF generation")
            return self._direct_generation(prompt, "GIF", "generate_gif")
        
        elif intent == "code" and hasattr(self._tool_router, 'generate_code'):
            logger.info("Using specialized code generation model")
            return self._tool_router.generate_code(prompt)
        
        elif intent == "vision" and hasattr(self._tool_router, 'describe_image'):
            logger.info("Vision routing detected, but no features provided")
            # Fall through - vision needs actual image input
            return None
        
        elif intent == "web":
            logger.info("Direct routing to web search")
            return self._direct_web_search(prompt)
        
        # Unknown intent or no direct handler - let AI handle it
        return None
    
    def _direct_generation(self, prompt: str, content_type: str, tool_name: str) -> str:
        """
        Generic direct generation handler for image/video/audio/3D/gif.
        
        Extracts description and calls the appropriate tool directly.
        """
        import re

        # Common patterns for extracting the actual content description
        description = prompt
        
        patterns = [
            r'(?:draw|paint|create|generate|make|produce)\s+(?:me\s+)?(?:a\s+)?(?:picture|image|photo|illustration|artwork|video|clip|animation|sound|audio|speech|model|mesh|gif)?\s*(?:of\s+)?(.+)',
            r'(?:draw|paint|create|generate|make|produce|speak|say|read)\s+(?:me\s+)?(.+)',
            r'(?:picture|image|photo|video|audio|model)\s+of\s+(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                description = match.group(1).strip()
                break
        
        # Clean up
        description = description.strip('.,!? ')
        if not description:
            description = prompt
        
        logger.info(f"Direct {content_type} generation: {description}")
        
        if not self._tool_executor:
            return f"To generate {content_type}, please use the {content_type.title()} tab. Direct generation not available."
        
        # Execute the tool (tool executor handles auto-loading)
        result = self._tool_executor.execute_tool(tool_name, {"prompt": description})
        
        if result.get("success"):
            path = result.get("path", result.get("result", {}).get("path", ""))
            duration = result.get("duration", 0)
            return f"I've generated {content_type} of '{description}' for you.\n\nSaved to: {path}\nGeneration time: {duration:.1f}s"
        else:
            error = result.get("error", "Unknown error")
            tab_name = content_type.title().replace("3d", "3D")
            return f"I tried to generate {content_type} but encountered an error: {error}\n\nYou can try using the {tab_name} tab directly."
    
    def _direct_web_search(self, prompt: str) -> str:
        """Direct web search without AI intermediary."""
        import re

        # Check if web access is enabled
        if not getattr(self, '_web_enabled', False):
            return "Web access is disabled. Click the 'Web' button in the chat header to enable internet access."

        # Extract search query
        query = prompt
        patterns = [
            r'(?:search|google|look up|find|browse)\s+(?:for\s+)?(.+)',
            r'what is\s+(.+)',
            r'who is\s+(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                break
        
        query = query.strip('?,!. ')
        
        if not self._tool_executor:
            return f"To search for '{query}', please use the web tools. Direct search not available."
        
        result = self._tool_executor.execute_tool("web_search", {"query": query})
        
        if result.get("success"):
            search_results = result.get("result", result.get("results", "No results found"))
            return f"Search results for '{query}':\n\n{search_results}"
        else:
            error = result.get("error", "Unknown error")
            return f"Web search failed: {error}"

    # =========================================================================
    # 🔧 INTERNAL GENERATION - Where the magic happens!
    # =========================================================================

    def _generate_text(
        self,
        prompt: str,
        max_gen: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        stop_strings: list[str] | None,
        use_cache: bool
    ) -> str:
        """
        Internal method for standard text generation.
        
        📖 THIS IS THE CORE GENERATION LOOP!
        
        📐 THE AUTOREGRESSIVE LOOP:
        ┌────────────────────────────────────────────────────────────────────┐
        │  tokens = [15496, 11, 703, 389]  # "Hello, how are"               │
        │                                                                    │
        │  REPEAT max_gen times:                                            │
        │    1. Feed tokens to model → logits                               │
        │    2. Apply repetition penalty to logits                          │
        │    3. Apply temperature scaling                                   │
        │    4. Apply top-k filtering                                       │
        │    5. Apply top-p (nucleus) filtering                             │
        │    6. Sample next token from probabilities                        │
        │    7. Add new token to sequence                                   │
        │    8. Check for stop strings                                      │
        │    9. Check for EOS token                                         │
        │                                                                    │
        │  tokens = [15496, 11, 703, 389, 499, 1804, 2651]                  │
        │                                    └─ newly generated             │
        └────────────────────────────────────────────────────────────────────┘
        
        📐 REPETITION PENALTY:
        Discourages the model from repeating the same tokens.
        For each token that already appeared in the sequence,
        divide its probability by repetition_penalty.
        
        📐 TEMPERATURE SCALING:
        logits = logits / temperature
        - Low temp (0.3): Makes high-prob tokens even more likely → focused
        - High temp (1.5): Flattens distribution → more random
        
        📐 TOP-K FILTERING:
        Keep only the K highest probability tokens, zero out the rest.
        Prevents sampling very unlikely tokens.
        
        📐 TOP-P (NUCLEUS) FILTERING:
        Sort tokens by probability, keep tokens until cumulative prob >= p.
        Dynamic cutoff - keeps more tokens when uncertain, fewer when confident.
        """
        # ─────────────────────────────────────────────────────────────────────
        # GGUF MODEL HANDLING (llama.cpp)
        # GGUF models have their own generation - use it directly
        # ─────────────────────────────────────────────────────────────────────
        if getattr(self, '_is_gguf', False) or (hasattr(self.model, 'is_loaded') and hasattr(self.model, 'chat')):
            # This is a GGUFModel - use its native generation
            try:
                text = self.model.generate(
                    prompt,
                    max_tokens=max_gen,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repetition_penalty,
                    stop=stop_strings
                )
                return text
            except Exception as e:
                logger.error(f"GGUF generation failed: {e}")
                return f"Error: {e}"
        
        # ─────────────────────────────────────────────────────────────────────
        # INPUT VALIDATION
        # ─────────────────────────────────────────────────────────────────────
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")

        if not prompt.strip():
            logger.warning("Empty prompt provided")
            return ""

        if max_gen <= 0:
            raise ValueError(f"max_gen must be positive, got {max_gen}")

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")

        if not 0 <= top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1, got {top_p}")

        if repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {repetition_penalty}")

        # ─────────────────────────────────────────────────────────────────────
        # TOKENIZE: Convert text to numbers the model understands
        # "Hello" → [15496]
        # ─────────────────────────────────────────────────────────────────────
        input_ids = self._encode_prompt(prompt)

        # ─────────────────────────────────────────────────────────────────────
        # GENERATE: Run the autoregressive loop
        # ─────────────────────────────────────────────────────────────────────
        with torch.no_grad():  # Disable gradient computation (inference only)
            if use_cache and hasattr(self.model, 'generate'):
                # Use model's built-in generate (has KV-cache optimization)
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_gen,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
            else:
                # Manual generation
                output_ids = self._generate_manual(
                    input_ids, max_gen, temperature, top_k, top_p, repetition_penalty
                )

        # Decode
        text = self._decode_output(output_ids)

        # Apply stop strings - only check in generated part (after prompt)
        # This prevents cutting off at stop strings that exist in the prompt itself
        if stop_strings:
            prompt_len = len(prompt)
            generated_part = text[prompt_len:] if len(text) > prompt_len else text
            for stop_str in stop_strings:
                if stop_str in generated_part:
                    stop_pos = generated_part.find(stop_str)
                    text = text[:prompt_len + stop_pos]
                    break

        return text

    def _generate_manual(
        self,
        input_ids: torch.Tensor,
        max_gen: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float
    ) -> torch.Tensor:
        """Manual autoregressive generation."""
        generated = input_ids

        for _ in range(max_gen):
            # Truncate if needed
            curr_input = generated
            max_len = self.model.config.max_seq_len
            if curr_input.shape[1] > max_len:
                curr_input = curr_input[:, -max_len:]

            # Forward pass
            logits = self.model(curr_input)

            # Sample next token
            next_token = self._sample_token(
                logits[:, -1, :],
                generated,
                temperature,
                top_k,
                top_p,
                repetition_penalty
            )

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            eos_id = getattr(self.tokenizer, 'eos_token_id', 2)
            if next_token[0, 0].item() == eos_id:
                break

        return generated

    def _sample_token(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float
    ) -> torch.Tensor:
        """Sample next token with various strategies."""
        # Apply repetition penalty - O(vocabulary) vectorized operation
        if repetition_penalty != 1.0:
            vocab_size = logits.shape[-1]
            # Clamp token IDs to valid vocab range and count occurrences
            token_ids = generated[0].clamp(0, vocab_size - 1)
            token_counts = torch.bincount(token_ids, minlength=vocab_size)
            # Create mask for tokens that have appeared
            appeared_mask = token_counts > 0
            # Apply penalty vectorized (much faster than loop)
            logits[0, appeared_mask] = logits[0, appeared_mask] / repetition_penalty

        # Temperature scaling
        logits = logits / max(temperature, 1e-8)

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_value = values[:, -1, None]
            logits = torch.where(logits < min_value, float('-inf'), logits)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # =========================================================================
    # Streaming Generation
    # =========================================================================

    def stream_generate(
        self,
        prompt: str,
        max_gen: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_tokens: int | None = None,  # Alias for max_gen (backward compatibility)
        max_new_tokens: int | None = None,  # Alias for max_gen (Forge model compatibility)
        max_length: int | None = None  # Alias for max_gen (common parameter name)
    ) -> Generator[str]:
        """
        Stream generated tokens one at a time.

        Args:
            prompt: Input text to continue
            max_gen: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            max_tokens: Alias for max_gen (backward compatibility)
            max_new_tokens: Alias for max_gen (Forge model compatibility)
            max_length: Alias for max_gen (common parameter name)

        Yields:
            Each newly generated token as it's produced
        """
        # Handle max_tokens, max_new_tokens, max_length aliases for backward compatibility
        if max_tokens is not None:
            max_gen = max_tokens
        if max_new_tokens is not None:
            max_gen = max_new_tokens
        if max_length is not None:
            max_gen = max_length
        
        input_ids = self._encode_prompt(prompt)
        generated = input_ids

        with torch.no_grad():
            for _ in range(max_gen):
                # Truncate if needed
                curr_input = generated
                max_len = self.model.config.max_seq_len
                if curr_input.shape[1] > max_len:
                    curr_input = curr_input[:, -max_len:]

                # Forward pass
                logits = self.model(curr_input)

                # Sample
                next_token = self._sample_token(
                    logits[:, -1, :],
                    generated,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty
                )

                generated = torch.cat([generated, next_token], dim=1)

                # Decode and yield
                token_id = next_token[0, 0].item()

                if hasattr(self.tokenizer, 'decode'):
                    token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
                else:
                    token_str = self.tokenizer.id_to_token.get(token_id, "")

                yield token_str

                # Check for EOS
                eos_id = getattr(self.tokenizer, 'eos_token_id', 2)
                if token_id == eos_id:
                    break

    # =========================================================================
    # Batch Generation
    # =========================================================================

    def batch_generate(
        self,
        prompts: list[str],
        max_gen: int = 100,
        **kwargs
    ) -> list[str]:
        """
        Generate text for multiple prompts in a single batched forward pass.

        Args:
            prompts: List of input prompts
            max_gen: Maximum tokens to generate per prompt
            **kwargs: Additional generation parameters (temperature, top_k, top_p, repetition_penalty)

        Returns:
            List of generated texts
        """
        if not prompts:
            return []
        
        # If only one prompt, use regular generate
        if len(prompts) == 1:
            return [self.generate(prompts[0], max_gen=max_gen, **kwargs)]
        
        # Extract generation parameters
        temperature = kwargs.get('temperature', 0.8)
        top_k = kwargs.get('top_k', 50)
        top_p = kwargs.get('top_p', 0.9)
        repetition_penalty = kwargs.get('repetition_penalty', 1.1)
        
        # Encode all prompts
        if hasattr(self.tokenizer, 'encode'):
            encoded = [self.tokenizer.encode(p) for p in prompts]
        else:
            encoded = [[self.tokenizer.token_to_id.get(t, 3) for t in p] for p in prompts]
        
        # Pad all sequences to the same length
        max_input_len = max(len(e) for e in encoded)
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
        
        # Create padded batch tensor
        batch_size = len(prompts)
        input_ids = torch.full(
            (batch_size, max_input_len),
            pad_id,
            dtype=torch.long,
            device=self.device
        )
        
        # Fill in the actual tokens
        for i, tokens in enumerate(encoded):
            input_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        
        # Track which sequences are still generating
        eos_id = getattr(self.tokenizer, 'eos_token_id', 2)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Generate tokens autoregressively
        generated = input_ids
        all_finished = False
        for step in range(max_gen):
            # Early exit if all sequences finished (check every 5 steps starting from step 5 to reduce overhead)
            if all_finished or (step >= 5 and step % 5 == 0 and finished.all()):
                all_finished = True
                break
            
            # Truncate if needed
            curr_input = generated
            max_len = self.model.config.max_seq_len
            if curr_input.shape[1] > max_len:
                curr_input = curr_input[:, -max_len:]
            
            # Forward pass for entire batch
            with torch.no_grad():
                logits = self.model(curr_input)
            
            # Sample next token for each sequence
            next_tokens = []
            for i in range(batch_size):
                if finished[i]:
                    # Already finished, use pad token
                    next_tokens.append(pad_id)
                else:
                    # Sample from logits
                    token = self._sample_token(
                        logits[i:i+1, -1:, :],
                        generated[i:i+1],
                        temperature,
                        top_k,
                        top_p,
                        repetition_penalty
                    )
                    next_tokens.append(token.item())
                    
                    # Check for EOS
                    if token.item() == eos_id:
                        finished[i] = True
            
            # Append next tokens to generated
            next_tokens_tensor = torch.tensor(
                [[t] for t in next_tokens],
                dtype=torch.long,
                device=self.device
            )
            generated = torch.cat([generated, next_tokens_tensor], dim=1)
        
        # Decode all outputs
        results = []
        for i in range(batch_size):
            # Handle case where generated[i] might not be a tensor
            try:
                ids = generated[i].cpu().tolist()
            except AttributeError:
                if isinstance(generated[i], str):
                    results.append(generated[i])
                    continue
                ids = list(generated[i]) if hasattr(generated[i], '__iter__') else [generated[i]]
            
            if hasattr(self.tokenizer, 'decode'):
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
            else:
                # Fallback
                text = "".join(
                    self.tokenizer.id_to_token.get(idx, "?")
                    for idx in ids
                )
            
            results.append(text)
        
        return results
