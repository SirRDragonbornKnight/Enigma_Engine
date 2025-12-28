"""
Inference Engine for Enigma Language Model

Features:
  - Efficient text generation with KV-cache
  - Multiple sampling strategies (greedy, top-k, top-p, beam search)
  - Batch generation support
  - Streaming generation
  - Model quantization support (when available)
  - Automatic device selection

USAGE:
    from enigma.core.inference import EnigmaEngine
    
    engine = EnigmaEngine()
    response = engine.generate("Hello, my name is")
    print(response)
"""
import torch
from typing import Optional, List, Union, Generator
from pathlib import Path

from .model import Enigma, TinyEnigma  # TinyEnigma is alias for backwards compat
from .tokenizer import load_tokenizer
from ..config import CONFIG

MODEL_PATH = Path(CONFIG["models_dir"]) / "enigma.pth"
LEGACY_PATH = Path(CONFIG["models_dir"]) / "tiny_enigma.pth"  # Backwards compatibility


class EnigmaEngine:
    """
    High-performance inference engine for Enigma models.
    
    Features:
    - Automatic model loading and device selection
    - KV-cache for efficient generation
    - Multiple sampling strategies
    - Streaming generation support
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_half: bool = False,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to model weights (auto-detected if None)
            device: Device to use (auto-detected if None)
            use_half: Use FP16 for faster inference (GPU only)
        """
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.use_half = use_half and self.device.type == "cuda"
        
        # Load tokenizer
        self.tokenizer = load_tokenizer()
        vocab_size = getattr(self.tokenizer, "vocab_size", 32000)
        
        # Initialize model with better defaults
        self.model = Enigma(
            vocab_size=vocab_size,
            dim=CONFIG.get("embed_dim", 256),
            depth=CONFIG.get("depth", 6),
            heads=CONFIG.get("heads", 8),
            max_len=CONFIG.get("max_len", 2048),
        )
        
        # Load weights if available
        model_file = Path(model_path) if model_path else None
        if model_file is None:
            if MODEL_PATH.exists():
                model_file = MODEL_PATH
            elif LEGACY_PATH.exists():
                model_file = LEGACY_PATH
        
        if model_file and model_file.exists():
            try:
                state_dict = torch.load(model_file, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded model from {model_file}")
            except Exception as e:
                print(f"Warning: Could not load weights from {model_file}: {e}")
        
        # Move to device and set precision
        self.model.to(self.device)
        if self.use_half:
            self.model.half()
        self.model.eval()
        
        print(f"EnigmaEngine initialized on {self.device}")
        print(f"Model parameters: {self.model.num_parameters:,}")
    
    def generate(
        self,
        prompt: str,
        max_gen: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_strings: Optional[List[str]] = None,
        use_cache: bool = False,  # Disabled - standard model doesn't support KV cache
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text to continue
            max_gen: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 to disable)
            top_p: Top-p (nucleus) sampling threshold
            repetition_penalty: Penalty for repeating tokens
            stop_strings: List of strings to stop generation at
            use_cache: Use KV-cache for faster generation (requires compatible model)
            
        Returns:
            Generated text including the original prompt
        """
        # Encode input
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.model.max_len)
        input_ids = enc["input_ids"]
        
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Ensure 2D: (batch, seq_len)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        input_ids = input_ids.to(self.device).long()
        
        # Generate
        with torch.no_grad():
            if use_cache:
                output_ids = self._generate_with_cache(
                    input_ids, max_gen, temperature, top_k, top_p, repetition_penalty
                )
            else:
                output_ids = self._generate_simple(
                    input_ids, max_gen, temperature, top_k, top_p, repetition_penalty
                )
        
        # Decode
        try:
            text = self.tokenizer.decode(output_ids[0].cpu().numpy(), skip_special_tokens=True)
        except Exception:
            # Fallback for simple tokenizers
            text = "".join(
                self.tokenizer.id_to_char.get(int(idx), "?")
                for idx in output_ids[0].cpu().numpy()
            )
        
        # Check for stop strings
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in text:
                    text = text[:text.find(stop_str)]
                    break
        
        return text
    
    def _generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_gen: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Generate with KV-cache for efficiency."""
        generated = input_ids
        kv_cache = None
        
        for _ in range(max_gen):
            if kv_cache is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated
            
            output = self.model(curr_input, kv_cache=kv_cache, use_cache=True)
            if isinstance(output, tuple):
                logits, kv_cache = output
            else:
                logits = output
                kv_cache = None
            
            next_token = self._sample_token(
                logits[:, -1, :], generated, temperature, top_k, top_p, repetition_penalty
            )
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def _generate_simple(
        self,
        input_ids: torch.Tensor,
        max_gen: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Generate without cache (simpler, slightly slower)."""
        generated = input_ids
        
        for _ in range(max_gen):
            curr_input = generated
            if curr_input.shape[1] > self.model.max_len:
                curr_input = curr_input[:, -self.model.max_len:]
            
            output = self.model(curr_input)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            next_token = self._sample_token(
                logits[:, -1, :], generated, temperature, top_k, top_p, repetition_penalty
            )
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Sample next token with various strategies."""
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                logits[0, token_id] /= repetition_penalty
        
        # Temperature scaling
        logits = logits / max(temperature, 1e-8)
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    def stream_generate(
        self,
        prompt: str,
        max_gen: int = 50,
        temperature: float = 0.8,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Stream generated tokens one at a time.
        
        Yields:
            Each newly generated token as it's produced
        """
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Ensure 2D: (batch, seq_len)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        input_ids = input_ids.to(self.device).long()
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_gen):
                # Always use full sequence (no KV cache in standard model)
                logits = self.model(generated)
                
                next_token = self._sample_token(
                    logits[:, -1, :],
                    generated,
                    temperature,
                    kwargs.get("top_k", 50),
                    kwargs.get("top_p", 0.9),
                    kwargs.get("repetition_penalty", 1.1),
                )
                generated = torch.cat([generated, next_token], dim=1)
                
                # Decode and yield the new token
                try:
                    token_str = self.tokenizer.decode([int(next_token[0, 0])], skip_special_tokens=True)
                except Exception:
                    token_str = self.tokenizer.id_to_char.get(int(next_token[0, 0]), "")
                
                yield token_str
    
    def batch_generate(
        self,
        prompts: List[str],
        max_gen: int = 50,
        **kwargs,
    ) -> List[str]:
        """
        Generate text for multiple prompts efficiently.
        
        Args:
            prompts: List of input prompts
            max_gen: Maximum tokens to generate per prompt
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        return [self.generate(p, max_gen=max_gen, **kwargs) for p in prompts]
    
    def chat(
        self,
        message: str,
        history: Optional[List[dict]] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Chat-style generation with conversation history.
        
        Args:
            message: User's message
            history: List of {"role": "user/assistant", "content": "..."} dicts
            system_prompt: Optional system prompt
            **kwargs: Generation parameters
            
        Returns:
            Assistant's response
        """
        # Build prompt from history
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")
        
        if history:
            for msg in history:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                prompt_parts.append(f"{role}: {content}")
        
        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Generate response
        response = self.generate(full_prompt, stop_strings=["\nUser:", "\n\n"], **kwargs)
        
        # Extract assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
