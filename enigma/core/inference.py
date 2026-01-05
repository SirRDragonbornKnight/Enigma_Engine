"""
Enigma Inference Engine
=======================

High-performance inference engine for Enigma language models.

Features:
  - Efficient text generation with KV-cache support
  - Multiple sampling strategies (greedy, top-k, top-p, beam search)
  - Streaming generation for real-time output
  - Batch generation support
  - Chat-style conversation interface
  - Automatic device selection and optimization

Usage:
    from enigma.core.inference import EnigmaEngine

    engine = EnigmaEngine()
    response = engine.generate("Hello, my name is")
    print(response)

    # Streaming
    for token in engine.stream_generate("Tell me a story"):
        print(token, end="", flush=True)

    # Chat
    response = engine.chat("What is AI?")
"""
import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Generator, Dict, Any
from pathlib import Path
import logging

from .model import Enigma, create_model, MODEL_PRESETS
from .tokenizer import get_tokenizer
from ..config import CONFIG
from ..utils.system_messages import system_msg, info_msg

logger = logging.getLogger(__name__)

# Default model paths
MODELS_DIR = Path(CONFIG.get("models_dir", "models"))
DEFAULT_MODEL = MODELS_DIR / "enigma.pth"
LEGACY_MODEL = MODELS_DIR / "tiny_enigma.pth"


# =============================================================================
# Inference Engine
# =============================================================================

class EnigmaEngine:
    """
    High-performance inference engine for Enigma models.

    Features:
    - Automatic model loading and device selection
    - KV-cache for efficient autoregressive generation
    - Multiple sampling strategies (greedy, top-k, top-p)
    - Streaming generation support
    - Chat-style conversation interface
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        use_half: bool = False,
        model_size: str = "auto",
        enable_tools: bool = False,
        module_manager: Optional[Any] = None
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to model weights (auto-detected if None)
            tokenizer_path: Path to tokenizer (auto-detected if None)
            device: Device to use ("cuda", "cpu", or auto-detected)
            use_half: Use FP16 for faster inference (GPU only)
            model_size: Model size hint if not loading from file
            enable_tools: Enable AI tool use system
            module_manager: ModuleManager instance for tool execution
        """
        # Device selection
        self.device = self._select_device(device)
        self.use_half = use_half and self.device.type == "cuda"
        
        # Store configuration
        self.enable_tools = enable_tools
        self.module_manager = module_manager
        
        # Initialize tool executor if tools are enabled
        self._tool_executor = None
        if enable_tools:
            from ..tools.tool_executor import ToolExecutor
            self._tool_executor = ToolExecutor(module_manager=module_manager)

        # Load tokenizer
        self.tokenizer = self._load_tokenizer(tokenizer_path, model_path)

        # Load or create model
        self.model = self._load_model(model_path, model_size)

        # Move to device and set precision
        self.model.to(self.device)
        if self.use_half:
            self.model.half()
        self.model.eval()

        # Log initialization
        self._log_init_info()

    def _select_device(self, device: Optional[str]) -> torch.device:
        """Select the best available device."""
        if device is not None:
            return torch.device(device)

        # Check power mode settings
        try:
            from .power_mode import get_power_manager
            power_mgr = get_power_manager()
            if not power_mgr.should_use_gpu():
                # Power mode disabled GPU - use CPU
                cpu_threads = CONFIG.get("cpu_threads", 0)
                if cpu_threads > 0:
                    torch.set_num_threads(cpu_threads)
                return torch.device("cpu")
        except ImportError:
            pass  # Power mode not available

        if torch.cuda.is_available():
            # Apply GPU memory limit from config
            gpu_fraction = CONFIG.get("gpu_memory_fraction", 0.9)
            try:
                torch.cuda.set_per_process_memory_fraction(gpu_fraction)
            except (RuntimeError, AttributeError) as e:
                logger.debug(f"Could not set GPU memory fraction: {e}")
            return torch.device("cuda")

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")

        # Apply CPU thread limit from config
        cpu_threads = CONFIG.get("cpu_threads", 0)
        if cpu_threads > 0:
            torch.set_num_threads(cpu_threads)

        return torch.device("cpu")

    def _load_tokenizer(
        self,
        tokenizer_path: Optional[Union[str, Path]],
        model_path: Optional[Union[str, Path]]
    ) -> Any:
        """Load the tokenizer."""
        # Try explicit tokenizer path first
        if tokenizer_path:
            try:
                from .advanced_tokenizer import AdvancedBPETokenizer
                return AdvancedBPETokenizer(vocab_file=Path(tokenizer_path))
            except Exception as e:
                logger.warning(f"Could not load tokenizer from {tokenizer_path}: {e}")

        # Try to find tokenizer next to model file
        if model_path:
            model_path = Path(model_path)
            tok_path = model_path.parent / f"{model_path.stem}_tokenizer.json"
            if tok_path.exists():
                try:
                    from .advanced_tokenizer import AdvancedBPETokenizer
                    return AdvancedBPETokenizer(vocab_file=tok_path)
                except Exception as e:
                    logger.warning(f"Could not load tokenizer from {tok_path}: {e}")

        # Auto-detect model file and find tokenizer
        detected_model = None
        if DEFAULT_MODEL.exists():
            detected_model = DEFAULT_MODEL
        elif LEGACY_MODEL.exists():
            detected_model = LEGACY_MODEL
        else:
            for f in MODELS_DIR.glob("*.pth"):
                detected_model = f
                break
        
        if detected_model:
            tok_path = detected_model.parent / f"{detected_model.stem}_tokenizer.json"
            if tok_path.exists():
                try:
                    from .advanced_tokenizer import AdvancedBPETokenizer
                    tok = AdvancedBPETokenizer(vocab_file=tok_path)
                    logger.info(f"Loaded tokenizer from {tok_path}")
                    return tok
                except Exception as e:
                    logger.warning(f"Could not load tokenizer from {tok_path}: {e}")

        # Fall back to default
        return get_tokenizer()

    def _load_model(
        self,
        model_path: Optional[Union[str, Path]],
        model_size: str
    ) -> Enigma:
        """Load or create the model."""
        # Find model file
        model_file = None
        if model_path:
            model_file = Path(model_path)
        elif DEFAULT_MODEL.exists():
            model_file = DEFAULT_MODEL
        elif LEGACY_MODEL.exists():
            model_file = LEGACY_MODEL
        else:
            # Look for any .pth file in models dir
            for f in MODELS_DIR.glob("*.pth"):
                model_file = f
                break

        vocab_size = getattr(self.tokenizer, "vocab_size", 8000)

        if model_file and model_file.exists():
            # Load state dict to infer model architecture
            from .model_registry import safe_load_weights
            state_dict = safe_load_weights(model_file, map_location="cpu")

            # Infer model size from state dict
            detected_size = self._infer_model_size(state_dict)

            # Get vocab size from embedding
            for key in state_dict.keys():
                if 'embed' in key.lower() or 'token' in key.lower():
                    vocab_size = state_dict[key].shape[0]
                    break

            # Create model with correct architecture
            model = create_model(
                detected_size,
                vocab_size=vocab_size
            )

            # Load weights
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded model from {model_file}")
            except Exception as e:
                logger.warning(f"Could not load weights: {e}")
        else:
            # Create new model
            if model_size == "auto":
                model_size = "small"

            model = create_model(model_size, vocab_size=vocab_size)
            logger.info(f"Created new {model_size} model (no weights loaded)")

        return model

    def _infer_model_size(self, state_dict: Dict) -> str:
        """Infer model size from state dict."""
        # Look for hidden dimension from embedding layer
        hidden_dim = None
        for key, tensor in state_dict.items():
            if ('embed' in key.lower() or 'token' in key.lower()) and tensor.dim() == 2:
                hidden_dim = tensor.shape[1]
                break
        
        # Fallback: look for norm weights
        if hidden_dim is None:
            for key, tensor in state_dict.items():
                if ('ln' in key.lower() or 'norm' in key.lower()) and tensor.dim() == 1:
                    hidden_dim = tensor.shape[0]
                    break

        if hidden_dim is None:
            return "small"

        # Match to preset - MODEL_PRESETS values are EnigmaConfig with 'dim' attribute
        for name, preset in MODEL_PRESETS.items():
            preset_dim = getattr(preset, 'dim', None)
            if preset_dim == hidden_dim:
                return name

        # Find closest match
        def get_dim(preset):
            return getattr(preset, 'dim', 512)

        diffs = [(name, abs(get_dim(preset) - hidden_dim))
                 for name, preset in MODEL_PRESETS.items()]
        return min(diffs, key=lambda x: x[1])[0]

    def _log_init_info(self):
        """Log initialization information."""
        num_params = sum(p.numel() for p in self.model.parameters())

        print(system_msg(f"EnigmaEngine initialized on {self.device}"))
        if self.device.type == "cuda":
            print(info_msg(f"GPU: {torch.cuda.get_device_name(0)}"))
        print(info_msg(f"Model parameters: {num_params:,}"))
        print(info_msg(f"Vocab size: {self.tokenizer.vocab_size:,}"))
        print(info_msg(f"Max sequence length: {self.model.config.max_seq_len}"))
        print(info_msg(f"FP16: {self.use_half}"))

    # =========================================================================
    # Generation Methods
    # =========================================================================

    def generate(
        self,
        prompt: str,
        max_gen: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_strings: Optional[List[str]] = None,
        use_cache: bool = True,
        execute_tools: bool = None,
        max_tool_iterations: int = 5
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text to continue
            max_gen: Maximum tokens to generate (must be > 0)
            temperature: Sampling temperature (higher = more random, > 0)
            top_k: Top-k sampling (>= 0 to disable)
            top_p: Top-p (nucleus) sampling threshold (0-1)
            repetition_penalty: Penalty for repeating tokens (>= 1.0)
            stop_strings: List of strings to stop generation at
            use_cache: Use KV-cache for faster generation

        Returns:
            Generated text (including the prompt)

        Raises:
            ValueError: If parameters are out of valid range
            TypeError: If prompt is not a string
        """
        # Determine if tools should be executed
        if execute_tools is None:
            execute_tools = self.enable_tools
        
        # Standard generation
        text = self._generate_text(
            prompt, max_gen, temperature, top_k, top_p, 
            repetition_penalty, stop_strings, use_cache
        )
        
        # Tool execution loop
        if execute_tools and self._tool_executor:
            text = self._execute_tools_in_text(
                text, max_iterations=max_tool_iterations,
                max_gen=max_gen, temperature=temperature,
                top_k=top_k, top_p=top_p, 
                repetition_penalty=repetition_penalty,
                stop_strings=stop_strings, use_cache=use_cache
            )
        
        return text
    
    def _generate_text(
        self,
        prompt: str,
        max_gen: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        stop_strings: Optional[List[str]],
        use_cache: bool
    ) -> str:
        """Internal method for standard text generation."""
        # Validate inputs
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

        # Encode input
        input_ids = self._encode_prompt(prompt)

        # Generate
        with torch.no_grad():
            if use_cache and hasattr(self.model, 'generate'):
                # Use model's built-in generate
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

        # Apply stop strings
        if stop_strings:
            for stop_str in stop_strings:
                if stop_str in text:
                    text = text[:text.find(stop_str)]
                    break

        return text

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a prompt to tensor."""
        if hasattr(self.tokenizer, 'encode'):
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        else:
            enc = self.tokenizer(prompt, return_tensors="pt")
            ids = enc["input_ids"]
            if hasattr(ids, 'tolist'):
                ids = ids.tolist()
            if isinstance(ids[0], list):
                ids = ids[0]

        # Convert to tensor
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        return input_ids

    def _decode_output(self, output_ids: torch.Tensor) -> str:
        """Decode output tensor to text."""
        ids = output_ids[0].cpu().tolist()

        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(ids, skip_special_tokens=True)

        # Fallback
        return "".join(
            self.tokenizer.id_to_token.get(idx, "?")
            for idx in ids
        )

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
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                if 0 <= token_id < logits.shape[-1]:
                    logits[0, token_id] /= repetition_penalty

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
        repetition_penalty: float = 1.1
    ) -> Generator[str, None, None]:
        """
        Stream generated tokens one at a time.

        Args:
            prompt: Input text to continue
            max_gen: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty

        Yields:
            Each newly generated token as it's produced
        """
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
        prompts: List[str],
        max_gen: int = 100,
        **kwargs
    ) -> List[str]:
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
            ids = generated[i].cpu().tolist()
            
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

    # =========================================================================
    # Chat Interface
    # =========================================================================

    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        max_gen: int = 200,
        **kwargs
    ) -> str:
        """
        Chat-style generation with conversation history.

        Args:
            message: User's message
            history: List of {"role": "user/assistant", "content": "..."} dicts
            system_prompt: Optional system prompt
            max_gen: Maximum tokens to generate
            **kwargs: Additional generation parameters

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

        # Generate
        response = self.generate(
            full_prompt,
            max_gen=max_gen,
            stop_strings=["\nUser:", "\n\n", "User:"],
            **kwargs
        )

        # Extract assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        return response

    def stream_chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        max_gen: int = 200,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream chat-style generation.

        Args:
            message: User's message
            history: Conversation history
            system_prompt: Optional system prompt
            max_gen: Maximum tokens
            **kwargs: Additional parameters

        Yields:
            Generated tokens one at a time
        """
        # Build prompt
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

        # Stream generation
        buffer = ""
        for token in self.stream_generate(full_prompt, max_gen=max_gen, **kwargs):
            buffer += token

            # Check for stop conditions
            if "\nUser:" in buffer or buffer.endswith("\n\n"):
                # Remove stop string from output
                for stop in ["\nUser:", "\n\n"]:
                    if stop in buffer:
                        buffer = buffer[:buffer.find(stop)]
                break

            yield token


    # =========================================================================
    # Tool-Aware Generation
    # =========================================================================
    
    def generate_with_tools(
        self,
        prompt: str,
        module_manager=None,
        max_gen: int = 200,
        max_tool_iterations: int = 5,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        include_system_prompt: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text with tool execution support.
        
        The AI can invoke tools during generation, and the results are fed back
        for continued generation. This enables the AI to:
          - Generate images when asked
          - Control avatar expressions
          - Search the web for information
          - Read/write files
          - And more
        
        Args:
            prompt: Input text or user query
            module_manager: ModuleManager instance for tool access
            max_gen: Maximum tokens per generation step
            max_tool_iterations: Maximum number of tool calls in sequence
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            include_system_prompt: Prepend tool usage instructions
            **kwargs: Additional generation parameters
            
        Returns:
            Complete generated text with tool results
        """
        from .tool_interface import ToolInterface
        from .tool_prompts import get_tool_enabled_system_prompt
        
        # Create tool interface
        tool_interface = ToolInterface(module_manager)
        
        # Prepend system prompt if requested
        if include_system_prompt:
            system_prompt = get_tool_enabled_system_prompt()
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
        
        # Generate with tool support
        current_prompt = full_prompt
        full_output = ""
        iterations = 0
        
        while iterations < max_tool_iterations:
            # Generate next chunk
            output = self.generate(
                current_prompt,
                max_gen=max_gen,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_cache=True
            )
            
            # Extract new content (remove the prompt if it's in the output)
            if current_prompt in output:
                new_content = output[len(current_prompt):]
            else:
                new_content = output
            
            # Check for tool calls in the new content
            tool_call = tool_interface.parse_tool_call(new_content)
            
            if tool_call:
                # Execute the tool
                result = tool_interface.execute_tool(tool_call)
                result_str = tool_interface.format_tool_result(result)
                
                # Append tool call and result to output
                full_output += new_content[:tool_call.end_pos - tool_call.start_pos]
                full_output += "\n" + result_str + "\n"
                
                # Update prompt for next iteration
                current_prompt = full_prompt + full_output
                iterations += 1
                
                # Continue generation after tool result
                continue
            else:
                # No tool call found, we're done
                full_output += new_content
                break
        
        return full_output
    
    def stream_generate_with_tools(
        self,
        prompt: str,
        module_manager=None,
        max_gen: int = 200,
        max_tool_iterations: int = 5,
        include_system_prompt: bool = True,
        **kwargs
    ):
        """
        Stream generation with tool execution support.
        
        Yields tokens as they're generated, pausing for tool execution
        when tool calls are detected.
        
        Args:
            prompt: Input text
            module_manager: ModuleManager for tool access
            max_gen: Maximum tokens per step
            max_tool_iterations: Maximum tool calls
            include_system_prompt: Include tool instructions
            **kwargs: Additional parameters
            
        Yields:
            Generated tokens, including tool results
        """
        from .tool_interface import ToolInterface
        from .tool_prompts import get_tool_enabled_system_prompt
        
        tool_interface = ToolInterface(module_manager)
        
        if include_system_prompt:
            system_prompt = get_tool_enabled_system_prompt()
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
        
        current_prompt = full_prompt
        buffer = ""
        iterations = 0
        
        while iterations < max_tool_iterations:
            # Stream generate
            for token in self.stream_generate(current_prompt, max_gen=max_gen, **kwargs):
                buffer += token
                yield token
                
                # Check if we have a complete tool call
                if '<|tool_end|>' in buffer:
                    tool_call = tool_interface.parse_tool_call(buffer)
                    if tool_call:
                        # Execute tool
                        result = tool_interface.execute_tool(tool_call)
                        result_str = tool_interface.format_tool_result(result)
                        
                        # Yield result
                        yield "\n" + result_str + "\n"
                        
                        # Update prompt
                        current_prompt = full_prompt + buffer + "\n" + result_str + "\n"
                        buffer = ""
                        iterations += 1
                        break
            else:
                # Generation completed without tool call
                break


# =============================================================================
# Convenience Functions
# =============================================================================

def generate(
    prompt: str,
    model_path: Optional[str] = None,
    max_gen: int = 100,
    **kwargs
) -> str:
    """
    Quick generation function.

    Args:
        prompt: Input text
        model_path: Optional model path
        max_gen: Maximum tokens
        **kwargs: Additional parameters

    Returns:
        Generated text
    """
    engine = EnigmaEngine(model_path=model_path)
    return engine.generate(prompt, max_gen=max_gen, **kwargs)


def load_engine(
    model_path: Optional[str] = None,
    device: Optional[str] = None
) -> EnigmaEngine:
    """
    Load an inference engine.

    Args:
        model_path: Path to model
        device: Device to use

    Returns:
        EnigmaEngine instance
    """
    return EnigmaEngine(model_path=model_path, device=device)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "EnigmaEngine",
    "generate",
    "load_engine",
]
