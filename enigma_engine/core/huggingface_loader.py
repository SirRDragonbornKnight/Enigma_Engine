"""
HuggingFace Model Loader
========================

Load and run HuggingFace Transformers models for text generation.
Enables using pre-trained models like GPT-2, Llama, Mistral, etc.

Usage:
    from enigma_engine.core.huggingface_loader import HuggingFaceModel
    
    # Load a model
    model = HuggingFaceModel("gpt2")
    model.load()
    
    # Generate text
    response = model.generate("Once upon a time")
    print(response)
    
    # Stream generation
    for token in model.stream_generate("Tell me a story"):
        print(token, end="", flush=True)
    
    # Chat interface
    response = model.chat("What is the capital of France?")
    
    # Get model info without loading
    info = get_huggingface_model_info("microsoft/DialoGPT-small")
    print(f"Size: {info['size_str']}, Params: {info['num_parameters']:,}")
"""

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)

# Type hints for optional dependencies
if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

# Check for transformers
HAVE_TRANSFORMERS = False
torch: Any = None
AutoModelForCausalLM: Any = None
AutoTokenizer: Any = None
TextIteratorStreamer: Any = None
GenerationConfig: Any = None
BitsAndBytesConfig: Any = None

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from transformers.generation.streamers import TextIteratorStreamer
    from transformers.utils.quantization_config import BitsAndBytesConfig
    HAVE_TRANSFORMERS = True
except ImportError:
    logger.warning("transformers not available - HuggingFace loading disabled")

# Check for threading (for streaming)
from threading import Thread


def format_param_count(num_params: int) -> str:
    """Format parameter count as human readable string (e.g., '124M', '7B')."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.1f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.0f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.0f}K"
    return str(num_params)


def get_huggingface_model_info(model_id: str, timeout: float = 10.0) -> dict[str, Any]:
    """
    Get model information from HuggingFace without downloading the full model.
    
    This fetches just the config file to estimate parameter count and architecture.
    
    Args:
        model_id: HuggingFace model ID (e.g., "microsoft/DialoGPT-small")
        timeout: Request timeout in seconds
        
    Returns:
        Dict with:
            - num_parameters: Estimated parameter count
            - size_str: Human readable size (e.g., "124M")
            - hidden_size: Model dimension
            - num_layers: Number of transformer layers
            - vocab_size: Vocabulary size
            - architecture: Model architecture type
            - error: Error message if failed (None if success)
    """
    result = {
        "num_parameters": 0,
        "size_str": "?",
        "hidden_size": 0,
        "num_layers": 0,
        "vocab_size": 0,
        "architecture": "unknown",
        "error": None
    }
    
    if not HAVE_TRANSFORMERS:
        result["error"] = "transformers not installed"
        return result
    
    try:
        from transformers import AutoConfig

        # Fetch just the config (small download)
        config = AutoConfig.from_pretrained(model_id)
        
        # Extract architecture info - different models use different attribute names
        hidden = getattr(config, 'hidden_size', None) or \
                 getattr(config, 'n_embd', None) or \
                 getattr(config, 'd_model', None) or 768
        
        layers = getattr(config, 'num_hidden_layers', None) or \
                 getattr(config, 'n_layer', None) or \
                 getattr(config, 'num_layers', None) or 12
        
        vocab = getattr(config, 'vocab_size', 50257)
        
        # Get architecture type
        arch = getattr(config, 'model_type', 'unknown')
        
        # Estimate parameter count
        # Formula: embed + output + layers * (attention + ffn)
        # Attention: 4 * hidden^2 (Q, K, V, output projections)
        # FFN: 8 * hidden^2 (typical 4x expansion)
        embed_params = vocab * hidden * 2  # input + output embeddings
        layer_params = layers * (4 * hidden * hidden + 8 * hidden * hidden)  # ~12 * hidden^2 per layer
        total_params = embed_params + layer_params
        
        result["num_parameters"] = total_params
        result["size_str"] = format_param_count(total_params)
        result["hidden_size"] = hidden
        result["num_layers"] = layers
        result["vocab_size"] = vocab
        result["architecture"] = arch
        
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"Failed to get HuggingFace model info for {model_id}: {e}")
    
    return result


class HuggingFaceModel:
    """
    HuggingFace Transformers model loader.
    
    Supports:
    - Any causal LM from HuggingFace Hub
    - Local model directories
    - Quantized loading (8-bit, 4-bit)
    - Streaming generation
    - Chat templates
    """
    
    # Popular model suggestions
    SUGGESTED_MODELS = {
        "tiny": "microsoft/DialoGPT-small",      # ~124M params, fast
        "small": "gpt2",                          # ~124M params, classic
        "medium": "gpt2-medium",                  # ~355M params
        "large": "gpt2-large",                    # ~774M params
        "xl": "gpt2-xl",                          # ~1.5B params
        "chat": "microsoft/DialoGPT-medium",     # Good for conversation
        "code": "Salesforce/codegen-350M-mono",  # For code generation
        "grok": "xai-org/grok-1",                 # ~314B params, xAI
    }
    
    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        max_memory: Optional[dict] = None,
    ):
        """
        Initialize HuggingFace model.
        
        Args:
            model_id: HuggingFace model ID or local path
            device: Device to use ("cuda", "cpu", "auto")
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
            torch_dtype: Data type ("float16", "bfloat16", "float32")
            trust_remote_code: Trust remote code for custom models
            max_memory: Memory allocation per device
        """
        if not HAVE_TRANSFORMERS:
            raise RuntimeError(
                "HuggingFace loading requires transformers. "
                "Install with: pip install transformers"
            )
        
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.max_memory = max_memory
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        logger.info(f"HuggingFace model initialized: {model_id}")
    
    def load(self) -> bool:
        """
        Load the model and tokenizer.
        
        Returns:
            True if loaded successfully
        """
        if self.is_loaded:
            logger.info("Model already loaded")
            return True
        
        try:
            logger.info(f"Loading model: {self.model_id}")
            
            # Determine torch dtype
            dtype = None
            if self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            elif self.torch_dtype == "float32":
                dtype = torch.float32
            elif self.device == "cuda":
                dtype = torch.float16  # Default to fp16 on GPU
            
            # Quantization config
            quantization_config = None
            if self.load_in_4bit or self.load_in_8bit:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=self.load_in_8bit,
                        load_in_4bit=self.load_in_4bit,
                    )
                except Exception as e:
                    logger.warning(f"Quantization not available: {e}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs: dict[str, Any] = {
                "trust_remote_code": self.trust_remote_code,
            }
            
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            elif self.device != "cpu":
                model_kwargs["device_map"] = "auto"
            
            if self.max_memory is not None:
                model_kwargs["max_memory"] = self.max_memory
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if "device_map" not in model_kwargs and self.device != "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.is_loaded = True
            
            logger.info(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def unload(self):
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.is_loaded = False
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stop_strings: Optional[list[str]] = None,
        custom_tokenizer: Optional[Any] = None,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repetition_penalty: Penalty for repeated tokens
            do_sample: Use sampling (False = greedy)
            stop_strings: Strings that stop generation
            custom_tokenizer: Optional custom tokenizer to use instead of model's own
        Returns:
            Generated text (excluding prompt)
        """
        if not self.is_loaded:
            self.load()
        
        # Ensure prompt is a string (tokenizers require str type)
        if prompt is None:
            prompt = ""
        prompt = str(prompt)
        if not prompt.strip():
            return "(Empty prompt provided)"
        
        # Type assertions for loaded state
        assert self.tokenizer is not None, "Tokenizer not loaded"
        assert self.model is not None, "Model not loaded"
        
        # Use custom tokenizer if provided, otherwise use model's tokenizer
        tokenizer = custom_tokenizer if custom_tokenizer is not None else self.tokenizer
        
        # Special handling for DialoGPT models - they need EOS token appended
        is_dialogpt = "dialogpt" in self.model_id.lower()
        
        # Handle custom tokenizer encoding
        if custom_tokenizer is not None:
            # Custom Forge tokenizer uses .encode() method
            if hasattr(custom_tokenizer, 'encode'):
                tokens = custom_tokenizer.encode(prompt)
                input_ids = torch.tensor([tokens]).to(self.model.device)
                attention_mask = torch.ones_like(input_ids)
            else:
                # Fallback to standard tokenizer call
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)
        else:
            # Standard HuggingFace tokenizer
            if is_dialogpt:
                # DialoGPT expects input to end with EOS token
                prompt_with_eos = prompt + tokenizer.eos_token
                inputs = tokenizer(prompt_with_eos, return_tensors="pt")
            else:
                inputs = tokenizer(prompt, return_tensors="pt")
                
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)
        
        # For custom tokenizer, we need to use model's special tokens for generation
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        
        # Decode only the new tokens
        generated_ids = outputs[0][input_ids.shape[1]:]
        
        # Use appropriate tokenizer for decoding
        if custom_tokenizer is not None and hasattr(custom_tokenizer, 'decode'):
            generated_text = custom_tokenizer.decode(generated_ids.tolist())
        else:
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Apply stop strings
        if stop_strings:
            for stop in stop_strings:
                if stop in generated_text:
                    generated_text = generated_text[:generated_text.index(stop)]
        
        return generated_text
    
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> Generator[str, None, None]:
        """
        Stream generate text token by token.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repetition_penalty: Penalty for repeated tokens
            
        Yields:
            Generated tokens one at a time
        """
        if not self.is_loaded:
            self.load()
        
        # Type assertions for loaded state
        assert self.tokenizer is not None, "Tokenizer not loaded"
        assert self.model is not None, "Model not loaded"
        
        # Encode input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        # Generation kwargs
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": True,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Run generation in a thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens as they come
        yield from streamer
        
        thread.join()
    
    def chat(
        self,
        message: str,
        history: Optional[list[dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """
        Chat-style generation with conversation history.
        
        Args:
            message: User message
            history: List of {"role": "user/assistant", "content": "..."} dicts
            system_prompt: System prompt to prepend
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Assistant response
        """
        if not self.is_loaded:
            self.load()
        
        # Ensure message is a string (tokenizers require str type)
        if message is None:
            message = ""
        message = str(message)
        if not message.strip():
            return "(Empty message provided)"
        
        # Type assertions for loaded state
        assert self.tokenizer is not None, "Tokenizer not loaded"
        assert self.model is not None, "Model not loaded"
        
        # Special handling for DialoGPT models
        is_dialogpt = "dialogpt" in self.model_id.lower()
        
        if is_dialogpt:
            # DialoGPT uses a specific format: previous turns separated by EOS token
            # Format: "turn1<|endoftext|>turn2<|endoftext|>current_input<|endoftext|>"
            conversation_parts = []
            
            if history:
                for msg in history:
                    conversation_parts.append(msg["content"])
            
            conversation_parts.append(message)
            
            # Join with EOS token and add final EOS for generation
            prompt = self.tokenizer.eos_token.join(conversation_parts) + self.tokenizer.eos_token
            
            # Generate with DialoGPT-specific settings
            response = self.generate(
                prompt,
                max_new_tokens=min(max_new_tokens, 50),  # DialoGPT works better with shorter responses
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,  # Higher repetition penalty for DialoGPT
                do_sample=True,
            )
            
            # Clean up response
            response = response.strip()
            # Remove any trailing special tokens
            if self.tokenizer.eos_token and response.endswith(self.tokenizer.eos_token):
                response = response[:-len(self.tokenizer.eos_token)].strip()
            
            return response if response else "I'm not sure how to respond to that."
        
        # Standard chat flow for other models
        # Build conversation
        messages = []
        
        # Add system prompt - use provided or default for instruct models
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif "instruct" in self.model_id.lower() or "chat" in self.model_id.lower():
            # Default system prompt for instruct/chat models (only if none provided)
            # Be explicit about conversational behavior to avoid code generation
            messages.append({"role": "system", "content": 
                "You are a friendly, helpful AI assistant. Have a natural conversation with the user. "
                "Respond in plain English sentences. Do not write code unless specifically asked. "
                "Keep responses concise and conversational."
            })
        
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": message})
        
        # Try to use chat template if available
        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                # Ensure prompt is a string (some templates return lists)
                if not isinstance(prompt, str):
                    prompt = str(prompt)
                logger.debug(f"Using chat template, prompt: {prompt[:200]}...")
            else:
                # Fallback: simple formatting
                prompt = self._format_chat_simple(messages)
        except Exception as e:
            logger.warning(f"Chat template failed: {e}, using simple format")
            prompt = self._format_chat_simple(messages)
        
        # Final string validation
        prompt = str(prompt) if prompt else ""
        
        # Model-specific stop strings for different chat formats
        # Include common formats: Zephyr/TinyLlama (<|user|>), Llama ([INST]), generic (User:, Human:)
        stop_strings = [
            "User:", "Human:", "\n\nUser", "\n\nHuman",  # Generic format
            "</s>", "[/INST]",  # Llama/Mistral format
            "<|user|>", "<|system|>", "<|end|>",  # Zephyr/TinyLlama format
            "\n<|", "```\n\n",  # Common continuation signals
        ]
        
        # Generate with slightly higher repetition penalty for small models
        is_small_model = any(x in self.model_id.lower() for x in ["tiny", "small", "mini", "1b", "1.1b"])
        rep_penalty = 1.2 if is_small_model else 1.1
        
        # Generate
        response = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=rep_penalty,
            stop_strings=stop_strings,
        )
        
        # Additional cleanup for small models that might include artifacts
        response = response.strip()
        
        # Remove any trailing incomplete code blocks or markdown artifacts
        if "```" in response and response.count("```") % 2 == 1:
            # Incomplete code block - truncate at the opening
            last_open = response.rfind("```")
            if last_open > 0:
                response = response[:last_open].strip()
        
        return response
    
    def _format_chat_simple(self, messages: list[dict[str, str]]) -> str:
        """Simple chat formatting for models without chat templates."""
        lines = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            if role == "System":
                lines.append(f"System: {content}\n")
            elif role == "User":
                lines.append(f"User: {content}\n")
            elif role == "Assistant":
                lines.append(f"Assistant: {content}\n")
        lines.append("Assistant:")
        return "".join(lines)
    
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            return {"loaded": False, "model_id": self.model_id}
        
        num_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "loaded": True,
            "model_id": self.model_id,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "num_parameters": num_params,
            "num_parameters_human": f"{num_params / 1e6:.1f}M" if num_params < 1e9 else f"{num_params / 1e9:.1f}B",
            "vocab_size": len(self.tokenizer),
            "max_length": getattr(self.model.config, "max_position_embeddings", "unknown"),
        }
    
    @classmethod
    def list_suggested_models(cls) -> dict[str, str]:
        """List suggested models by category."""
        return cls.SUGGESTED_MODELS.copy()


class HuggingFaceEngine:
    """
    High-level inference engine compatible with EnigmaEngine interface.
    
    Drop-in replacement for EnigmaEngine using HuggingFace models.
    """
    
    def __init__(
        self,
        model_id: str = "gpt2",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize HuggingFace engine.
        
        Args:
            model_id: HuggingFace model ID or path
            device: Device to use
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
        """
        self.hf_model = HuggingFaceModel(
            model_id=model_id,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
        self.hf_model.load()
        
        # Chat history for stateful conversations
        self.chat_history: list[dict[str, str]] = []
        self._max_chat_history: int = 100
    
    def generate(
        self,
        prompt: str,
        max_gen: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        return self.hf_model.generate(
            prompt,
            max_new_tokens=max_gen,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    
    def stream_generate(
        self,
        prompt: str,
        max_gen: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream generate text."""
        yield from self.hf_model.stream_generate(
            prompt,
            max_new_tokens=max_gen,
            temperature=temperature,
        )
    
    def chat(
        self,
        message: str,
        max_gen: int = 200,
        temperature: float = 0.7,
        reset_history: bool = False,
        **kwargs
    ) -> str:
        """Chat with history tracking."""
        if reset_history:
            self.chat_history = []
        
        response = self.hf_model.chat(
            message,
            history=self.chat_history,
            max_new_tokens=max_gen,
            temperature=temperature,
        )
        
        # Update history
        self.chat_history.append({"role": "user", "content": message})
        self.chat_history.append({"role": "assistant", "content": response})
        
        # Trim history if too long
        if len(self.chat_history) > self._max_chat_history:
            self.chat_history = self.chat_history[-self._max_chat_history:]
        
        return response
    
    def chat_with_tools(
        self,
        message: str,
        max_gen: int = 200,
        temperature: float = 0.7,
        reset_history: bool = False,
        **kwargs
    ) -> str:
        """
        Chat with automatic tool routing based on user intent.
        
        Uses the UniversalToolRouter which works with ANY model.
        Tools are triggered by keywords in the user message.
        
        Args:
            message: User message
            max_gen: Max tokens for response
            temperature: Sampling temperature
            reset_history: Clear chat history
            
        Returns:
            Response (either from tool or from model)
        """
        if reset_history:
            self.chat_history = []
        
        # Use the universal router that works with any model
        from .universal_router import chat_with_tools

        # Create a chat function that includes history
        def chat_fn(msg, **kw):
            return self.chat(msg, max_gen=max_gen, temperature=temperature, 
                           reset_history=False, **kw)
        
        response = chat_with_tools(message, chat_fn, **kwargs)
        
        # Update history
        self.chat_history.append({"role": "user", "content": message})
        self.chat_history.append({"role": "assistant", "content": response})
        
        # Trim history if too long
        if len(self.chat_history) > self._max_chat_history:
            self.chat_history = self.chat_history[-self._max_chat_history:]
        
        return response
    
    def reset_history(self):
        """Clear chat history."""
        self.chat_history = []


# Convenience function
def load_huggingface_model(
    model_id: str = "gpt2",
    **kwargs
) -> HuggingFaceModel:
    """
    Load a HuggingFace model.
    
    Args:
        model_id: Model ID or path
        **kwargs: Additional arguments for HuggingFaceModel
        
    Returns:
        Loaded HuggingFaceModel instance
    """
    model = HuggingFaceModel(model_id, **kwargs)
    model.load()
    return model


def convert_hf_config_to_forge(hf_config) -> dict:
    """
    Convert HuggingFace config to Forge config dictionary.
    
    Supports various HuggingFace model architectures:
    - GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
    - GPT-Neo/GPT-J (EleutherAI models)
    - GPT-NeoX (EleutherAI models)
    - LLaMA family (Meta LLaMA, LLaMA-2)
    - Mistral (Mistral-7B, Mixtral)
    - Phi models (Microsoft Phi-1, Phi-2)
    
    Args:
        hf_config: HuggingFace model config object
        
    Returns:
        Dictionary with Forge config parameters
        
    Raises:
        ValueError: If model type not supported
    """
    model_type = getattr(hf_config, 'model_type', 'unknown')
    
    logger.info(f"Converting HuggingFace config: model_type={model_type}")
    
    # Base config that works for most models
    config = {}
    
    # Vocabulary size (universal)
    config['vocab_size'] = hf_config.vocab_size
    
    # Model dimension (different attribute names)
    if hasattr(hf_config, 'hidden_size'):
        config['dim'] = hf_config.hidden_size
    elif hasattr(hf_config, 'n_embd'):
        config['dim'] = hf_config.n_embd
    elif hasattr(hf_config, 'd_model'):
        config['dim'] = hf_config.d_model
    else:
        raise ValueError("Cannot find model dimension in config")
    
    # Number of layers
    if hasattr(hf_config, 'num_hidden_layers'):
        config['n_layers'] = hf_config.num_hidden_layers
    elif hasattr(hf_config, 'n_layer'):
        config['n_layers'] = hf_config.n_layer
    elif hasattr(hf_config, 'num_layers'):
        config['n_layers'] = hf_config.num_layers
    else:
        raise ValueError("Cannot find number of layers in config")
    
    # Number of attention heads
    if hasattr(hf_config, 'num_attention_heads'):
        config['n_heads'] = hf_config.num_attention_heads
    elif hasattr(hf_config, 'n_head'):
        config['n_heads'] = hf_config.n_head
    else:
        raise ValueError("Cannot find number of attention heads in config")
    
    # Maximum sequence length
    if hasattr(hf_config, 'max_position_embeddings'):
        config['max_seq_len'] = hf_config.max_position_embeddings
    elif hasattr(hf_config, 'n_positions'):
        config['max_seq_len'] = hf_config.n_positions
    elif hasattr(hf_config, 'max_sequence_length'):
        config['max_seq_len'] = hf_config.max_sequence_length
    else:
        config['max_seq_len'] = 2048  # Default fallback
        logger.warning("Cannot find max_seq_len in config, using default 2048")
    
    # Model-specific configurations
    if model_type in ['gpt2', 'gpt_neo', 'gpt_neox']:
        # GPT-2 family - standard transformer
        pass  # Base config is sufficient
    
    elif model_type == 'llama':
        # LLaMA family - has GQA
        if hasattr(hf_config, 'num_key_value_heads'):
            config['n_kv_heads'] = hf_config.num_key_value_heads
        
        # LLaMA uses RoPE
        if hasattr(hf_config, 'rope_theta'):
            config['rope_theta'] = hf_config.rope_theta
    
    elif model_type == 'mistral':
        # Mistral - similar to LLaMA but with sliding window
        if hasattr(hf_config, 'num_key_value_heads'):
            config['n_kv_heads'] = hf_config.num_key_value_heads
        
        if hasattr(hf_config, 'sliding_window'):
            config['sliding_window'] = hf_config.sliding_window
        
        if hasattr(hf_config, 'rope_theta'):
            config['rope_theta'] = hf_config.rope_theta
    
    elif model_type in ['phi', 'phi-msft']:
        # Phi models - Microsoft's small efficient models
        # Use partial rotary embeddings
        if hasattr(hf_config, 'partial_rotary_factor'):
            config['partial_rotary_factor'] = hf_config.partial_rotary_factor
    
    else:
        logger.warning(
            f"Model type '{model_type}' not explicitly supported. "
            f"Using generic config mapping - conversion may not be perfect."
        )
    
    logger.info(f"Converted config: {config}")
    return config


def convert_hf_weights_to_forge(hf_state_dict: dict, model_type: str) -> dict:
    """
    Convert HuggingFace weights to Forge format using the weight mapper.
    
    This is a convenience wrapper around WeightMapper.map_huggingface_to_forge()
    that provides a cleaner API for standalone use.
    
    Args:
        hf_state_dict: HuggingFace model state dict
        model_type: Model type (e.g., 'gpt2', 'llama', 'mistral')
        
    Returns:
        Forge-compatible state dict
        
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> hf_weights = hf_model.state_dict()
        >>> forge_weights = convert_hf_weights_to_forge(hf_weights, "gpt2")
    """
    from .weight_mapping import WeightMapper
    
    logger.info(f"Converting {len(hf_state_dict)} HuggingFace weights to Forge format")
    logger.info(f"Model type: {model_type}")
    
    # Create weight mapper
    mapper = WeightMapper()
    
    # Perform mapping
    forge_state_dict = mapper.map_huggingface_to_forge(hf_state_dict)
    
    logger.info(f"Conversion complete: {len(forge_state_dict)} Forge weights")
    
    # Log some statistics
    hf_param_count = sum(t.numel() for t in hf_state_dict.values() if hasattr(t, 'numel'))
    forge_param_count = sum(t.numel() for t in forge_state_dict.values() if hasattr(t, 'numel'))
    
    if hf_param_count > 0:
        coverage = (forge_param_count / hf_param_count) * 100
        logger.info(f"Weight coverage: {coverage:.1f}% ({forge_param_count:,} / {hf_param_count:,} parameters)")
    
    return forge_state_dict


def convert_huggingface_to_forge(
    model_id: str,
    **kwargs
) -> 'Forge':
    """
    Load a HuggingFace model and convert it to Forge format.
    
    This function downloads a HuggingFace model, extracts its weights,
    and maps them to Forge architecture.
    
    Args:
        model_id: HuggingFace model ID (e.g., "gpt2") or local path
        **kwargs: Additional arguments
        
    Returns:
        Forge model with HuggingFace weights loaded
        
    Raises:
        RuntimeError: If transformers not installed
        ValueError: If model incompatible with Forge
    """
    if not HAVE_TRANSFORMERS:
        raise RuntimeError(
            "HuggingFace conversion requires transformers. "
            "Install with: pip install transformers"
        )
    
    # Import Forge components
    from .model import Forge, ForgeConfig
    
    logger.info(f"Loading HuggingFace model: {model_id}")
    
    # Load HuggingFace model and tokenizer
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=kwargs.get('torch_dtype', torch.float32),
            device_map=kwargs.get('device_map', None),
            trust_remote_code=kwargs.get('trust_remote_code', False)
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=kwargs.get('trust_remote_code', False))
        
        logger.info(f"HuggingFace model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load HuggingFace model: {e}")
        raise
    
    # Get HuggingFace config
    hf_config = hf_model.config
    
    # Convert HuggingFace config to Forge config using standalone function
    forge_config_dict = convert_hf_config_to_forge(hf_config)
    
    logger.info(f"Creating Forge config: {forge_config_dict}")
    forge_config = ForgeConfig(**forge_config_dict)
    
    # Create Forge model
    forge_model = Forge(config=forge_config)
    
    # Get HuggingFace state dict
    hf_state_dict = hf_model.state_dict()
    logger.info(f"HuggingFace model has {len(hf_state_dict)} parameters")
    
    # Convert weights using standalone function
    model_type = getattr(hf_config, 'model_type', 'unknown')
    forge_state_dict = convert_hf_weights_to_forge(hf_state_dict, model_type)
    
    # Load weights into Forge model
    logger.info("Loading mapped weights into Forge model...")
    try:
        missing_keys, unexpected_keys = forge_model.load_state_dict(forge_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing {len(missing_keys)} keys - will be randomly initialized")
            logger.debug(f"Missing keys: {missing_keys[:10]}")
        
        if unexpected_keys:
            logger.warning(f"Unexpected {len(unexpected_keys)} keys - will be ignored")
            logger.debug(f"Unexpected keys: {unexpected_keys[:10]}")
        
        logger.info("HuggingFace model successfully converted to Forge format")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        raise ValueError(f"Weight conversion failed: {e}")
    
    # Set to eval mode
    forge_model.eval()
    
    return forge_model


if __name__ == "__main__":
    # Demo - configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logger.info("HuggingFace Model Loader Demo")
    logger.info("=" * 50)
    
    logger.info("Suggested models:")
    for name, model_id in HuggingFaceModel.SUGGESTED_MODELS.items():
        logger.info(f"  {name}: {model_id}")
    
    logger.info("Loading GPT-2...")
    model = HuggingFaceModel("gpt2")
    model.load()
    
    logger.info("Model info:")
    info = model.get_model_info()
    for k, v in info.items():
        logger.info(f"  {k}: {v}")
    
    logger.info("Generating text...")
    prompt = "The future of artificial intelligence is"
    logger.info(f"Prompt: {prompt}")
    response = model.generate(prompt, max_new_tokens=50)
    logger.info(f"Response: {response}")
    
    logger.info("Streaming generation...")
    logger.info("Prompt: Once upon a time")
    logger.info("Response: (streaming tokens)")
    streamed_output = ""
    for token in model.stream_generate("Once upon a time", max_new_tokens=30):
        streamed_output += token
    logger.info(f"Streamed: {streamed_output}")
    
    logger.info("Done!")
