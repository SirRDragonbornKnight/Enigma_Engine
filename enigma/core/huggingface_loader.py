"""
HuggingFace Model Loader
========================

Load and run HuggingFace Transformers models for text generation.
Enables using pre-trained models like GPT-2, Llama, Mistral, etc.

Usage:
    from enigma.core.huggingface_loader import HuggingFaceModel
    
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
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Generator, Union, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Type hints for optional dependencies
if TYPE_CHECKING:
    import torch as torch_types
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
    from transformers.generation.configuration_utils import GenerationConfig
    from transformers.utils.quantization_config import BitsAndBytesConfig
    HAVE_TRANSFORMERS = True
except ImportError:
    logger.warning("transformers not available - HuggingFace loading disabled")

# Check for threading (for streaming)
from threading import Thread


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
    }
    
    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        max_memory: Optional[Dict] = None,
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
            model_kwargs: Dict[str, Any] = {
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
        stop_strings: Optional[List[str]] = None,
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
            
        Returns:
            Generated text (excluding prompt)
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
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        generated_ids = outputs[0][input_ids.shape[1]:]
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
        for token in streamer:
            yield token
        
        thread.join()
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
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
        
        # Type assertions for loaded state
        assert self.tokenizer is not None, "Tokenizer not loaded"
        assert self.model is not None, "Model not loaded"
        
        # Build conversation
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
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
            else:
                # Fallback: simple formatting
                prompt = self._format_chat_simple(messages)
        except Exception:
            prompt = self._format_chat_simple(messages)
        
        # Generate
        response = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_strings=["User:", "Human:", "\n\nUser", "\n\nHuman"],
        )
        
        return response.strip()
    
    def _format_chat_simple(self, messages: List[Dict[str, str]]) -> str:
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
    
    def get_model_info(self) -> Dict[str, Any]:
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
    def list_suggested_models(cls) -> Dict[str, str]:
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
        self.chat_history: List[Dict[str, str]] = []
    
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


if __name__ == "__main__":
    # Demo
    print("HuggingFace Model Loader Demo")
    print("=" * 50)
    
    print("\nSuggested models:")
    for name, model_id in HuggingFaceModel.SUGGESTED_MODELS.items():
        print(f"  {name}: {model_id}")
    
    print("\nLoading GPT-2...")
    model = HuggingFaceModel("gpt2")
    model.load()
    
    print("\nModel info:")
    info = model.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    print("\nGenerating text...")
    prompt = "The future of artificial intelligence is"
    print(f"Prompt: {prompt}")
    response = model.generate(prompt, max_new_tokens=50)
    print(f"Response: {response}")
    
    print("\nStreaming generation...")
    print("Prompt: Once upon a time")
    print("Response: ", end="")
    for token in model.stream_generate("Once upon a time", max_new_tokens=30):
        print(token, end="", flush=True)
    print("\n")
    
    print("Done!")
