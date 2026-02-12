"""
GPTQ/AWQ Quantized Model Loader
===============================

Load and run GPTQ and AWQ quantized models for efficient inference.
These formats provide high compression with minimal quality loss.

GPTQ: Post-training quantization using optimal brain compression
AWQ: Activation-aware weight quantization (preserves salient weights)

Usage:
    from enigma_engine.core.gptq_awq_loader import GPTQModel, AWQModel
    
    # Load GPTQ model
    model = GPTQModel("path/to/model-gptq")
    model.load()
    response = model.generate("Hello!")
    
    # Load AWQ model  
    model = AWQModel("path/to/model-awq")
    model.load()
    response = model.generate("Hello!")
"""

import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)

# Check for required libraries
HAVE_TORCH = False
HAVE_AUTO_GPTQ = False
HAVE_AWQ = False
HAVE_TRANSFORMERS = False

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    torch = None  # type: ignore

try:
    from transformers import AutoTokenizer, AutoConfig
    HAVE_TRANSFORMERS = True
except ImportError:
    pass  # Intentionally silent

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    HAVE_AUTO_GPTQ = True
except ImportError:
    pass  # Intentionally silent

try:
    from awq import AutoAWQForCausalLM
    HAVE_AWQ = True
except ImportError:
    pass  # Intentionally silent


class QuantizationType(Enum):
    """Quantization format type."""
    GPTQ = "gptq"
    AWQ = "awq"
    AUTO = "auto"  # Auto-detect from model files


@dataclass
class QuantConfig:
    """Quantization configuration."""
    bits: int = 4  # 2, 3, 4, or 8 bits
    group_size: int = 128  # Weight grouping
    desc_act: bool = True  # Descending activation order
    sym: bool = False  # Symmetric quantization
    true_sequential: bool = True  # True sequential quantization
    
    # AWQ specific
    version: str = "gemm"  # "gemm" or "gemv"
    zero_point: bool = True  # Use zero point


@dataclass
class ModelMetadata:
    """Model metadata."""
    name: str = ""
    quant_type: QuantizationType = QuantizationType.AUTO
    bits: int = 4
    group_size: int = 128
    model_type: str = ""  # e.g., "llama", "mistral"
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    context_length: int = 2048
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseQuantizedModel:
    """Base class for quantized model loading."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        trust_remote_code: bool = False,
        use_safetensors: bool = True,
        max_memory: Optional[Dict[int, str]] = None,
        low_cpu_mem_usage: bool = True,
    ):
        """
        Initialize quantized model.
        
        Args:
            model_path: Path to model directory or HuggingFace model ID
            device: Device to load on ("auto", "cuda", "cpu")
            trust_remote_code: Trust remote code for custom models
            use_safetensors: Prefer safetensors format
            max_memory: GPU memory limits per device
            low_cpu_mem_usage: Reduce CPU memory during loading
        """
        if not HAVE_TORCH:
            raise RuntimeError("PyTorch required. Install with: pip install torch")
        if not HAVE_TRANSFORMERS:
            raise RuntimeError("Transformers required. Install with: pip install transformers")
        
        self.model_path = model_path
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.use_safetensors = use_safetensors
        self.max_memory = max_memory
        self.low_cpu_mem_usage = low_cpu_mem_usage
        
        self.model = None
        self.tokenizer = None
        self.config = None
        self.metadata: Optional[ModelMetadata] = None
        self._loaded = False
    
    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        if self.device != "auto":
            return self.device
        
        if HAVE_TORCH and torch.cuda.is_available():
            return "cuda"
        elif HAVE_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_tokenizer(self) -> None:
        """Load tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _extract_metadata(self) -> ModelMetadata:
        """Extract model metadata from config."""
        metadata = ModelMetadata(name=str(self.model_path))
        
        if self.config:
            metadata.vocab_size = getattr(self.config, 'vocab_size', 0)
            metadata.hidden_size = getattr(self.config, 'hidden_size', 0)
            metadata.num_layers = getattr(self.config, 'num_hidden_layers', 0)
            metadata.num_attention_heads = getattr(self.config, 'num_attention_heads', 0)
            metadata.context_length = getattr(self.config, 'max_position_embeddings', 2048)
            metadata.model_type = getattr(self.config, 'model_type', '')
        
        return metadata
    
    def load(self) -> bool:
        """Load the model. Override in subclasses."""
        raise NotImplementedError
    
    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if HAVE_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._loaded = False
        logger.info(f"Unloaded model: {self.model_path}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            repetition_penalty: Penalty for repetition
            do_sample: Use sampling vs greedy
            **kwargs: Additional generation arguments
        
        Returns:
            Generated text
        """
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Move to device
        device = self._detect_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode, removing input prompt
        input_length = inputs['input_ids'].shape[1]
        generated = outputs[0][input_length:]
        result = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return result
    
    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Generate text with streaming output.
        
        Yields tokens as they're generated.
        """
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self._detect_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Generation kwargs
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "streamer": streamer,
            "do_sample": temperature > 0,
            **kwargs
        }
        
        # Run generation in thread
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        # Yield tokens
        for text in streamer:
            yield text
        
        thread.join()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Chat completion with message history.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            max_new_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            **kwargs: Generation arguments
        
        Returns:
            Assistant response
        """
        # Format as chat
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Use chat template if available
            full_messages = messages.copy()
            if system_prompt:
                full_messages.insert(0, {"role": "system", "content": system_prompt})
            
            prompt = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Manual formatting
            prompt = ""
            if system_prompt:
                prompt += f"System: {system_prompt}\n\n"
            for msg in messages:
                role = msg.get("role", "user").title()
                content = msg.get("content", "")
                prompt += f"{role}: {content}\n"
            prompt += "Assistant:"
        
        return self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "path": str(self.model_path),
            "loaded": self._loaded,
            "device": self._detect_device(),
        }
        
        if self.metadata:
            info.update({
                "name": self.metadata.name,
                "type": self.metadata.model_type,
                "quant_type": self.metadata.quant_type.value,
                "bits": self.metadata.bits,
                "group_size": self.metadata.group_size,
                "vocab_size": self.metadata.vocab_size,
                "hidden_size": self.metadata.hidden_size,
                "num_layers": self.metadata.num_layers,
                "context_length": self.metadata.context_length,
            })
        
        return info


class GPTQModel(BaseQuantizedModel):
    """
    GPTQ quantized model loader.
    
    GPTQ (Generalized Post-Training Quantization) uses optimal brain 
    compression techniques for efficient quantization with minimal
    accuracy loss.
    
    Requires: pip install auto-gptq
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        use_triton: bool = False,
        inject_fused_attention: bool = True,
        inject_fused_mlp: bool = True,
        disable_exllama: bool = False,
        disable_exllamav2: bool = False,
        **kwargs
    ):
        """
        Initialize GPTQ model.
        
        Args:
            model_path: Path to GPTQ model
            device: Device to load on
            use_triton: Use Triton kernels (faster on some GPUs)
            inject_fused_attention: Use fused attention
            inject_fused_mlp: Use fused MLP layers
            disable_exllama: Disable ExLlama kernels
            disable_exllamav2: Disable ExLlamaV2 kernels
        """
        if not HAVE_AUTO_GPTQ:
            raise RuntimeError(
                "GPTQ loading requires auto-gptq. "
                "Install with: pip install auto-gptq"
            )
        
        super().__init__(model_path, device, **kwargs)
        
        self.use_triton = use_triton
        self.inject_fused_attention = inject_fused_attention
        self.inject_fused_mlp = inject_fused_mlp
        self.disable_exllama = disable_exllama
        self.disable_exllamav2 = disable_exllamav2
        self.quant_config: Optional[BaseQuantizeConfig] = None
    
    def _detect_gptq_config(self) -> Optional[Dict[str, Any]]:
        """Detect GPTQ configuration from model files."""
        model_dir = Path(self.model_path)
        
        # Check for quantize_config.json
        config_path = model_dir / "quantize_config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        
        # Check for config.json with quantization_config
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                if "quantization_config" in config:
                    return config["quantization_config"]
        
        return None
    
    def load(self) -> bool:
        """Load GPTQ model."""
        try:
            logger.info(f"Loading GPTQ model from: {self.model_path}")
            
            # Load config
            self.config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
            )
            
            # Detect quantization config
            gptq_config = self._detect_gptq_config()
            if gptq_config:
                logger.info(f"GPTQ config: {gptq_config.get('bits', 4)}-bit, "
                           f"group_size={gptq_config.get('group_size', 128)}")
            
            # Load tokenizer
            self._load_tokenizer()
            
            # Determine device
            device = self._detect_device()
            device_map = "auto" if device == "cuda" else device
            
            # Load GPTQ model
            self.model = AutoGPTQForCausalLM.from_quantized(
                self.model_path,
                device_map=device_map,
                trust_remote_code=self.trust_remote_code,
                use_safetensors=self.use_safetensors,
                use_triton=self.use_triton,
                inject_fused_attention=self.inject_fused_attention,
                inject_fused_mlp=self.inject_fused_mlp,
                disable_exllama=self.disable_exllama,
                disable_exllamav2=self.disable_exllamav2,
                max_memory=self.max_memory,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
            )
            
            # Extract metadata
            self.metadata = self._extract_metadata()
            self.metadata.quant_type = QuantizationType.GPTQ
            if gptq_config:
                self.metadata.bits = gptq_config.get('bits', 4)
                self.metadata.group_size = gptq_config.get('group_size', 128)
            
            self._loaded = True
            logger.info(f"Successfully loaded GPTQ model on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load GPTQ model: {e}")
            raise
    
    @staticmethod
    def quantize(
        model_path: str,
        output_path: str,
        calibration_data: List[str],
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = True,
        sym: bool = False,
        batch_size: int = 1,
    ) -> str:
        """
        Quantize a model to GPTQ format.
        
        Args:
            model_path: Path to original model
            output_path: Where to save quantized model
            calibration_data: Sample texts for calibration
            bits: Quantization bits (2, 3, 4, 8)
            group_size: Weight grouping size
            desc_act: Descending activation order
            sym: Symmetric quantization
            batch_size: Calibration batch size
        
        Returns:
            Path to quantized model
        """
        if not HAVE_AUTO_GPTQ:
            raise RuntimeError("auto-gptq required for quantization")
        
        from transformers import AutoModelForCausalLM
        
        logger.info(f"Quantizing model to {bits}-bit GPTQ...")
        
        # Load original model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Create quantization config
        quant_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
        )
        
        # Prepare calibration dataset
        def prepare_data(examples):
            return tokenizer(
                examples,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            )
        
        # Quantize
        quantized_model = AutoGPTQForCausalLM.from_pretrained(
            model,
            quant_config,
        )
        
        # Calibrate and quantize
        quantized_model.quantize(
            calibration_data,
            batch_size=batch_size,
        )
        
        # Save
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        quantized_model.save_quantized(str(output_path), use_safetensors=True)
        tokenizer.save_pretrained(str(output_path))
        
        logger.info(f"Quantized model saved to: {output_path}")
        return str(output_path)


class AWQModel(BaseQuantizedModel):
    """
    AWQ (Activation-aware Weight Quantization) model loader.
    
    AWQ preserves salient weights by analyzing activation patterns,
    providing better quality than uniform quantization.
    
    Requires: pip install autoawq
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        fuse_layers: bool = True,
        max_seq_len: int = 2048,
        **kwargs
    ):
        """
        Initialize AWQ model.
        
        Args:
            model_path: Path to AWQ model
            device: Device to load on
            fuse_layers: Use fused layers for speed
            max_seq_len: Maximum sequence length
        """
        if not HAVE_AWQ:
            raise RuntimeError(
                "AWQ loading requires autoawq. "
                "Install with: pip install autoawq"
            )
        
        super().__init__(model_path, device, **kwargs)
        
        self.fuse_layers = fuse_layers
        self.max_seq_len = max_seq_len
    
    def _detect_awq_config(self) -> Optional[Dict[str, Any]]:
        """Detect AWQ configuration from model files."""
        model_dir = Path(self.model_path)
        
        # Check for quant_config.json
        for config_name in ["quant_config.json", "quantize_config.json"]:
            config_path = model_dir / config_name
            if config_path.exists():
                with open(config_path) as f:
                    return json.load(f)
        
        # Check config.json
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                if "quantization_config" in config:
                    qc = config["quantization_config"]
                    if qc.get("quant_method") == "awq":
                        return qc
        
        return None
    
    def load(self) -> bool:
        """Load AWQ model."""
        try:
            logger.info(f"Loading AWQ model from: {self.model_path}")
            
            # Load config first
            self.config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
            )
            
            # Detect AWQ config
            awq_config = self._detect_awq_config()
            if awq_config:
                logger.info(f"AWQ config: {awq_config.get('bits', 4)}-bit, "
                           f"group_size={awq_config.get('group_size', 128)}")
            
            # Load tokenizer
            self._load_tokenizer()
            
            # Determine device
            device = self._detect_device()
            
            # Load AWQ model
            self.model = AutoAWQForCausalLM.from_quantized(
                self.model_path,
                fuse_layers=self.fuse_layers,
                trust_remote_code=self.trust_remote_code,
                safetensors=self.use_safetensors,
                max_seq_len=self.max_seq_len,
            )
            
            # Move to device if needed
            if device == "cuda" and hasattr(self.model, 'to'):
                self.model = self.model.to(device)
            
            # Extract metadata
            self.metadata = self._extract_metadata()
            self.metadata.quant_type = QuantizationType.AWQ
            if awq_config:
                self.metadata.bits = awq_config.get('bits', 4)
                self.metadata.group_size = awq_config.get('group_size', 128)
            
            self._loaded = True
            logger.info(f"Successfully loaded AWQ model on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load AWQ model: {e}")
            raise
    
    @staticmethod
    def quantize(
        model_path: str,
        output_path: str,
        calibration_data: List[str],
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        version: str = "gemm",
    ) -> str:
        """
        Quantize a model to AWQ format.
        
        Args:
            model_path: Path to original model
            output_path: Where to save quantized model
            calibration_data: Sample texts for calibration
            bits: Quantization bits (4)
            group_size: Weight grouping size
            zero_point: Use zero point quantization
            version: AWQ version ("gemm" or "gemv")
        
        Returns:
            Path to quantized model
        """
        if not HAVE_AWQ:
            raise RuntimeError("autoawq required for quantization")
        
        logger.info(f"Quantizing model to {bits}-bit AWQ...")
        
        # Load model
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Quantization config
        quant_config = {
            "zero_point": zero_point,
            "q_group_size": group_size,
            "w_bit": bits,
            "version": version,
        }
        
        # Quantize
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calibration_data,
        )
        
        # Save
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        logger.info(f"Quantized model saved to: {output_path}")
        return str(output_path)


def detect_quantization_type(model_path: str) -> Optional[QuantizationType]:
    """
    Auto-detect quantization type from model files.
    
    Args:
        model_path: Path to model directory
    
    Returns:
        Detected quantization type or None
    """
    model_dir = Path(model_path)
    
    # Check for explicit config files
    if (model_dir / "quantize_config.json").exists():
        with open(model_dir / "quantize_config.json") as f:
            config = json.load(f)
            if "quant_method" in config:
                method = config["quant_method"].lower()
                if method == "awq":
                    return QuantizationType.AWQ
                elif method == "gptq":
                    return QuantizationType.GPTQ
            # GPTQ configs typically have 'bits' and 'group_size' without method
            if "bits" in config and "group_size" in config:
                return QuantizationType.GPTQ
    
    # Check config.json
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            if "quantization_config" in config:
                qc = config["quantization_config"]
                method = qc.get("quant_method", "").lower()
                if method == "awq":
                    return QuantizationType.AWQ
                elif method == "gptq":
                    return QuantizationType.GPTQ
    
    # Check model name patterns
    path_str = str(model_path).lower()
    if "awq" in path_str:
        return QuantizationType.AWQ
    elif "gptq" in path_str:
        return QuantizationType.GPTQ
    
    return None


def load_quantized_model(
    model_path: str,
    quant_type: Optional[QuantizationType] = None,
    **kwargs
) -> Union[GPTQModel, AWQModel]:
    """
    Load a quantized model, auto-detecting format if needed.
    
    Args:
        model_path: Path to quantized model
        quant_type: Force specific quantization type (auto-detect if None)
        **kwargs: Additional arguments for model loader
    
    Returns:
        Loaded quantized model (GPTQModel or AWQModel)
    
    Example:
        model = load_quantized_model("./models/llama-7b-awq")
        response = model.generate("Hello!")
    """
    # Auto-detect if not specified
    if quant_type is None or quant_type == QuantizationType.AUTO:
        quant_type = detect_quantization_type(model_path)
        
        if quant_type is None:
            raise ValueError(
                f"Could not auto-detect quantization type for: {model_path}. "
                "Specify quant_type explicitly."
            )
    
    # Load appropriate model
    if quant_type == QuantizationType.GPTQ:
        model = GPTQModel(model_path, **kwargs)
    elif quant_type == QuantizationType.AWQ:
        model = AWQModel(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown quantization type: {quant_type}")
    
    model.load()
    return model


class QuantizedModelRegistry:
    """
    Registry for managing multiple quantized models.
    
    Supports lazy loading and caching for efficient memory use.
    """
    
    def __init__(self, max_loaded: int = 2):
        """
        Initialize registry.
        
        Args:
            max_loaded: Maximum models to keep loaded simultaneously
        """
        self._models: Dict[str, Union[GPTQModel, AWQModel]] = {}
        self._load_order: List[str] = []  # LRU tracking
        self.max_loaded = max_loaded
    
    def register(
        self,
        name: str,
        model_path: str,
        quant_type: Optional[QuantizationType] = None,
        **kwargs
    ) -> None:
        """Register a model without loading it."""
        if quant_type is None or quant_type == QuantizationType.AUTO:
            quant_type = detect_quantization_type(model_path)
        
        if quant_type == QuantizationType.GPTQ:
            model = GPTQModel(model_path, **kwargs)
        elif quant_type == QuantizationType.AWQ:
            model = AWQModel(model_path, **kwargs)
        else:
            # Default to GPTQ
            model = GPTQModel(model_path, **kwargs)
        
        self._models[name] = model
    
    def get(self, name: str) -> Union[GPTQModel, AWQModel]:
        """
        Get a model by name, loading if necessary.
        
        Uses LRU eviction when max_loaded exceeded.
        """
        if name not in self._models:
            raise KeyError(f"Model not registered: {name}")
        
        model = self._models[name]
        
        # Load if needed
        if not model._loaded:
            # Evict oldest if at capacity
            loaded_count = sum(1 for m in self._models.values() if m._loaded)
            while loaded_count >= self.max_loaded:
                # Find oldest loaded model
                for old_name in self._load_order:
                    if old_name in self._models and self._models[old_name]._loaded:
                        self._models[old_name].unload()
                        self._load_order.remove(old_name)
                        loaded_count -= 1
                        break
                else:
                    break
            
            model.load()
            self._load_order.append(name)
        else:
            # Move to end (most recently used)
            if name in self._load_order:
                self._load_order.remove(name)
            self._load_order.append(name)
        
        return model
    
    def unload(self, name: str) -> None:
        """Unload a specific model."""
        if name in self._models and self._models[name]._loaded:
            self._models[name].unload()
            if name in self._load_order:
                self._load_order.remove(name)
    
    def unload_all(self) -> None:
        """Unload all models."""
        for model in self._models.values():
            if model._loaded:
                model.unload()
        self._load_order.clear()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models and their status."""
        return [
            {
                "name": name,
                "loaded": model._loaded,
                "path": model.model_path,
                "type": model.metadata.quant_type.value if model.metadata else "unknown",
            }
            for name, model in self._models.items()
        ]


# Convenience singleton
_registry: Optional[QuantizedModelRegistry] = None

def get_quantized_registry(max_loaded: int = 2) -> QuantizedModelRegistry:
    """Get or create the global quantized model registry."""
    global _registry
    if _registry is None:
        _registry = QuantizedModelRegistry(max_loaded=max_loaded)
    return _registry


# Export public API
__all__ = [
    'GPTQModel',
    'AWQModel',
    'QuantizationType',
    'QuantConfig',
    'ModelMetadata',
    'load_quantized_model',
    'detect_quantization_type',
    'QuantizedModelRegistry',
    'get_quantized_registry',
    'HAVE_AUTO_GPTQ',
    'HAVE_AWQ',
]
