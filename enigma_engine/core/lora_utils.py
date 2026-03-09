"""
LoRA/QLoRA Training Utilities for Enigma AI Engine

Provides:
- LoraConfig: Configuration for LoRA adapters
- QLoraConfig: Configuration for 4-bit quantized LoRA
- OffloadConfig: Configuration for CPU offloading
- LoraTrainer: Training class for LoRA/QLoRA fine-tuning
- Memory management utilities for VRAM and RAM

LoRA = Low-Rank Adaptation - trains small adapter weights instead of full model
QLoRA = Quantized LoRA - uses 4-bit quantization to reduce VRAM by ~75%
CPU Offload = Stores optimizer states and some weights in system RAM

Usage:
    from enigma_engine.core.lora_utils import LoraConfig, LoraTrainer
    
    config = LoraConfig(rank=8, alpha=16)
    trainer = LoraTrainer(model, tokenizer, lora_config=config)
    trainer.train(data)
    trainer.save_adapter("my_lora.pth")
"""

import gc
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Optional imports for PEFT and quantization
PEFT_AVAILABLE = False
BITSANDBYTES_AVAILABLE = False
ACCELERATE_AVAILABLE = False

try:
    from peft import (
        LoraConfig as PeftLoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
    PEFT_AVAILABLE = True
except ImportError:
    logger.debug("PEFT not installed - install with: pip install peft")

try:
    import bitsandbytes as bnb  # noqa: F401
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    logger.debug("bitsandbytes not installed - install with: pip install bitsandbytes")

try:
    from accelerate import Accelerator, dispatch_model, infer_auto_device_map  # noqa: F401
    ACCELERATE_AVAILABLE = True
except ImportError:
    logger.debug("accelerate not installed - install with: pip install accelerate")


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

def clear_vram() -> None:
    """
    Clear GPU VRAM by emptying PyTorch cache.
    
    Call this when VRAM is full to free cached memory.
    Does NOT unload models - just releases cached tensors.
    
    Example:
        clear_vram()  # Free up VRAM
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("VRAM cache cleared")

    # Also run Python garbage collection
    gc.collect()


def get_memory_info() -> Dict[str, float]:
    """
    Get current memory usage info for both RAM and VRAM.
    
    Returns:
        Dict with memory stats in GB:
        - ram_total_gb: Total system RAM
        - ram_available_gb: Available RAM
        - ram_used_gb: Used RAM
        - vram_total_gb: Total GPU VRAM (0 if no GPU)
        - vram_used_gb: Used GPU VRAM
        - vram_available_gb: Available VRAM
    
    Example:
        info = get_memory_info()
        print(f"RAM: {info['ram_available_gb']:.1f}GB free")
        print(f"VRAM: {info['vram_available_gb']:.1f}GB free")
    """
    import psutil

    # System RAM
    ram = psutil.virtual_memory()
    info = {
        'ram_total_gb': ram.total / (1024 ** 3),
        'ram_available_gb': ram.available / (1024 ** 3),
        'ram_used_gb': ram.used / (1024 ** 3),
        'vram_total_gb': 0.0,
        'vram_used_gb': 0.0,
        'vram_available_gb': 0.0,
    }

    # GPU VRAM (if available)
    if torch.cuda.is_available():
        try:
            # Get VRAM info for default device
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)

            info['vram_total_gb'] = props.total_memory / (1024 ** 3)
            info['vram_used_gb'] = torch.cuda.memory_allocated(device) / (1024 ** 3)
            info['vram_available_gb'] = info['vram_total_gb'] - info['vram_used_gb']
        except Exception as e:
            logger.debug(f"Could not get VRAM info: {e}")

    return info


def estimate_training_memory(
    model_params: int,
    batch_size: int = 1,
    seq_length: int = 512,
    use_lora: bool = True,
    lora_rank: int = 8,
    use_qlora: bool = False,
    gradient_checkpointing: bool = False
) -> Dict[str, float]:
    """
    Estimate memory requirements for training.
    
    Args:
        model_params: Number of parameters in model
        batch_size: Training batch size
        seq_length: Sequence length
        use_lora: Whether using LoRA (much less memory)
        lora_rank: LoRA rank (higher = more memory)
        use_qlora: Whether using 4-bit quantization
        gradient_checkpointing: Whether using gradient checkpointing
    
    Returns:
        Dict with estimated memory in GB:
        - model_memory_gb: Memory for model weights
        - optimizer_memory_gb: Memory for optimizer states
        - gradient_memory_gb: Memory for gradients
        - activation_memory_gb: Memory for activations
        - total_vram_gb: Total VRAM needed
        - recommended_ram_gb: Recommended system RAM for offloading
    
    Example:
        mem = estimate_training_memory(7_000_000_000, use_qlora=True)
        print(f"Need ~{mem['total_vram_gb']:.1f}GB VRAM")
    """
    bytes_per_param = 4  # fp32

    if use_qlora:
        bytes_per_param = 0.5  # 4-bit = 0.5 bytes
    elif use_lora:
        bytes_per_param = 2  # fp16 for base model

    # Model memory
    model_memory = model_params * bytes_per_param

    # LoRA adds small number of trainable params
    if use_lora:
        # Rough estimate: LoRA adds ~0.1-1% trainable params
        lora_params = int(model_params * 0.01 * (lora_rank / 8))
        model_memory += lora_params * 4  # LoRA weights in fp32

    # Optimizer states (AdamW needs 2x params for momentum + variance)
    if use_lora:
        optimizer_memory = lora_params * 4 * 2  # Only LoRA params
    else:
        optimizer_memory = model_params * 4 * 2  # Full model

    # Gradients
    if use_lora:
        gradient_memory = lora_params * 4
    else:
        gradient_memory = model_params * 4

    # Activations (rough estimate based on batch size and seq length)
    activation_memory = batch_size * seq_length * 768 * 4  # Assume hidden_size 768
    if gradient_checkpointing:
        activation_memory *= 0.3  # Checkpointing reduces by ~70%

    # Convert to GB
    total_vram = (model_memory + gradient_memory + activation_memory) / (1024 ** 3)
    optimizer_gb = optimizer_memory / (1024 ** 3)

    return {
        'model_memory_gb': model_memory / (1024 ** 3),
        'optimizer_memory_gb': optimizer_gb,
        'gradient_memory_gb': gradient_memory / (1024 ** 3),
        'activation_memory_gb': activation_memory / (1024 ** 3),
        'total_vram_gb': total_vram,
        'recommended_ram_gb': optimizer_gb + 4,  # Optimizer + buffer
    }


# =============================================================================
# LORA CONFIGURATION
# =============================================================================

@dataclass
class LoraConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation) training.
    
    LoRA adds small trainable matrices to attention layers instead of
    fine-tuning the entire model. This drastically reduces memory usage
    and training time.
    
    Attributes:
        rank: Rank of the low-rank matrices (4-64, default 8)
              Lower = faster + less memory, Higher = more capacity
        alpha: Scaling factor (usually 2x rank)
               Controls how much LoRA affects output
        dropout: Dropout rate for LoRA layers (0.0-0.3)
        target_modules: Which modules to add LoRA to
                       Default targets attention projections
        bias: How to handle bias ("none", "all", "lora_only")
        task_type: Type of task ("CAUSAL_LM", "SEQ_CLS", etc.)
    
    Example:
        config = LoraConfig(rank=16, alpha=32)  # Higher capacity
        config = LoraConfig(rank=4, alpha=8)    # Faster, less memory
    """
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_peft_config(self):
        """Convert to PEFT LoraConfig."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT not installed. Run: pip install peft")

        return PeftLoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM if self.task_type == "CAUSAL_LM" else self.task_type,
        )


@dataclass
class QLoraConfig(LoraConfig):
    """
    Configuration for QLoRA (Quantized LoRA) training.
    
    QLoRA uses 4-bit quantization for the base model, reducing VRAM
    usage by ~75% compared to full precision. Only LoRA weights are
    trained in full precision.
    
    Attributes:
        All from LoraConfig, plus:
        load_in_4bit: Enable 4-bit quantization (default True)
        bnb_4bit_quant_type: Quantization type ("nf4" or "fp4")
        bnb_4bit_compute_dtype: Compute dtype (bfloat16 or float16)
        bnb_4bit_use_double_quant: Use double quantization (saves more memory)
    
    Example:
        config = QLoraConfig(rank=16)  # 4-bit base + 16-rank LoRA
    """
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    def to_bnb_config(self):
        """Convert to BitsAndBytesConfig for quantization."""
        if not BITSANDBYTES_AVAILABLE:
            raise ImportError("bitsandbytes not installed. Run: pip install bitsandbytes")

        compute_dtype = torch.bfloat16 if self.bnb_4bit_compute_dtype == "bfloat16" else torch.float16

        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )


@dataclass
class OffloadConfig:
    """
    Configuration for CPU offloading during training.
    
    When VRAM is limited, offloading stores optimizer states and
    some model weights in system RAM. This is slower but allows
    training larger models than would fit in VRAM alone.
    
    Attributes:
        cpu_offload: Offload some layers to CPU (default True)
        offload_optimizer: Offload optimizer states to CPU (default True)
        gradient_checkpointing: Trade compute for memory (default True)
        pin_memory: Pin CPU memory for faster GPU transfers
        max_memory_gpu: Max GB to use on GPU (None = auto)
        max_memory_cpu: Max GB to use on CPU (None = auto)
    
    Example:
        # For 8GB VRAM GPU with 32GB RAM:
        config = OffloadConfig(max_memory_gpu=7, max_memory_cpu=24)
    """
    cpu_offload: bool = True
    offload_optimizer: bool = True
    gradient_checkpointing: bool = True
    pin_memory: bool = True
    max_memory_gpu: Optional[float] = None
    max_memory_cpu: Optional[float] = None


# =============================================================================
# LORA MODEL CREATION
# =============================================================================

def create_lora_model(
    model: nn.Module,
    config: Optional[LoraConfig] = None
) -> nn.Module:
    """
    Wrap a model with LoRA adapters.
    
    Args:
        model: Base PyTorch model
        config: LoRA configuration (uses defaults if None)
    
    Returns:
        Model with LoRA adapters added
    
    Example:
        model = load_model("my_model.pth")
        lora_model = create_lora_model(model, LoraConfig(rank=16))
        # Now train lora_model - only LoRA weights update
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT not installed. Run: pip install peft")

    config = config or LoraConfig()
    peft_config = config.to_peft_config()

    logger.info(f"Creating LoRA model with rank={config.rank}, alpha={config.alpha}")

    # Wrap model with PEFT
    lora_model = get_peft_model(model, peft_config)

    # Log trainable params
    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())
    logger.info(f"LoRA trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return lora_model


def create_qlora_model(
    model: nn.Module,
    config: Optional[QLoraConfig] = None
) -> nn.Module:
    """
    Wrap a model with QLoRA (4-bit quantized LoRA).
    
    QLoRA quantizes the base model to 4-bit while keeping LoRA
    weights in full precision. This reduces VRAM by ~75%.
    
    Args:
        model: Base PyTorch model
        config: QLoRA configuration (uses defaults if None)
    
    Returns:
        Quantized model with LoRA adapters
    
    Example:
        model = load_model("my_model.pth")
        qlora_model = create_qlora_model(model, QLoraConfig(rank=16))
        # 4-bit base + full precision LoRA
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT not installed. Run: pip install peft")
    if not BITSANDBYTES_AVAILABLE:
        raise ImportError("bitsandbytes not installed. Run: pip install bitsandbytes")

    config = config or QLoraConfig()

    logger.info(f"Creating QLoRA model (4-bit) with rank={config.rank}")

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    peft_config = config.to_peft_config()
    qlora_model = get_peft_model(model, peft_config)

    # Log memory savings
    trainable = sum(p.numel() for p in qlora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in qlora_model.parameters())
    logger.info(f"QLoRA trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return qlora_model


# =============================================================================
# LORA WEIGHT MANAGEMENT
# =============================================================================

def load_lora_weights(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load LoRA adapter weights from file.
    
    Args:
        path: Path to LoRA weights file (.pth or .safetensors)
    
    Returns:
        Dictionary of LoRA weight tensors
    
    Example:
        weights = load_lora_weights("my_adapter.pth")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {path}")

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            weights = load_file(str(path))
        except ImportError:
            raise ImportError("safetensors required for .safetensors files") from None
    else:
        weights = torch.load(path, map_location="cpu", weights_only=True)

    logger.info(f"Loaded LoRA weights from: {path} ({len(weights)} tensors)")
    return weights


def apply_lora(
    model: nn.Module,
    lora_weights: Dict[str, torch.Tensor],
    adapter_name: str = "default",
    merge: bool = False
) -> None:
    """
    Apply LoRA weights to a model.
    
    Args:
        model: Target model
        lora_weights: Dictionary of LoRA weight tensors
        adapter_name: Name for the adapter
        merge: If True, merge into base weights permanently
    
    Example:
        weights = load_lora_weights("coding_adapter.pth")
        apply_lora(model, weights, "coding")
    """
    if merge:
        merge_lora_weights(model, lora_weights)
        return

    # Apply without merging - keep as separate adapter
    if not hasattr(model, '_lora_adapters'):
        model._lora_adapters = {}

    model._lora_adapters[adapter_name] = lora_weights

    # Load weights into model's LoRA layers
    state_dict = model.state_dict()
    for key, value in lora_weights.items():
        if key in state_dict:
            state_dict[key] = value

    model.load_state_dict(state_dict, strict=False)
    logger.info(f"Applied LoRA adapter: {adapter_name}")


def merge_lora_weights(
    model: nn.Module,
    lora_weights: Dict[str, torch.Tensor]
) -> None:
    """
    Permanently merge LoRA weights into base model.
    
    After merging, LoRA weights become part of base weights
    and the adapter can be deleted to save memory.
    
    Args:
        model: Target model with LoRA adapters
        lora_weights: LoRA weight dictionary
    
    Example:
        merge_lora_weights(model, weights)  # Now permanent
    """
    # If model is a PEFT model, use built-in merge_and_unload
    if hasattr(model, 'merge_and_unload'):
        model.merge_and_unload()
        logger.info("Merged LoRA using PEFT merge_and_unload")
        return

    # Manual merge for non-PEFT models
    state_dict = model.state_dict()

    for key, lora_weight in lora_weights.items():
        if key in state_dict:
            # Simple addition for merged weights
            state_dict[key] = state_dict[key] + lora_weight

    model.load_state_dict(state_dict)
    logger.info("Merged LoRA weights into base model")


# =============================================================================
# LORA TRAINER
# =============================================================================

class LoraTrainer:
    """
    Trainer for LoRA/QLoRA fine-tuning.
    
    Features:
    - Automatic VRAM management
    - CPU offloading when needed
    - Gradient checkpointing
    - Progress callbacks for GUI
    - Checkpoint saving
    
    Example:
        trainer = LoraTrainer(
            model=my_model,
            tokenizer=my_tokenizer,
            lora_config=LoraConfig(rank=16),
            offload_config=OffloadConfig(cpu_offload=True)
        )
        trainer.train(training_data)
        trainer.save_adapter("my_lora.pth")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        lora_config: Optional[LoraConfig] = None,
        offload_config: Optional[OffloadConfig] = None,
        output_dir: str = "models/lora_adapters",
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        epochs: int = 3,
        gradient_accumulation_steps: int = 4,
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            model: Model to fine-tune
            tokenizer: Tokenizer for encoding text
            lora_config: LoRA configuration (default: rank=8)
            offload_config: CPU offloading settings
            output_dir: Where to save adapters
            learning_rate: Training learning rate
            batch_size: Training batch size
            epochs: Number of epochs
            gradient_accumulation_steps: Accumulate gradients
        """
        self.tokenizer = tokenizer
        self.lora_config = lora_config or LoraConfig()
        self.offload_config = offload_config or OffloadConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Apply LoRA to model
        if isinstance(self.lora_config, QLoraConfig):
            self.model = create_qlora_model(model, self.lora_config)
        else:
            self.model = create_lora_model(model, self.lora_config)

        # Enable gradient checkpointing if configured
        if self.offload_config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")

        # Callbacks
        self.on_progress: Optional[Callable[[int, str], None]] = None
        self.on_loss: Optional[Callable[[float], None]] = None

        # Training state
        self._stop_requested = False
        self._lock = threading.Lock()

        # Clear VRAM before training
        clear_vram()

        logger.info(f"LoraTrainer initialized: rank={self.lora_config.rank}, "
                   f"offload={self.offload_config.cpu_offload}")

    def request_stop(self) -> None:
        """Request graceful stop of training."""
        with self._lock:
            self._stop_requested = True

    def _should_stop(self) -> bool:
        """Check if stop was requested."""
        with self._lock:
            return self._stop_requested

    def _emit_progress(self, percent: int, message: str) -> None:
        """Emit progress callback."""
        if self.on_progress:
            try:
                self.on_progress(percent, message)
            except Exception:
                pass

    def train(
        self,
        data: Union[str, List[Dict[str, str]]],
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Train LoRA adapter on data.
        
        Args:
            data: Training data - text or list of {"prompt": "...", "completion": "..."}
            max_length: Maximum sequence length
        
        Returns:
            Training results dictionary
        
        Example:
            results = trainer.train([
                {"prompt": "Hello", "completion": "Hi there!"},
                {"prompt": "How are you?", "completion": "I'm doing great!"}
            ])
        """
        self._stop_requested = False
        self._emit_progress(0, "Preparing training data...")

        # Parse data
        if isinstance(data, str):
            data = [{"prompt": "", "completion": data}]

        # Clear VRAM before starting
        clear_vram()

        # Check memory
        mem_info = get_memory_info()
        logger.info(f"Starting training - VRAM: {mem_info['vram_available_gb']:.1f}GB, "
                   f"RAM: {mem_info['ram_available_gb']:.1f}GB")

        # Setup device (with CPU offload if needed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.offload_config.cpu_offload and ACCELERATE_AVAILABLE:
            # Use accelerate for smart device placement
            accelerator = Accelerator(
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                cpu=self.offload_config.offload_optimizer,
            )
            self.model, optimizer, data_loader = accelerator.prepare(
                self.model,
                self._get_optimizer(),
                self._create_dataloader(data, max_length)
            )
        else:
            self.model = self.model.to(device)
            optimizer = self._get_optimizer()

        self.model.train()

        # Training loop
        total_loss = 0.0
        epoch_loss = 0.0
        step = 0

        self._emit_progress(5, f"Training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            if self._should_stop():
                break

            epoch_loss = 0.0

            for batch_idx, batch in enumerate(self._create_batches(data, max_length)):
                if self._should_stop():
                    break

                # Move batch to device
                input_ids = batch.to(device)

                # Forward pass
                try:
                    outputs = self.model(input_ids, labels=input_ids)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                except Exception as e:
                    # Handle OOM
                    if "out of memory" in str(e).lower():
                        logger.warning("OOM detected - clearing VRAM and retrying")
                        clear_vram()
                        continue
                    raise

                # Backward pass
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    # Clear cache periodically
                    if batch_idx % 10 == 0:
                        clear_vram()

                epoch_loss += loss.item()
                step += 1

                # Progress update
                progress = int(5 + 90 * (epoch * len(data) + batch_idx) / (self.epochs * len(data)))
                self._emit_progress(progress, f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

                if self.on_loss:
                    self.on_loss(loss.item())

            total_loss += epoch_loss
            logger.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f}")

        # Final cleanup
        clear_vram()

        self._emit_progress(100, "Training complete!")

        return {
            "total_loss": total_loss,
            "epochs": self.epochs,
            "steps": step,
            'final_loss': epoch_loss,
        }

    def _get_optimizer(self):
        """Create optimizer for training."""
        from torch.optim import AdamW

        # Only optimize LoRA parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        return AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=0.01,
        )

    def _create_batches(
        self,
        data: List[Dict[str, str]],
        max_length: int
    ) -> List[torch.Tensor]:
        """Create batches from training data."""
        batches = []

        for item in data:
            text = item.get("prompt", "") + item.get("completion", "")
            if not text:
                continue

            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            if len(tokens) < 2:
                continue

            batches.append(torch.tensor([tokens], dtype=torch.long))

        return batches

    def _create_dataloader(self, data, max_length):
        """Create a DataLoader for accelerate."""
        from torch.utils.data import DataLoader, Dataset

        class SimpleDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                text = item.get("prompt", "") + item.get("completion", "")
                tokens = self.tokenizer.encode(text)[:self.max_length]
                return torch.tensor(tokens)

        dataset = SimpleDataset(data, self.tokenizer, max_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def save_adapter(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save LoRA adapter weights.
        
        Args:
            path: Output path (default: output_dir/adapter.pth)
        
        Returns:
            Path where adapter was saved
        
        Example:
            trainer.save_adapter("my_coding_lora.pth")
        """
        if path is None:
            path = self.output_dir / "adapter.pth"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        # Get only LoRA weights
        if hasattr(self.model, 'save_pretrained'):
            # PEFT model
            self.model.save_pretrained(path.parent / path.stem)
            logger.info(f"Saved PEFT adapter to: {path.parent / path.stem}")
        else:
            # Manual save - extract LoRA weights
            lora_weights = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    lora_weights[name] = param.data.cpu()

            from enigma_engine.core.safe_save import atomic_torch_save
            atomic_torch_save(lora_weights, path)
            logger.info(f"Saved LoRA weights to: {path}")

        return path


# =============================================================================
# LORA ADAPTER MANAGER (FP-D) — per-task LoRA adapters
# =============================================================================

class LoRAAdapterManager:
    """Manage multiple LoRA adapters per task/skill.

    Each task (e.g. "coding", "math", "writing") gets its own LoRA
    weights stored in a subdirectory.  You can create, switch, list,
    and merge adapters without losing specializations.

    This prevents catastrophic forgetting: the base model stays frozen
    and each skill lives in a separate adapter file.

    Args:
        base_dir: Root directory for adapter storage.
            Defaults to ``models/lora_adapters/``.

    Example::

        mgr = LoRAAdapterManager()
        mgr.create("coding", model, LoraConfig(rank=16))
        mgr.save("coding", model)
        mgr.switch("math", model)   # loads math adapter weights
    """

    def __init__(self, base_dir: Union[str, Path, None] = None):
        self.base_dir = Path(base_dir or "models/lora_adapters")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._active_task: Optional[str] = None

    # -- directory helpers --

    def _task_dir(self, task: str) -> Path:
        return self.base_dir / task

    def _weights_path(self, task: str) -> Path:
        return self._task_dir(task) / "adapter.pth"

    def _meta_path(self, task: str) -> Path:
        return self._task_dir(task) / "meta.json"

    # -- public API --

    def list_tasks(self) -> List[str]:
        """Return sorted list of available task adapters."""
        tasks = []
        if self.base_dir.exists():
            for child in sorted(self.base_dir.iterdir()):
                if child.is_dir() and (child / "adapter.pth").exists():
                    tasks.append(child.name)
        return tasks

    @property
    def active_task(self) -> Optional[str]:
        return self._active_task

    def create(
        self,
        task: str,
        model: nn.Module,
        config: Optional[LoraConfig] = None,
    ) -> Path:
        """Create a new LoRA adapter for a task and save initial weights.

        If the task already exists, this is a no-op (returns existing path).
        """
        wpath = self._weights_path(task)
        if wpath.exists():
            logger.info("Adapter '%s' already exists at %s", task, wpath)
            return wpath

        self._task_dir(task).mkdir(parents=True, exist_ok=True)
        config = config or LoraConfig()

        # Save metadata
        import json
        meta = {
            "task": task,
            "rank": config.rank,
            "alpha": config.alpha,
            "target_modules": config.target_modules,
        }
        self._meta_path(task).write_text(
            json.dumps(meta, indent=2), encoding="utf-8")

        # Save initial adapter (trainable params only)
        lora_weights: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                lora_weights[name] = param.data.cpu()

        from enigma_engine.core.safe_save import atomic_torch_save
        atomic_torch_save(lora_weights, wpath)

        logger.info("Created LoRA adapter '%s' (%d tensors)", task, len(lora_weights))
        return wpath

    def save(
        self,
        task: str,
        model: nn.Module,
    ) -> Path:
        """Save current trainable weights as an adapter for *task*."""
        self._task_dir(task).mkdir(parents=True, exist_ok=True)
        wpath = self._weights_path(task)

        lora_weights: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                lora_weights[name] = param.data.cpu()

        from enigma_engine.core.safe_save import atomic_torch_save
        atomic_torch_save(lora_weights, wpath)

        self._active_task = task
        logger.info("Saved adapter '%s' (%d tensors)", task, len(lora_weights))
        return wpath

    def switch(
        self,
        task: str,
        model: nn.Module,
        save_current: bool = True,
    ) -> None:
        """Switch the model to a different task adapter.

        If *save_current* is True, saves the active adapter first.
        Then loads the requested task's adapter weights.
        """
        if save_current and self._active_task is not None:
            self.save(self._active_task, model)

        wpath = self._weights_path(task)
        if not wpath.exists():
            raise FileNotFoundError(
                f"No adapter for task '{task}'. "
                f"Available: {self.list_tasks()}")

        weights = load_lora_weights(wpath)
        apply_lora(model, weights, adapter_name=task)
        self._active_task = task
        logger.info("Switched to adapter '%s'", task)

    def delete(self, task: str) -> None:
        """Delete a task adapter from disk."""
        tdir = self._task_dir(task)
        if tdir.exists():
            import shutil
            shutil.rmtree(tdir)
            if self._active_task == task:
                self._active_task = None
            logger.info("Deleted adapter '%s'", task)

    def merge_into_base(
        self,
        task: str,
        model: nn.Module,
    ) -> None:
        """Permanently merge a task adapter into the base model.

        After merging, the adapter's specialization becomes part of
        the base weights and the adapter file can be deleted.
        """
        wpath = self._weights_path(task)
        if not wpath.exists():
            raise FileNotFoundError(f"No adapter for task '{task}'")

        weights = load_lora_weights(wpath)
        merge_lora_weights(model, weights)
        logger.info("Merged adapter '%s' into base model", task)
