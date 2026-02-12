"""
Mixed Precision Training

Automatic mixed precision (AMP) training with FP16/BF16 support.
Includes gradient scaling, loss scaling, and precision policies.

FILE: enigma_engine/core/mixed_precision.py
TYPE: Core/Training
MAIN CLASSES: MixedPrecisionTrainer, GradScaler, PrecisionPolicy
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler as TorchGradScaler
    from torch.cuda.amp import autocast
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Precision(Enum):
    """Precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"


class ScalerState(Enum):
    """Gradient scaler state."""
    READY = "ready"
    UNSCALED = "unscaled"
    STEPPED = "stepped"


@dataclass
class PrecisionConfig:
    """Mixed precision configuration."""
    # Main precision
    compute_dtype: Precision = Precision.FP16
    storage_dtype: Precision = Precision.FP32
    
    # Gradient scaling
    use_grad_scaler: bool = True
    initial_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    max_scale: float = 2 ** 24
    min_scale: float = 1.0
    
    # Safety
    enabled: bool = True
    skip_nan_gradients: bool = True
    nan_skip_threshold: int = 20  # Skip if NaN this many times
    
    # Layer policies
    keep_fp32_modules: list[str] = None  # Modules to keep in FP32
    
    def __post_init__(self):
        if self.keep_fp32_modules is None:
            self.keep_fp32_modules = ["norm", "layernorm", "rmsnorm", "embedding"]


if HAS_TORCH:
    
    class GradScaler:
        """
        Gradient scaler for mixed precision training.
        
        Scales loss to prevent underflow in FP16 gradients,
        and unscales gradients before optimizer step.
        """
        
        def __init__(self, config: PrecisionConfig = None) -> None:
            self.config = config or PrecisionConfig()
            
            self._scale = self.config.initial_scale
            self._growth_tracker = 0
            self._state = ScalerState.READY
            
            # NaN tracking
            self._nan_count = 0
            self._total_steps = 0
            
            # Use torch's scaler if available
            self._torch_scaler = None
            if torch.cuda.is_available():
                self._torch_scaler = TorchGradScaler(
                    init_scale=self.config.initial_scale,
                    growth_factor=self.config.growth_factor,
                    backoff_factor=self.config.backoff_factor,
                    growth_interval=self.config.growth_interval,
                    enabled=self.config.enabled
                )
        
        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            """Scale loss for backward pass."""
            if not self.config.enabled:
                return loss
            
            if self._torch_scaler:
                return self._torch_scaler.scale(loss)
            
            self._state = ScalerState.READY
            return loss * self._scale
        
        def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
            """Unscale gradients before optimizer step."""
            if not self.config.enabled:
                return
            
            if self._torch_scaler:
                self._torch_scaler.unscale_(optimizer)
                return
            
            if self._state == ScalerState.UNSCALED:
                return
            
            inv_scale = 1.0 / self._scale
            
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.mul_(inv_scale)
            
            self._state = ScalerState.UNSCALED
        
        def step(self, optimizer: torch.optim.Optimizer) -> bool:
            """
            Step optimizer with gradient checking.
            
            Returns:
                True if step was taken, False if skipped due to NaN
            """
            if not self.config.enabled:
                optimizer.step()
                return True
            
            if self._torch_scaler:
                self._torch_scaler.step(optimizer)
                return True
            
            # Unscale if not already done
            if self._state != ScalerState.UNSCALED:
                self.unscale_(optimizer)
            
            # Check for NaN/Inf gradients
            has_nan = self._check_nan_gradients(optimizer)
            
            self._total_steps += 1
            
            if has_nan:
                self._nan_count += 1
                self._handle_nan()
                return False
            
            optimizer.step()
            self._state = ScalerState.STEPPED
            
            # Update scale
            self._growth_tracker += 1
            if self._growth_tracker >= self.config.growth_interval:
                self._scale = min(
                    self._scale * self.config.growth_factor,
                    self.config.max_scale
                )
                self._growth_tracker = 0
            
            return True
        
        def update(self) -> None:
            """Update scaler state after optimizer step."""
            if self._torch_scaler:
                self._torch_scaler.update()
            self._state = ScalerState.READY
        
        def _check_nan_gradients(self, optimizer: torch.optim.Optimizer) -> bool:
            """Check if any gradients are NaN or Inf."""
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            return True
            return False
        
        def _handle_nan(self) -> None:
            """Handle NaN gradients by reducing scale."""
            old_scale = self._scale
            self._scale = max(
                self._scale * self.config.backoff_factor,
                self.config.min_scale
            )
            self._growth_tracker = 0
            
            logger.warning(
                f"NaN gradients detected. Scale reduced: {old_scale:.1f} -> {self._scale:.1f}"
            )
            
            if self._nan_count >= self.config.nan_skip_threshold:
                logger.error(
                    f"Too many NaN gradients ({self._nan_count}). "
                    "Consider increasing min_scale or disabling AMP."
                )
        
        def get_scale(self) -> float:
            """Get current scale value."""
            if self._torch_scaler:
                return self._torch_scaler.get_scale()
            return self._scale
        
        def state_dict(self) -> dict[str, Any]:
            """Get state dict for checkpointing."""
            if self._torch_scaler:
                return self._torch_scaler.state_dict()
            return {
                "scale": self._scale,
                "growth_tracker": self._growth_tracker,
                "nan_count": self._nan_count,
                "total_steps": self._total_steps
            }
        
        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            """Load state dict."""
            if self._torch_scaler:
                self._torch_scaler.load_state_dict(state_dict)
            else:
                self._scale = state_dict["scale"]
                self._growth_tracker = state_dict["growth_tracker"]
                self._nan_count = state_dict.get("nan_count", 0)
                self._total_steps = state_dict.get("total_steps", 0)
    
    
    class PrecisionPolicy:
        """
        Policy for which operations run in which precision.
        
        Keeps certain sensitive operations in FP32 for stability.
        """
        
        def __init__(self, config: PrecisionConfig = None) -> None:
            self.config = config or PrecisionConfig()
            self._dtype_map = {
                Precision.FP32: torch.float32,
                Precision.FP16: torch.float16,
                Precision.BF16: torch.bfloat16,
            }
        
        def get_dtype(self, precision: Precision = None) -> torch.dtype:
            """Get torch dtype for precision."""
            precision = precision or self.config.compute_dtype
            return self._dtype_map.get(precision, torch.float32)
        
        def should_keep_fp32(self, module: nn.Module) -> bool:
            """Check if module should be kept in FP32."""
            module_name = module.__class__.__name__.lower()
            
            for pattern in self.config.keep_fp32_modules:
                if pattern.lower() in module_name:
                    return True
            
            return False
        
        def apply_to_model(self, model: nn.Module) -> None:
            """
            Apply precision policy to model.
            
            Converts most modules to compute dtype while keeping
            specific modules in FP32.
            """
            compute_dtype = self.get_dtype(self.config.compute_dtype)
            
            for name, module in model.named_modules():
                if self.should_keep_fp32(module):
                    module.float()
                    logger.debug(f"Keeping {name} in FP32")
                else:
                    # Convert buffers and parameters
                    for param_name, param in module.named_parameters(recurse=False):
                        if param.dtype == torch.float32:
                            param.data = param.data.to(compute_dtype)
                    
                    for buf_name, buf in module.named_buffers(recurse=False):
                        if buf is not None and buf.dtype == torch.float32:
                            module.register_buffer(
                                buf_name, buf.to(compute_dtype)
                            )
    
    
    class MixedPrecisionTrainer:
        """
        Trainer with automatic mixed precision support.
        
        Handles autocasting, gradient scaling, and precision policies.
        """
        
        def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            config: PrecisionConfig = None
        ) -> None:
            self.model = model
            self.optimizer = optimizer
            self.config = config or PrecisionConfig()
            
            # Components
            self.scaler = GradScaler(self.config)
            self.policy = PrecisionPolicy(self.config)
            
            # Apply precision policy to model
            if self.config.enabled:
                self.policy.apply_to_model(model)
        
        @contextmanager
        def autocast_context(self):
            """Context manager for automatic mixed precision."""
            if not self.config.enabled:
                yield
                return
            
            dtype = self.policy.get_dtype(self.config.compute_dtype)
            
            if torch.cuda.is_available():
                with autocast(enabled=True, dtype=dtype):
                    yield
            else:
                # CPU autocast (PyTorch 2.0+)
                try:
                    with torch.autocast(device_type='cpu', dtype=dtype):
                        yield
                except (RuntimeError, AttributeError):
                    yield  # Fall back to default precision if autocast unavailable
        
        def train_step(
            self,
            batch: Any,
            loss_fn: Callable[[Any, Any], torch.Tensor]
        ) -> tuple[float, bool]:
            """
            Perform a single training step with mixed precision.
            
            Args:
                batch: Input batch (inputs, targets)
                loss_fn: Loss function (model_output, targets) -> loss
            
            Returns:
                (loss_value, step_taken)
            """
            self.model.train()
            self.optimizer.zero_grad()
            
            inputs, targets = batch
            
            # Forward pass with autocast
            with self.autocast_context():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
            
            # Backward pass with scaled loss
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            
            # Optimizer step
            self.scaler.unscale_(self.optimizer)
            
            # Optional gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            step_taken = self.scaler.step(self.optimizer)
            self.scaler.update()
            
            return loss.item(), step_taken
        
        def eval_step(
            self,
            batch: Any,
            loss_fn: Callable[[Any, Any], torch.Tensor]
        ) -> float:
            """
            Perform evaluation step with mixed precision.
            
            Args:
                batch: Input batch
                loss_fn: Loss function
            
            Returns:
                loss_value
            """
            self.model.eval()
            
            inputs, targets = batch
            
            with torch.no_grad():
                with self.autocast_context():
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets)
            
            return loss.item()
        
        def get_state(self) -> dict[str, Any]:
            """Get trainer state for checkpointing."""
            return {
                "scaler": self.scaler.state_dict(),
                "config": {
                    "compute_dtype": self.config.compute_dtype.value,
                    "enabled": self.config.enabled
                }
            }
        
        def load_state(self, state: dict[str, Any]) -> None:
            """Load trainer state."""
            self.scaler.load_state_dict(state["scaler"])
    
    
    def enable_mixed_precision(
        model: nn.Module,
        precision: Precision = Precision.FP16,
        **kwargs
    ) -> PrecisionPolicy:
        """
        Enable mixed precision on a model.
        
        Args:
            model: Model to convert
            precision: Target compute precision
            **kwargs: Additional config options
        
        Returns:
            PrecisionPolicy used
        """
        config = PrecisionConfig(compute_dtype=precision, **kwargs)
        policy = PrecisionPolicy(config)
        policy.apply_to_model(model)
        return policy
    
    
    @contextmanager
    def mixed_precision_context(
        dtype: Precision = Precision.FP16,
        device: str = "cuda"
    ):
        """
        Context manager for mixed precision forward pass.
        
        Args:
            dtype: Compute dtype
            device: Device type
        """
        dtype_map = {
            Precision.FP16: torch.float16,
            Precision.BF16: torch.bfloat16,
            Precision.FP32: torch.float32
        }
        
        torch_dtype = dtype_map.get(dtype, torch.float16)
        
        if device == "cuda" and torch.cuda.is_available():
            with autocast(enabled=True, dtype=torch_dtype):
                yield
        else:
            try:
                with torch.autocast(device_type='cpu', dtype=torch_dtype):
                    yield
            except (RuntimeError, AttributeError):
                yield  # Fall back to default precision

else:
    # Stubs when torch not available
    class GradScaler:
        pass
    
    class PrecisionPolicy:
        pass
    
    class MixedPrecisionTrainer:
        pass
    
    def enable_mixed_precision(*args, **kwargs):
        raise ImportError("PyTorch required for mixed precision")
    
    @contextmanager
    def mixed_precision_context(*args, **kwargs):
        yield
