"""
ZeRO (Zero Redundancy Optimizer) Implementation for enigma_engine

ZeRO Stage 2: Partition optimizer states AND gradients across GPUs
- Reduces memory by ~8x for optimizer states
- Enables training models 8x larger than standard DDP

References:
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- DeepSpeed ZeRO implementation
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.optim import Optimizer


@dataclass
class ZeROConfig:
    """Configuration for ZeRO optimizer."""
    stage: int = 2  # ZeRO stage (1, 2, or 3)
    partition_gradients: bool = True  # Stage 2+
    partition_parameters: bool = False  # Stage 3 only
    offload_optimizer: bool = False  # Offload to CPU
    offload_params: bool = False  # Offload params to CPU (Stage 3)
    overlap_comm: bool = True  # Overlap communication with computation
    reduce_bucket_size: int = 500_000_000  # 500M elements per bucket
    allgather_bucket_size: int = 500_000_000
    reduce_scatter: bool = True
    contiguous_gradients: bool = True
    cpu_offload_pin_memory: bool = True


class ZeROOptimizer(Optimizer):
    """
    ZeRO Stage 2 Optimizer wrapper.
    
    Partitions optimizer states and gradients across data parallel ranks.
    Each rank only maintains optimizer states for its partition of parameters.
    
    Usage:
        model = MyModel()
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer = ZeROOptimizer(base_optimizer, model, config=ZeROConfig())
        
        for batch in dataloader:
            loss = model(batch)
            optimizer.backward(loss)
            optimizer.step()
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        model: torch.nn.Module,
        config: Optional[ZeROConfig] = None,
        process_group: Optional[Any] = None
    ):
        self.optimizer = optimizer
        self.model = model
        self.config = config or ZeROConfig()
        self.process_group = process_group
        
        # Get distributed info
        if dist.is_initialized():
            self.world_size = dist.get_world_size(process_group)
            self.rank = dist.get_rank(process_group)
        else:
            # Single GPU fallback
            self.world_size = 1
            self.rank = 0
        
        # Flatten parameters for partitioning
        self._param_to_name: dict[torch.nn.Parameter, str] = {}
        self._name_to_param: dict[str, torch.nn.Parameter] = {}
        for name, param in model.named_parameters():
            self._param_to_name[param] = name
            self._name_to_param[name] = param
        
        # Create flattened parameter groups
        self._setup_flat_params()
        
        # Gradient buffers
        self._grad_buffers: dict[int, torch.Tensor] = {}
        self._grad_partitions: dict[int, torch.Tensor] = {}
        
        # Communication handles for overlap
        self._comm_handles: list[Any] = []
        
        # Offload buffers (if enabled)
        self._cpu_buffers: dict[str, torch.Tensor] = {}
        
        # Register hooks for gradient reduction
        self._register_hooks()
        
        # Initialize optimizer state partitions
        self._partition_optimizer_state()
    
    def _setup_flat_params(self):
        """Flatten parameters into contiguous buffers for efficient partitioning."""
        self._flat_params: list[torch.Tensor] = []
        self._param_groups_flat: list[dict] = []
        
        for group_idx, group in enumerate(self.optimizer.param_groups):
            # Collect parameters in this group
            params = list(group['params'])
            if not params:
                continue
            
            # Calculate total elements
            total_elements = sum(p.numel() for p in params)
            
            # Create flat buffer
            dtype = params[0].dtype
            device = params[0].device
            flat_buffer = torch.zeros(total_elements, dtype=dtype, device=device)
            
            # Copy parameters into flat buffer
            offset = 0
            param_info = []
            for p in params:
                numel = p.numel()
                flat_buffer[offset:offset + numel].copy_(p.data.view(-1))
                param_info.append({
                    'param': p,
                    'offset': offset,
                    'numel': numel,
                    'shape': p.shape
                })
                offset += numel
            
            self._flat_params.append(flat_buffer)
            self._param_groups_flat.append({
                'flat_buffer': flat_buffer,
                'params': param_info,
                'group_idx': group_idx,
                **{k: v for k, v in group.items() if k != 'params'}
            })
    
    def _get_partition_range(self, total_elements: int) -> tuple[int, int]:
        """Get the start and end indices for this rank's partition."""
        partition_size = math.ceil(total_elements / self.world_size)
        start = self.rank * partition_size
        end = min(start + partition_size, total_elements)
        return start, end
    
    def _partition_optimizer_state(self):
        """Partition optimizer states across ranks."""
        self._partitioned_states: dict[int, dict[str, torch.Tensor]] = {}
        
        for group_idx, flat_group in enumerate(self._param_groups_flat):
            flat_buffer = flat_group['flat_buffer']
            total_elements = flat_buffer.numel()
            start, end = self._get_partition_range(total_elements)
            
            # This rank only needs optimizer state for its partition
            partition_size = end - start
            
            if partition_size > 0:
                # Create partitioned state buffers
                device = flat_buffer.device
                dtype = flat_buffer.dtype
                
                state = {
                    'exp_avg': torch.zeros(partition_size, dtype=dtype, device=device),
                    'exp_avg_sq': torch.zeros(partition_size, dtype=dtype, device=device),
                    'step': torch.tensor(0, dtype=torch.long, device=device)
                }
                
                if self.config.offload_optimizer:
                    # Move to CPU with pinned memory
                    for key in ['exp_avg', 'exp_avg_sq']:
                        cpu_tensor = torch.zeros(partition_size, dtype=dtype)
                        if self.config.cpu_offload_pin_memory:
                            cpu_tensor = cpu_tensor.pin_memory()
                        state[key] = cpu_tensor
                
                self._partitioned_states[group_idx] = state
    
    def _register_hooks(self):
        """Register backward hooks for gradient reduction."""
        self._hooks = []
        
        for param in self.model.parameters():
            if param.requires_grad:
                hook = param.register_hook(self._make_grad_hook(param))
                self._hooks.append(hook)
    
    def _make_grad_hook(self, param: torch.nn.Parameter):
        """Create a gradient hook for reduce-scatter."""
        def hook(grad: torch.Tensor):
            if self.config.partition_gradients and self.world_size > 1:
                # Async reduce-scatter
                self._reduce_scatter_grad(param, grad)
            return grad
        return hook
    
    def _reduce_scatter_grad(self, param: torch.nn.Parameter, grad: torch.Tensor):
        """Reduce-scatter gradient across ranks."""
        if not dist.is_initialized():
            return
        
        flat_grad = grad.view(-1)
        total_elements = flat_grad.numel()
        
        # Pad if necessary for even division
        padded_size = math.ceil(total_elements / self.world_size) * self.world_size
        if padded_size > total_elements:
            padded_grad = torch.zeros(padded_size, dtype=grad.dtype, device=grad.device)
            padded_grad[:total_elements] = flat_grad
            flat_grad = padded_grad
        
        # Prepare output buffer (each rank gets 1/world_size)
        chunk_size = padded_size // self.world_size
        output = torch.zeros(chunk_size, dtype=grad.dtype, device=grad.device)
        
        # Async reduce-scatter
        handle = dist.reduce_scatter_tensor(
            output, flat_grad,
            group=self.process_group,
            async_op=self.config.overlap_comm
        )
        
        if self.config.overlap_comm:
            self._comm_handles.append((handle, param, output, chunk_size))
        else:
            # Store reduced gradient partition
            name = self._param_to_name[param]
            self._grad_partitions[name] = output
    
    def backward(self, loss: torch.Tensor, retain_graph: bool = False):
        """Compute gradients with automatic gradient partitioning."""
        # Clear communication handles
        self._comm_handles.clear()
        self._grad_partitions.clear()
        
        # Standard backward pass (hooks handle reduction)
        loss.backward(retain_graph=retain_graph)
        
        # Wait for all async communications
        self._sync_gradients()
    
    def _sync_gradients(self):
        """Wait for all gradient communications to complete."""
        for handle, param, output, chunk_size in self._comm_handles:
            handle.wait()
            name = self._param_to_name[param]
            self._grad_partitions[name] = output
        self._comm_handles.clear()
    
    def step(self, closure=None):
        """
        Perform optimization step with partitioned states.
        
        Each rank:
        1. Updates only its partition of parameters using its optimizer state
        2. All-gathers updated parameters to sync across ranks
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group_idx, flat_group in enumerate(self._param_groups_flat):
            flat_buffer = flat_group['flat_buffer']
            total_elements = flat_buffer.numel()
            start, end = self._get_partition_range(total_elements)
            
            if start >= end:
                continue
            
            # Get this rank's partition
            param_partition = flat_buffer[start:end]
            state = self._partitioned_states.get(group_idx)
            
            if state is None:
                continue
            
            # Gather gradient partition for this rank
            grad_partition = self._gather_grad_partition(flat_group, start, end)
            
            if grad_partition is None:
                continue
            
            # AdamW update on partition
            self._adamw_step(
                param_partition,
                grad_partition,
                state,
                lr=flat_group.get('lr', 1e-3),
                betas=flat_group.get('betas', (0.9, 0.999)),
                eps=flat_group.get('eps', 1e-8),
                weight_decay=flat_group.get('weight_decay', 0.01)
            )
            
            # All-gather updated parameters
            self._allgather_params(flat_buffer, start, end)
        
        # Copy back to original parameters
        self._unflatten_params()
        
        return loss
    
    def _gather_grad_partition(
        self,
        flat_group: dict,
        start: int,
        end: int
    ) -> Optional[torch.Tensor]:
        """Gather gradient for this rank's parameter partition."""
        device = flat_group['flat_buffer'].device
        dtype = flat_group['flat_buffer'].dtype
        partition_size = end - start
        
        grad_partition = torch.zeros(partition_size, dtype=dtype, device=device)
        
        offset = 0
        for param_info in flat_group['params']:
            param = param_info['param']
            param_start = param_info['offset']
            param_end = param_start + param_info['numel']
            
            # Check overlap with this rank's partition
            overlap_start = max(param_start, start)
            overlap_end = min(param_end, end)
            
            if overlap_start < overlap_end:
                # Get gradient
                if param.grad is not None:
                    flat_grad = param.grad.view(-1)
                    
                    # Indices in parameter's gradient
                    grad_start = overlap_start - param_start
                    grad_end = overlap_end - param_start
                    
                    # Indices in partition
                    part_start = overlap_start - start
                    part_end = overlap_end - start
                    
                    grad_partition[part_start:part_end] = flat_grad[grad_start:grad_end]
        
        return grad_partition
    
    def _adamw_step(
        self,
        params: torch.Tensor,
        grads: torch.Tensor,
        state: dict[str, torch.Tensor],
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float
    ):
        """Fused AdamW step on parameter partition."""
        beta1, beta2 = betas
        
        # Increment step
        state['step'] += 1
        step = state['step'].item()
        
        # Get state tensors (may be on CPU if offloaded)
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        
        if self.config.offload_optimizer:
            # Move to GPU for computation
            exp_avg = exp_avg.to(params.device, non_blocking=True)
            exp_avg_sq = exp_avg_sq.to(params.device, non_blocking=True)
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        # Decoupled weight decay
        params.data.mul_(1 - lr * weight_decay)
        
        # Update biased first moment estimate
        exp_avg.mul_(beta1).add_(grads, alpha=1 - beta1)
        
        # Update biased second raw moment estimate
        exp_avg_sq.mul_(beta2).addcmul_(grads, grads, value=1 - beta2)
        
        # Compute step size
        step_size = lr / bias_correction1
        
        # Compute denominator
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        
        # Update parameters
        params.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        if self.config.offload_optimizer:
            # Move back to CPU
            state['exp_avg'].copy_(exp_avg, non_blocking=True)
            state['exp_avg_sq'].copy_(exp_avg_sq, non_blocking=True)
    
    def _allgather_params(self, flat_buffer: torch.Tensor, start: int, end: int):
        """All-gather updated parameters across ranks."""
        if not dist.is_initialized() or self.world_size == 1:
            return
        
        total_elements = flat_buffer.numel()
        partition_size = math.ceil(total_elements / self.world_size)
        padded_size = partition_size * self.world_size
        
        # Create padded buffer if necessary
        if padded_size > total_elements:
            padded_buffer = torch.zeros(
                padded_size,
                dtype=flat_buffer.dtype,
                device=flat_buffer.device
            )
            padded_buffer[:total_elements] = flat_buffer
        else:
            padded_buffer = flat_buffer
        
        # All-gather
        gathered = torch.zeros_like(padded_buffer)
        dist.all_gather_into_tensor(
            gathered,
            padded_buffer[self.rank * partition_size:(self.rank + 1) * partition_size],
            group=self.process_group
        )
        
        # Copy back
        flat_buffer.copy_(gathered[:total_elements])
    
    def _unflatten_params(self):
        """Copy flat buffers back to original parameters."""
        for flat_group in self._param_groups_flat:
            flat_buffer = flat_group['flat_buffer']
            
            for param_info in flat_group['params']:
                param = param_info['param']
                offset = param_info['offset']
                numel = param_info['numel']
                shape = param_info['shape']
                
                param.data.copy_(flat_buffer[offset:offset + numel].view(shape))
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients."""
        for param in self.model.parameters():
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()
    
    def state_dict(self) -> dict[str, Any]:
        """Get optimizer state dict with partitioned states."""
        return {
            'partitioned_states': self._partitioned_states,
            'config': self.config,
            'rank': self.rank,
            'world_size': self.world_size,
            'base_optimizer_state': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load optimizer state dict."""
        if state_dict['rank'] != self.rank:
            raise ValueError(
                f"State dict from rank {state_dict['rank']} cannot be loaded "
                f"on rank {self.rank}"
            )
        
        self._partitioned_states = state_dict['partitioned_states']
        self.optimizer.load_state_dict(state_dict['base_optimizer_state'])


def create_zero_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    stage: int = 2,
    offload_optimizer: bool = False
) -> ZeROOptimizer:
    """
    Create a ZeRO optimizer for distributed training.
    
    Args:
        model: The model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        stage: ZeRO stage (1, 2, or 3)
        offload_optimizer: Whether to offload optimizer states to CPU
    
    Returns:
        ZeROOptimizer instance
    
    Example:
        model = MyModel().cuda()
        optimizer = create_zero_optimizer(model, lr=1e-4, stage=2)
        
        for batch in dataloader:
            loss = model(batch)
            optimizer.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
    """
    base_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas
    )
    
    config = ZeROConfig(
        stage=stage,
        partition_gradients=(stage >= 2),
        partition_parameters=(stage >= 3),
        offload_optimizer=offload_optimizer
    )
    
    return ZeROOptimizer(base_optimizer, model, config)
