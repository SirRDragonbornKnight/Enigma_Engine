"""
Tensor Parallelism for enigma_engine

Distribute model layers across multiple GPUs for:
- Training/inference of models too large for single GPU
- Linear speedup with multiple GPUs
- Efficient memory utilization

Strategies:
- Column Parallel: Split weight columns across GPUs (for Linear1 in FFN)
- Row Parallel: Split weight rows across GPUs (for Linear2 in FFN)
- Pipeline Parallel: Split layers across GPUs (handled separately)

References:
- "Megatron-LM: Training Multi-Billion Parameter Language Models"
- "Reducing Activation Recomputation in Large Transformer Models"
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass 
class ParallelConfig:
    """Configuration for tensor parallelism."""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    sequence_parallel: bool = False  # Also parallelize sequence dimension
    
    # Communication
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    async_comm: bool = True


def init_parallel(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    backend: str = "nccl"
) -> ParallelConfig:
    """
    Initialize distributed parallelism.
    
    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of GPUs for pipeline parallelism
        backend: Distributed backend
    
    Returns:
        ParallelConfig
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    
    world_size = dist.get_world_size()
    data_parallel_size = world_size // (tensor_parallel_size * pipeline_parallel_size)
    
    config = ParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        backend=backend
    )
    
    logger.info(
        f"Initialized parallelism: TP={tensor_parallel_size}, "
        f"PP={pipeline_parallel_size}, DP={data_parallel_size}"
    )
    
    return config


class ParallelState:
    """Global state for tensor parallelism."""
    
    _instance = None
    
    def __init__(self):
        self.tensor_parallel_size = 1
        self.tensor_parallel_rank = 0
        self.tensor_parallel_group = None
        
        self.pipeline_parallel_size = 1
        self.pipeline_parallel_rank = 0
        self.pipeline_parallel_group = None
        
        self.data_parallel_size = 1
        self.data_parallel_rank = 0
        self.data_parallel_group = None
    
    @classmethod
    def get(cls) -> 'ParallelState':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def setup(self, config: ParallelConfig):
        """Set up parallel groups."""
        if not dist.is_initialized():
            # Single GPU fallback
            return
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        self.tensor_parallel_size = config.tensor_parallel_size
        self.pipeline_parallel_size = config.pipeline_parallel_size
        self.data_parallel_size = config.data_parallel_size
        
        # Create tensor parallel groups
        # GPUs 0,1,2,3 might be split into groups [0,1] and [2,3] for TP=2
        tp_groups = []
        for i in range(world_size // config.tensor_parallel_size):
            start = i * config.tensor_parallel_size
            end = start + config.tensor_parallel_size
            group_ranks = list(range(start, end))
            group = dist.new_group(group_ranks)
            tp_groups.append((group_ranks, group))
            
            if rank in group_ranks:
                self.tensor_parallel_group = group
                self.tensor_parallel_rank = group_ranks.index(rank)
        
        logger.info(f"Rank {rank}: TP rank {self.tensor_parallel_rank}")


def get_tensor_parallel_rank() -> int:
    """Get current tensor parallel rank."""
    state = ParallelState.get()
    return state.tensor_parallel_rank


def get_tensor_parallel_size() -> int:
    """Get tensor parallel world size."""
    state = ParallelState.get()
    return state.tensor_parallel_size


def get_tensor_parallel_group():
    """Get tensor parallel process group."""
    state = ParallelState.get()
    return state.tensor_parallel_group


class _AllReduce(torch.autograd.Function):
    """All-reduce in forward, identity in backward."""
    
    @staticmethod
    def forward(ctx, x):
        if get_tensor_parallel_size() == 1:
            return x
        dist.all_reduce(x, group=get_tensor_parallel_group())
        return x
    
    @staticmethod
    def backward(ctx, grad):
        return grad


class _AllGather(torch.autograd.Function):
    """All-gather in forward, reduce-scatter in backward."""
    
    @staticmethod
    def forward(ctx, x):
        if get_tensor_parallel_size() == 1:
            return x
        
        world_size = get_tensor_parallel_size()
        gathered = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x, group=get_tensor_parallel_group())
        return torch.cat(gathered, dim=-1)
    
    @staticmethod
    def backward(ctx, grad):
        if get_tensor_parallel_size() == 1:
            return grad
        
        world_size = get_tensor_parallel_size()
        rank = get_tensor_parallel_rank()
        
        # Reduce-scatter: each rank gets gradients for its portion
        chunk_size = grad.shape[-1] // world_size
        return grad[..., rank * chunk_size:(rank + 1) * chunk_size]


class _ReduceScatter(torch.autograd.Function):
    """Reduce-scatter in forward, all-gather in backward."""
    
    @staticmethod
    def forward(ctx, x):
        if get_tensor_parallel_size() == 1:
            return x
        
        world_size = get_tensor_parallel_size()
        rank = get_tensor_parallel_rank()
        
        # Split input and reduce
        chunk_size = x.shape[-1] // world_size
        output = torch.empty(
            *x.shape[:-1], chunk_size,
            dtype=x.dtype, device=x.device
        )
        dist.reduce_scatter_tensor(output, x, group=get_tensor_parallel_group())
        return output
    
    @staticmethod
    def backward(ctx, grad):
        if get_tensor_parallel_size() == 1:
            return grad
        
        world_size = get_tensor_parallel_size()
        gathered = [torch.empty_like(grad) for _ in range(world_size)]
        dist.all_gather(gathered, grad, group=get_tensor_parallel_group())
        return torch.cat(gathered, dim=-1)


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """All-reduce tensor across tensor parallel group."""
    return _AllReduce.apply(x)


def all_gather(x: torch.Tensor) -> torch.Tensor:
    """All-gather tensor across tensor parallel group."""
    return _AllGather.apply(x)


def reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """Reduce-scatter tensor across tensor parallel group."""
    return _ReduceScatter.apply(x)


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.
    
    The weight matrix is split along the column dimension:
    W = [W1 | W2 | ... | Wn] where each Wi is on GPU i
    
    Output is split across GPUs. Use with RowParallelLinear to get full output.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: str = "xavier"
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        
        tp_size = get_tensor_parallel_size()
        
        assert out_features % tp_size == 0, \
            f"out_features ({out_features}) must be divisible by TP size ({tp_size})"
        
        self.out_features_per_partition = out_features // tp_size
        
        # Each rank has a portion of the output dimension
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights(init_method)
    
    def _init_weights(self, method: str):
        if method == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif method == "normal":
            nn.init.normal_(self.weight, std=0.02)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local matmul
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        
        # Gather if needed
        if self.gather_output:
            output = all_gather(output)
        
        return output
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, gather_output: bool = True) -> 'ColumnParallelLinear':
        """Create from existing linear layer."""
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            gather_output=gather_output
        )
        
        # Copy weights for this partition
        tp_rank = get_tensor_parallel_rank()
        tp_size = get_tensor_parallel_size()
        
        partition_size = linear.out_features // tp_size
        start = tp_rank * partition_size
        end = start + partition_size
        
        layer.weight.data.copy_(linear.weight.data[start:end, :])
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data[start:end])
        
        return layer


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.
    
    The weight matrix is split along the row dimension:
    W = [W1; W2; ...; Wn] where each Wi is on GPU i
    
    Input should be split across GPUs (from ColumnParallelLinear).
    Output is all-reduced to get full result.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        init_method: str = "xavier"
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        
        tp_size = get_tensor_parallel_size()
        
        assert in_features % tp_size == 0, \
            f"in_features ({in_features}) must be divisible by TP size ({tp_size})"
        
        self.in_features_per_partition = in_features // tp_size
        
        # Each rank has a portion of the input dimension
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights(init_method)
    
    def _init_weights(self, method: str):
        if method == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif method == "normal":
            nn.init.normal_(self.weight, std=0.02)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is not already parallel, split it
        if not self.input_is_parallel:
            tp_rank = get_tensor_parallel_rank()
            tp_size = get_tensor_parallel_size()
            chunk_size = x.shape[-1] // tp_size
            x = x[..., tp_rank * chunk_size:(tp_rank + 1) * chunk_size]
        
        # Local matmul
        output = torch.nn.functional.linear(x, self.weight)
        
        # All-reduce to get full output
        output = all_reduce(output)
        
        # Add bias after all-reduce
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, input_is_parallel: bool = True) -> 'RowParallelLinear':
        """Create from existing linear layer."""
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            input_is_parallel=input_is_parallel
        )
        
        # Copy weights for this partition
        tp_rank = get_tensor_parallel_rank()
        tp_size = get_tensor_parallel_size()
        
        partition_size = linear.in_features // tp_size
        start = tp_rank * partition_size
        end = start + partition_size
        
        layer.weight.data.copy_(linear.weight.data[:, start:end])
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        
        return layer


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with vocabulary parallelism.
    
    Vocabulary is split across GPUs, reducing memory per GPU.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        tp_size = get_tensor_parallel_size()
        tp_rank = get_tensor_parallel_rank()
        
        # Each rank handles a portion of the vocabulary
        self.vocab_start_idx = tp_rank * (num_embeddings // tp_size)
        self.vocab_end_idx = (tp_rank + 1) * (num_embeddings // tp_size)
        self.num_embeddings_per_partition = self.vocab_end_idx - self.vocab_start_idx
        
        self.embedding = nn.Embedding(
            self.num_embeddings_per_partition,
            embedding_dim,
            padding_idx=None  # Handle padding manually
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mask tokens not in this partition
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        
        # Shift indices for local embedding
        local_x = x - self.vocab_start_idx
        local_x = local_x.clamp(0, self.num_embeddings_per_partition - 1)
        
        # Get embeddings
        output = self.embedding(local_x)
        
        # Zero out embeddings for tokens not in this partition
        output = output * mask.unsqueeze(-1).float()
        
        # All-reduce to combine embeddings from all partitions
        output = all_reduce(output)
        
        return output


def parallelize_model(
    model: nn.Module,
    config: ParallelConfig
) -> nn.Module:
    """
    Apply tensor parallelism to a model.
    
    Automatically replaces Linear layers with parallel versions.
    
    Args:
        model: Model to parallelize
        config: Parallel configuration
    
    Returns:
        Parallelized model
    """
    # Set up parallel state
    state = ParallelState.get()
    state.setup(config)
    
    if config.tensor_parallel_size == 1:
        return model
    
    # Replace layers
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Determine parallelism type based on layer name/position
            # This is a heuristic - real implementation would be model-specific
            
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            
            # First linear in FFN: column parallel
            # Second linear in FFN: row parallel
            # Attention projections: column parallel
            if 'out_proj' in name or 'down_proj' in name or 'fc2' in name:
                parallel_layer = RowParallelLinear.from_linear(module)
            else:
                parallel_layer = ColumnParallelLinear.from_linear(module, gather_output=False)
            
            setattr(parent, child_name, parallel_layer)
    
    logger.info(f"Parallelized model with TP={config.tensor_parallel_size}")
    
    return model
