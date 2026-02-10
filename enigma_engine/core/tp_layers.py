"""
Tensor Parallel Layers

Advanced layer splitting across multiple GPUs for large model training.
Supports column parallel, row parallel, and sequence parallel modes.

FILE: enigma_engine/core/tp_layers.py
TYPE: Training/Inference
MAIN CLASSES: TPColumnLinear, TPRowLinear, TPEmbedding
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class TPMode(Enum):
    """Tensor parallel modes."""
    COLUMN = "column"
    ROW = "row"
    SEQUENCE = "sequence"
    VOCAB = "vocab"


@dataclass
class TPConfig:
    """Tensor parallelism configuration."""
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"
    sequence_parallel: bool = False
    overlap_communication: bool = True


class TPState:
    """Global tensor parallel state (singleton)."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance.config = None
            cls._instance.group = None
        return cls._instance
    
    def init(self, config: TPConfig):
        """Initialize TP state."""
        if self._initialized:
            return
        
        self.config = config
        
        if config.world_size > 1 and dist.is_initialized():
            self.group = dist.new_group(list(range(config.world_size)))
        
        self._initialized = True
        logger.info(f"TP initialized: rank {config.rank}/{config.world_size}")
    
    @property
    def world_size(self) -> int:
        return self.config.world_size if self.config else 1
    
    @property
    def rank(self) -> int:
        return self.config.rank if self.config else 0


def get_tp() -> TPState:
    """Get tensor parallel state."""
    return TPState()


class _AllReduceForward(torch.autograd.Function):
    """All-reduce in forward, identity in backward."""
    
    @staticmethod
    def forward(ctx, input_: Tensor) -> Tensor:
        if get_tp().world_size == 1:
            return input_
        output = input_.clone()
        dist.all_reduce(output, group=get_tp().group)
        return output
    
    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:
        return grad


class _AllReduceBackward(torch.autograd.Function):
    """Identity in forward, all-reduce in backward."""
    
    @staticmethod
    def forward(ctx, input_: Tensor) -> Tensor:
        return input_
    
    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:
        if get_tp().world_size == 1:
            return grad
        output = grad.clone()
        dist.all_reduce(output, group=get_tp().group)
        return output


class _AllGatherForward(torch.autograd.Function):
    """All-gather in forward, reduce-scatter in backward."""
    
    @staticmethod
    def forward(ctx, input_: Tensor, dim: int) -> Tensor:
        ctx.dim = dim
        if get_tp().world_size == 1:
            return input_
        
        gathered = [torch.empty_like(input_) for _ in range(get_tp().world_size)]
        dist.all_gather(gathered, input_, group=get_tp().group)
        return torch.cat(gathered, dim=dim)
    
    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:
        if get_tp().world_size == 1:
            return grad, None
        
        dim = ctx.dim
        size = grad.size(dim) // get_tp().world_size
        return grad.narrow(dim, get_tp().rank * size, size).contiguous(), None


def all_reduce_forward(x: Tensor) -> Tensor:
    """All-reduce in forward pass."""
    return _AllReduceForward.apply(x)


def all_reduce_backward(x: Tensor) -> Tensor:
    """All-reduce in backward pass."""
    return _AllReduceBackward.apply(x)


def all_gather_forward(x: Tensor, dim: int = -1) -> Tensor:
    """All-gather in forward pass."""
    return _AllGatherForward.apply(x, dim)


class TPColumnLinear(nn.Module):
    """
    Column-parallel linear layer.
    
    Splits output features across GPUs.
    Each GPU holds weight[out_features/world_size, in_features].
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 gather_output: bool = True):
        super().__init__()
        
        tp = get_tp()
        
        assert out_features % tp.world_size == 0, \
            f"out_features {out_features} must be divisible by world_size {tp.world_size}"
        
        self.in_features = in_features
        self.out_features = out_features
        self.partition_size = out_features // tp.world_size
        self.gather_output = gather_output
        
        self.weight = nn.Parameter(torch.empty(self.partition_size, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.partition_size))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: Tensor) -> Tensor:
        # Copy input so gradients reduce correctly
        x = all_reduce_backward(x)
        
        # Local linear
        out = nn.functional.linear(x, self.weight, self.bias)
        
        if self.gather_output:
            out = all_gather_forward(out, dim=-1)
        
        return out
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, gather_output: bool = True) -> 'TPColumnLinear':
        """Convert nn.Linear to column parallel."""
        tp = get_tp()
        layer = cls(linear.in_features, linear.out_features, 
                    bias=linear.bias is not None, gather_output=gather_output)
        
        with torch.no_grad():
            start = tp.rank * layer.partition_size
            layer.weight.copy_(linear.weight[start:start + layer.partition_size])
            if layer.bias is not None:
                layer.bias.copy_(linear.bias[start:start + layer.partition_size])
        
        return layer


class TPRowLinear(nn.Module):
    """
    Row-parallel linear layer.
    
    Splits input features across GPUs.
    Each GPU holds weight[out_features, in_features/world_size].
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 input_is_parallel: bool = False):
        super().__init__()
        
        tp = get_tp()
        
        assert in_features % tp.world_size == 0, \
            f"in_features {in_features} must be divisible by world_size {tp.world_size}"
        
        self.in_features = in_features
        self.out_features = out_features
        self.partition_size = in_features // tp.world_size
        self.input_is_parallel = input_is_parallel
        
        self.weight = nn.Parameter(torch.empty(out_features, self.partition_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        tp = get_tp()
        
        if not self.input_is_parallel:
            # Split input
            x = x.narrow(-1, tp.rank * self.partition_size, self.partition_size)
        
        # Local linear (no bias yet)
        out = nn.functional.linear(x, self.weight)
        
        # All-reduce to sum partial outputs
        out = all_reduce_forward(out)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, input_is_parallel: bool = False) -> 'TPRowLinear':
        """Convert nn.Linear to row parallel."""
        tp = get_tp()
        layer = cls(linear.in_features, linear.out_features,
                    bias=linear.bias is not None, input_is_parallel=input_is_parallel)
        
        with torch.no_grad():
            start = tp.rank * layer.partition_size
            layer.weight.copy_(linear.weight[:, start:start + layer.partition_size])
            if layer.bias is not None:
                layer.bias.copy_(linear.bias)
        
        return layer


class TPEmbedding(nn.Module):
    """
    Vocabulary-parallel embedding layer.
    
    Splits vocabulary across GPUs.
    """
    
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None):
        super().__init__()
        
        tp = get_tp()
        
        assert num_embeddings % tp.world_size == 0, \
            f"num_embeddings {num_embeddings} must be divisible by world_size {tp.world_size}"
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.partition_size = num_embeddings // tp.world_size
        self.vocab_start = tp.rank * self.partition_size
        self.vocab_end = self.vocab_start + self.partition_size
        self.padding_idx = padding_idx
        
        self.weight = nn.Parameter(torch.empty(self.partition_size, embedding_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.weight)
    
    def forward(self, x: Tensor) -> Tensor:
        # Mask tokens outside this partition
        mask = (x >= self.vocab_start) & (x < self.vocab_end)
        local_x = x.clone()
        local_x[mask] -= self.vocab_start
        local_x[~mask] = 0
        
        # Local embedding
        out = nn.functional.embedding(local_x, self.weight)
        out[~mask] = 0.0
        
        # Sum across all partitions
        return all_reduce_forward(out)


class TPAttention(nn.Module):
    """Tensor-parallel multi-head attention."""
    
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        
        tp = get_tp()
        
        assert num_heads % tp.world_size == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.local_heads = num_heads // tp.world_size
        
        # QKV projection (column parallel, no gather)
        self.qkv_proj = TPColumnLinear(
            hidden_size, 3 * hidden_size, bias=False, gather_output=False
        )
        
        # Output projection (row parallel)
        self.out_proj = TPRowLinear(
            hidden_size, hidden_size, bias=False, input_is_parallel=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.local_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            scores = scores + mask
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # [B, H, S, D]
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        
        # Output projection
        return self.out_proj(out)


class TPMLP(nn.Module):
    """Tensor-parallel MLP block."""
    
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 activation: str = "gelu"):
        super().__init__()
        
        # Up projection (column parallel)
        self.gate_proj = TPColumnLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        self.up_proj = TPColumnLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        
        # Down projection (row parallel)
        self.down_proj = TPRowLinear(
            intermediate_size, hidden_size, bias=False, input_is_parallel=True
        )
        
        self.activation = nn.GELU() if activation == "gelu" else nn.SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU-style: activation(gate) * up
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def parallelize_model(model: nn.Module, config: TPConfig) -> nn.Module:
    """
    Convert model layers to tensor parallel.
    
    Args:
        model: PyTorch model
        config: TP configuration
        
    Returns:
        Model with TP layers
    """
    # Initialize global state
    tp = get_tp()
    tp.init(config)
    
    if config.world_size == 1:
        return model
    
    # Find and replace layers
    replacements = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                replacements[name] = TPColumnLinear.from_linear(module, gather_output=False)
            elif 'o_proj' in name:
                replacements[name] = TPRowLinear.from_linear(module, input_is_parallel=True)
            elif 'gate_proj' in name or 'up_proj' in name:
                replacements[name] = TPColumnLinear.from_linear(module, gather_output=False)
            elif 'down_proj' in name:
                replacements[name] = TPRowLinear.from_linear(module, input_is_parallel=True)
    
    # Apply replacements
    for name, new_module in replacements.items():
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    logger.info(f"Parallelized {len(replacements)} layers across {config.world_size} GPUs")
    return model


__all__ = [
    'TPConfig',
    'TPMode',
    'TPState',
    'get_tp',
    'TPColumnLinear',
    'TPRowLinear',
    'TPEmbedding',
    'TPAttention',
    'TPMLP',
    'parallelize_model',
    'all_reduce_forward',
    'all_reduce_backward',
    'all_gather_forward'
]
