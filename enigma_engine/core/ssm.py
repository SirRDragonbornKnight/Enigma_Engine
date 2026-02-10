"""
State Space Models

Mamba-style selective state space model implementation.
Linear-time sequence modeling alternative to transformers.

FILE: enigma_engine/core/ssm.py
TYPE: Core AI Architecture
MAIN CLASSES: S4Layer, MambaBlock, MambaModel
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class SSMConfig:
    """State Space Model configuration."""
    hidden_size: int = 768
    state_size: int = 16
    conv_kernel_size: int = 4
    expand_factor: int = 2
    dt_rank: str = "auto"  # "auto" or int
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    bias: bool = False


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6).
    
    Core component of Mamba architecture with input-dependent
    state transitions.
    """
    
    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dt_rank: int = None):
        """
        Initialize Selective SSM.
        
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            dt_rank: Rank of delta projection
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
        # Delta projection rank
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)
        
        # Input projection (to inner dim)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # SSM parameters
        # A is diagonal, initialized to special values
        self.A_log = nn.Parameter(self._init_A(self.d_inner, d_state))
        
        # B, C, D projections (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # Delta projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def _init_A(self, d: int, n: int) -> Tensor:
        """Initialize A matrix with HiPPO-inspired values."""
        A = torch.arange(1, n + 1).repeat(d, 1).float()
        return torch.log(A)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with selective scan.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # [B, L, D]
        
        # Convolution
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.conv1d(x)[:, :, :seq_len]  # Trim padding
        x = x.transpose(1, 2)  # [B, L, D]
        
        # Project to get B, C, dt
        x_dbl = self.x_proj(x)  # [B, L, dt_rank + 2*N]
        dt, B, C = torch.split(
            x_dbl, 
            [self.dt_rank, self.d_state, self.d_state], 
            dim=-1
        )
        
        # Delta projection
        dt = self.dt_proj(dt)  # [B, L, D]
        dt = F.softplus(dt)  # Ensure positive
        
        # Get A (negative for stability)
        A = -torch.exp(self.A_log)  # [D, N]
        
        # Selective scan
        y = self._selective_scan(x, dt, A, B, C)
        
        # Combine with gated skip connection
        y = y * F.silu(z)
        
        # Output projection
        return self.out_proj(y)
    
    def _selective_scan(self,
                        x: Tensor,
                        dt: Tensor,
                        A: Tensor,
                        B: Tensor,
                        C: Tensor) -> Tensor:
        """
        Selective scan algorithm.
        
        Implements discretized state space model recurrence:
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t + D * x_t
        
        Args:
            x: Input [batch, seq_len, d_inner]
            dt: Time deltas [batch, seq_len, d_inner]
            A: State matrix [d_inner, d_state]
            B: Input matrix [batch, seq_len, d_state]
            C: Output matrix [batch, seq_len, d_state]
            
        Returns:
            Output [batch, seq_len, d_inner]
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Discretize A and B
        # A_bar = exp(dt * A)
        # B_bar = (A_bar - I) * A^{-1} * B ≈ dt * B
        
        dt = dt.unsqueeze(-1)  # [B, L, D, 1]
        A = A.unsqueeze(0).unsqueeze(0)  # [1, 1, D, N]
        
        A_bar = torch.exp(dt * A)  # [B, L, D, N]
        
        B = B.unsqueeze(2)  # [B, L, 1, N]
        B_bar = dt * B  # [B, L, D, N]
        
        x = x.unsqueeze(-1)  # [B, L, D, 1]
        
        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        # Recurrence
        outputs = []
        for t in range(seq_len):
            # State update: h = A_bar * h + B_bar * x
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t]
            
            # Output: y = C * h
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # [B, D]
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)  # [B, L, D]
        
        # Add D skip connection
        y = y + self.D * x.squeeze(-1)
        
        return y


class MambaBlock(nn.Module):
    """
    Mamba block combining SSM with residual connections.
    
    Similar to a transformer block but with SSM instead of attention.
    """
    
    def __init__(self, config: SSMConfig):
        """
        Initialize Mamba block.
        
        Args:
            config: SSM configuration
        """
        super().__init__()
        
        self.config = config
        
        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # Selective SSM
        dt_rank = config.dt_rank if config.dt_rank != "auto" else None
        self.ssm = SelectiveSSM(
            d_model=config.hidden_size,
            d_state=config.state_size,
            d_conv=config.conv_kernel_size,
            expand=config.expand_factor,
            dt_rank=dt_rank
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual."""
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        return x + residual


class MambaModel(nn.Module):
    """
    Complete Mamba language model.
    
    Stack of Mamba blocks with embeddings and output head.
    """
    
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int = 768,
                 num_layers: int = 24,
                 state_size: int = 16,
                 conv_kernel_size: int = 4,
                 expand_factor: int = 2):
        """
        Initialize Mamba model.
        
        Args:
            vocab_size: Vocabulary size
            hidden_size: Model dimension
            num_layers: Number of Mamba blocks
            state_size: SSM state dimension
            conv_kernel_size: Convolution kernel size
            expand_factor: FFN expansion factor
        """
        super().__init__()
        
        self.config = SSMConfig(
            hidden_size=hidden_size,
            state_size=state_size,
            conv_kernel_size=conv_kernel_size,
            expand_factor=expand_factor
        )
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(self.config) for _ in range(num_layers)
        ])
        
        # Output
        self.norm_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self,
                input_ids: Tensor,
                labels: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            labels: Optional labels for loss computation
            
        Returns:
            logits: Output logits
            loss: Optional cross-entropy loss
        """
        # Embed
        x = self.embed(input_ids)
        
        # Process through layers
        for layer in self.layers:
            x = layer(x)
        
        # Output
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    def generate(self,
                 input_ids: Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50) -> Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Initial context [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Forward pass
            logits, _ = self(input_ids)
            
            # Get last token logits
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


class S4Block(nn.Module):
    """
    S4 (Structured State Space) block.
    
    Original S4 architecture with HIPPO initialization.
    """
    
    def __init__(self,
                 d_model: int,
                 d_state: int = 64,
                 dropout: float = 0.1,
                 bidirectional: bool = False):
        """
        Initialize S4 block.
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            dropout: Dropout rate
            bidirectional: Use bidirectional processing
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_model)
        
        # S4 kernel parameters
        self.log_A = nn.Parameter(torch.randn(d_model, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Delta (step size)
        self.log_dt = nn.Parameter(torch.rand(d_model) * 0.01 - 2.0)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        residual = x
        x = self.norm(x)
        x = self.in_proj(x)
        
        # Get kernel
        A = -torch.exp(self.log_A)
        dt = torch.exp(self.log_dt)
        
        # Simple discretization
        A_bar = torch.exp(dt.unsqueeze(-1) * A)
        B_bar = self.B * dt.unsqueeze(-1)
        
        # Convolve (simplified - real S4 uses FFT)
        batch, seq_len, d = x.shape
        
        # State space recurrence
        h = torch.zeros(batch, d, self.d_state, device=x.device)
        outputs = []
        
        x_t = x.transpose(1, 2)  # [B, D, L]
        for t in range(seq_len):
            h = A_bar * h + B_bar * x_t[:, :, t:t+1]
            y = (h * self.C).sum(dim=-1) + self.D * x_t[:, :, t]
            outputs.append(y)
        
        y = torch.stack(outputs, dim=-1).transpose(1, 2)  # [B, L, D]
        
        if self.bidirectional:
            # Reverse pass
            h = torch.zeros(batch, d, self.d_state, device=x.device)
            outputs_rev = []
            for t in range(seq_len - 1, -1, -1):
                h = A_bar * h + B_bar * x_t[:, :, t:t+1]
                y_rev = (h * self.C).sum(dim=-1) + self.D * x_t[:, :, t]
                outputs_rev.append(y_rev)
            
            y_rev = torch.stack(outputs_rev[::-1], dim=-1).transpose(1, 2)
            y = (y + y_rev) / 2
        
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y + residual


def create_mamba_model(vocab_size: int = 32000,
                       hidden_size: int = 768,
                       num_layers: int = 24) -> MambaModel:
    """
    Create a Mamba model with default configuration.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Model dimension
        num_layers: Number of layers
        
    Returns:
        Mamba model
    """
    return MambaModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )


__all__ = [
    'SSMConfig',
    'SelectiveSSM',
    'MambaBlock',
    'MambaModel',
    'S4Block',
    'create_mamba_model'
]
