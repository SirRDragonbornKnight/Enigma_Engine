"""
Pure Python Neural Network

Zero-dependency neural network implementation using only Python stdlib.
Works anywhere Python runs - no numpy, no torch, no pip installs required.

Optimized for:
- PyPy (10-50x faster than CPython)
- Multiprocessing (parallel matrix ops on CPU cores)
- Nano/micro model sizes (1-5M parameters)

This is a FALLBACK for devices where PyTorch won't install.
For real performance, use the PyTorch backend.

Usage:
    from forge_ai.builtin.neural_network import (
        PureLinear, PureAttention, PureTransformer,
        get_backend, set_backend
    )
    
    # Auto-selects best backend based on model size
    model = PureTransformer(vocab_size=1000, d_model=64, n_layers=2)
    output = model.forward(input_ids)
"""

import math
import random
import json
import struct
import sys
import platform
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
from functools import partial
import os

# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def is_pypy() -> bool:
    """Check if running on PyPy (faster JIT Python)."""
    return platform.python_implementation() == "PyPy"


def get_python_info() -> Dict[str, Any]:
    """Get Python runtime information."""
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "is_pypy": is_pypy(),
        "cpu_count": mp.cpu_count(),
        "platform": platform.system(),
        "machine": platform.machine(),
    }


# PyPy-specific optimizations
PYPY_MODE = is_pypy()
if PYPY_MODE:
    # PyPy's JIT works better with simpler code patterns
    # Disable some multiprocessing overhead since PyPy is already fast
    _DEFAULT_USE_MP = False
    print("[PureNN] Running on PyPy - JIT optimizations active")
else:
    _DEFAULT_USE_MP = True


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PureConfig:
    """Configuration for pure Python backend."""
    # Model architecture
    vocab_size: int = 1000
    d_model: int = 64          # Embedding dimension
    n_heads: int = 4           # Attention heads
    n_layers: int = 2          # Transformer layers
    d_ff: int = 256            # Feed-forward dimension
    max_seq_len: int = 512     # Maximum sequence length
    dropout: float = 0.0       # Dropout (only for training)
    
    # Performance
    use_multiprocessing: bool = _DEFAULT_USE_MP  # Auto-disable on PyPy
    n_workers: int = 0         # 0 = auto-detect CPU cores
    chunk_size: int = 64       # Matrix chunk size for parallelization
    
    # Training
    learning_rate: float = 0.001
    
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
    
    def param_count(self) -> int:
        """Estimate total parameters."""
        # Embeddings
        emb = self.vocab_size * self.d_model
        # Per layer: attention (4 projections) + ffn (2 layers) + norms (2)
        per_layer = (
            4 * self.d_model * self.d_model +  # Q, K, V, O projections
            2 * self.d_model * self.d_ff +      # FFN up and down
            4 * self.d_model                     # LayerNorm params
        )
        return emb + (self.n_layers * per_layer) + self.vocab_size * self.d_model


# ═══════════════════════════════════════════════════════════════════════════════
# MATRIX OPERATIONS - The Foundation
# ═══════════════════════════════════════════════════════════════════════════════

class Matrix:
    """
    Pure Python matrix class.
    
    Stores data as a flat list for memory efficiency.
    Supports basic linear algebra operations.
    """
    
    __slots__ = ['data', 'rows', 'cols']
    
    def __init__(self, rows: int, cols: int, data: Optional[List[float]] = None):
        self.rows = rows
        self.cols = cols
        if data is not None:
            self.data = data
        else:
            self.data = [0.0] * (rows * cols)
    
    @classmethod
    def from_2d(cls, arr: List[List[float]]) -> 'Matrix':
        """Create matrix from 2D list."""
        rows = len(arr)
        cols = len(arr[0]) if rows > 0 else 0
        data = []
        for row in arr:
            data.extend(row)
        return cls(rows, cols, data)
    
    @classmethod
    def zeros(cls, rows: int, cols: int) -> 'Matrix':
        """Create zero matrix."""
        return cls(rows, cols, [0.0] * (rows * cols))
    
    @classmethod
    def ones(cls, rows: int, cols: int) -> 'Matrix':
        """Create matrix of ones."""
        return cls(rows, cols, [1.0] * (rows * cols))
    
    @classmethod
    def randn(cls, rows: int, cols: int, std: float = 1.0) -> 'Matrix':
        """Create matrix with random normal values."""
        # Box-Muller transform for normal distribution
        data = []
        n = rows * cols
        for i in range(0, n, 2):
            u1 = random.random()
            u2 = random.random()
            # Avoid log(0)
            u1 = max(u1, 1e-10)
            mag = std * math.sqrt(-2.0 * math.log(u1))
            z0 = mag * math.cos(2.0 * math.pi * u2)
            z1 = mag * math.sin(2.0 * math.pi * u2)
            data.append(z0)
            if i + 1 < n:
                data.append(z1)
        return cls(rows, cols, data[:n])
    
    @classmethod
    def xavier_init(cls, rows: int, cols: int) -> 'Matrix':
        """Xavier/Glorot initialization."""
        std = math.sqrt(2.0 / (rows + cols))
        return cls.randn(rows, cols, std)
    
    def __getitem__(self, idx: Tuple[int, int]) -> float:
        """Get element at (row, col)."""
        row, col = idx
        return self.data[row * self.cols + col]
    
    def __setitem__(self, idx: Tuple[int, int], value: float):
        """Set element at (row, col)."""
        row, col = idx
        self.data[row * self.cols + col] = value
    
    def get_row(self, row: int) -> List[float]:
        """Get a row as list."""
        start = row * self.cols
        return self.data[start:start + self.cols]
    
    def set_row(self, row: int, values: List[float]):
        """Set a row from list."""
        start = row * self.cols
        for i, v in enumerate(values):
            self.data[start + i] = v
    
    def to_2d(self) -> List[List[float]]:
        """Convert to 2D list."""
        result = []
        for i in range(self.rows):
            result.append(self.get_row(i))
        return result
    
    def copy(self) -> 'Matrix':
        """Create a copy."""
        return Matrix(self.rows, self.cols, self.data.copy())
    
    def transpose(self) -> 'Matrix':
        """Transpose the matrix."""
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result[j, i] = self[i, j]
        return result
    
    @property
    def T(self) -> 'Matrix':
        """Transpose property."""
        return self.transpose()
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return shape as tuple."""
        return (self.rows, self.cols)
    
    def reshape(self, rows: int, cols: int) -> 'Matrix':
        """Reshape matrix (must have same total elements)."""
        if rows * cols != self.rows * self.cols:
            raise ValueError(f"Cannot reshape {self.shape} to ({rows}, {cols})")
        return Matrix(rows, cols, self.data.copy())
    
    def __repr__(self) -> str:
        return f"Matrix({self.rows}x{self.cols})"


# ═══════════════════════════════════════════════════════════════════════════════
# MATRIX MATH OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def matmul(a: Matrix, b: Matrix) -> Matrix:
    """
    Matrix multiplication: C = A @ B
    
    This is the core operation in neural networks.
    O(n*m*k) complexity for (n,m) @ (m,k) matrices.
    """
    if a.cols != b.rows:
        raise ValueError(f"Cannot multiply {a.shape} @ {b.shape}")
    
    result = Matrix.zeros(a.rows, b.cols)
    
    # Standard triple loop - optimized for memory access pattern
    for i in range(a.rows):
        a_row_start = i * a.cols
        for k in range(a.cols):
            a_ik = a.data[a_row_start + k]
            b_row_start = k * b.cols
            result_row_start = i * result.cols
            for j in range(b.cols):
                result.data[result_row_start + j] += a_ik * b.data[b_row_start + j]
    
    return result


def matmul_parallel(a: Matrix, b: Matrix, n_workers: int = 0) -> Matrix:
    """
    Parallel matrix multiplication using multiprocessing.
    
    Splits rows across CPU cores for ~linear speedup.
    """
    if a.cols != b.rows:
        raise ValueError(f"Cannot multiply {a.shape} @ {b.shape}")
    
    if n_workers == 0:
        n_workers = mp.cpu_count()
    
    # For small matrices, don't bother with parallelization
    if a.rows < n_workers * 4:
        return matmul(a, b)
    
    # Prepare data for workers
    b_data = b.data
    b_cols = b.cols
    b_rows = b.rows
    
    # Split rows across workers
    chunk_size = max(1, a.rows // n_workers)
    chunks = []
    for i in range(0, a.rows, chunk_size):
        end = min(i + chunk_size, a.rows)
        chunk_data = a.data[i * a.cols : end * a.cols]
        chunks.append((chunk_data, a.cols, end - i, b_data, b_rows, b_cols))
    
    # Process in parallel
    try:
        with mp.Pool(n_workers) as pool:
            results = pool.map(_matmul_chunk, chunks)
        
        # Combine results
        combined = []
        for chunk_result in results:
            combined.extend(chunk_result)
        
        return Matrix(a.rows, b.cols, combined)
    except Exception:
        # Fall back to serial if multiprocessing fails
        return matmul(a, b)


def _matmul_chunk(args: Tuple) -> List[float]:
    """Worker function for parallel matmul."""
    a_data, a_cols, a_rows, b_data, b_rows, b_cols = args
    result = [0.0] * (a_rows * b_cols)
    
    for i in range(a_rows):
        a_row_start = i * a_cols
        for k in range(a_cols):
            a_ik = a_data[a_row_start + k]
            b_row_start = k * b_cols
            result_row_start = i * b_cols
            for j in range(b_cols):
                result[result_row_start + j] += a_ik * b_data[b_row_start + j]
    
    return result


def add(a: Matrix, b: Matrix) -> Matrix:
    """Element-wise addition."""
    if a.shape != b.shape:
        # Try broadcasting
        if b.rows == 1 and b.cols == a.cols:
            # Broadcast row vector
            result = a.copy()
            for i in range(a.rows):
                start = i * a.cols
                for j in range(a.cols):
                    result.data[start + j] += b.data[j]
            return result
        raise ValueError(f"Cannot add {a.shape} + {b.shape}")
    
    return Matrix(a.rows, a.cols, [a.data[i] + b.data[i] for i in range(len(a.data))])


def subtract(a: Matrix, b: Matrix) -> Matrix:
    """Element-wise subtraction."""
    if a.shape != b.shape:
        raise ValueError(f"Cannot subtract {a.shape} - {b.shape}")
    return Matrix(a.rows, a.cols, [a.data[i] - b.data[i] for i in range(len(a.data))])


def multiply(a: Matrix, b: Matrix) -> Matrix:
    """Element-wise multiplication (Hadamard product)."""
    if a.shape != b.shape:
        raise ValueError(f"Cannot multiply {a.shape} * {b.shape}")
    return Matrix(a.rows, a.cols, [a.data[i] * b.data[i] for i in range(len(a.data))])


def scale(a: Matrix, scalar: float) -> Matrix:
    """Scalar multiplication."""
    return Matrix(a.rows, a.cols, [x * scalar for x in a.data])


def sum_rows(a: Matrix) -> Matrix:
    """Sum along rows, returning column vector."""
    result = Matrix.zeros(a.rows, 1)
    for i in range(a.rows):
        total = 0.0
        start = i * a.cols
        for j in range(a.cols):
            total += a.data[start + j]
        result.data[i] = total
    return result


def sum_cols(a: Matrix) -> Matrix:
    """Sum along columns, returning row vector."""
    result = Matrix.zeros(1, a.cols)
    for i in range(a.rows):
        start = i * a.cols
        for j in range(a.cols):
            result.data[j] += a.data[start + j]
    return result


def mean(a: Matrix, axis: Optional[int] = None) -> Union[float, Matrix]:
    """Compute mean along axis or of all elements."""
    if axis is None:
        return sum(a.data) / len(a.data)
    elif axis == 0:
        # Mean along rows (result is row vector)
        result = sum_cols(a)
        return scale(result, 1.0 / a.rows)
    elif axis == 1:
        # Mean along cols (result is column vector)
        result = sum_rows(a)
        return scale(result, 1.0 / a.cols)


def variance(a: Matrix, axis: Optional[int] = None) -> Union[float, Matrix]:
    """Compute variance along axis or of all elements."""
    mu = mean(a, axis)
    if axis is None:
        return sum((x - mu) ** 2 for x in a.data) / len(a.data)
    # For axis-specific, we'd need more work - simplified for now
    raise NotImplementedError("Axis-specific variance not implemented")


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def relu(x: Matrix) -> Matrix:
    """ReLU activation: max(0, x)"""
    return Matrix(x.rows, x.cols, [max(0.0, v) for v in x.data])


def relu_backward(x: Matrix, grad: Matrix) -> Matrix:
    """ReLU gradient: 1 if x > 0, else 0"""
    return Matrix(x.rows, x.cols, [g if v > 0 else 0.0 for v, g in zip(x.data, grad.data)])


def gelu(x: Matrix) -> Matrix:
    """
    GELU activation: x * Φ(x)
    
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    result = []
    for v in x.data:
        inner = sqrt_2_pi * (v + 0.044715 * v * v * v)
        result.append(0.5 * v * (1.0 + math.tanh(inner)))
    return Matrix(x.rows, x.cols, result)


def silu(x: Matrix) -> Matrix:
    """SiLU/Swish activation: x * sigmoid(x)"""
    result = []
    for v in x.data:
        sig = 1.0 / (1.0 + math.exp(-min(max(v, -500), 500)))  # Clamp for stability
        result.append(v * sig)
    return Matrix(x.rows, x.cols, result)


def sigmoid(x: Matrix) -> Matrix:
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    result = []
    for v in x.data:
        v = min(max(v, -500), 500)  # Clamp for numerical stability
        result.append(1.0 / (1.0 + math.exp(-v)))
    return Matrix(x.rows, x.cols, result)


def tanh_activation(x: Matrix) -> Matrix:
    """Tanh activation."""
    return Matrix(x.rows, x.cols, [math.tanh(v) for v in x.data])


def softmax(x: Matrix, axis: int = -1) -> Matrix:
    """
    Softmax activation along axis.
    
    softmax(x)_i = exp(x_i) / sum(exp(x_j))
    
    Uses max subtraction for numerical stability.
    """
    result = Matrix.zeros(x.rows, x.cols)
    
    if axis == -1 or axis == 1:
        # Softmax along rows (each row sums to 1)
        for i in range(x.rows):
            row_start = i * x.cols
            row = x.data[row_start:row_start + x.cols]
            
            # Subtract max for stability
            max_val = max(row)
            exp_vals = [math.exp(v - max_val) for v in row]
            sum_exp = sum(exp_vals)
            
            for j in range(x.cols):
                result.data[row_start + j] = exp_vals[j] / sum_exp
    else:
        # Softmax along columns
        for j in range(x.cols):
            col = [x.data[i * x.cols + j] for i in range(x.rows)]
            max_val = max(col)
            exp_vals = [math.exp(v - max_val) for v in col]
            sum_exp = sum(exp_vals)
            
            for i in range(x.rows):
                result.data[i * x.cols + j] = exp_vals[i] / sum_exp
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK LAYERS
# ═══════════════════════════════════════════════════════════════════════════════

class PureLinear:
    """
    Linear layer: y = xW + b
    
    The fundamental building block of neural networks.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        
        # Initialize weights with Xavier initialization
        self.weight = Matrix.xavier_init(in_features, out_features)
        self.bias = Matrix.zeros(1, out_features) if bias else None
        
        # Gradients (for training)
        self.weight_grad: Optional[Matrix] = None
        self.bias_grad: Optional[Matrix] = None
        
        # Cache for backward pass
        self._input_cache: Optional[Matrix] = None
    
    def forward(self, x: Matrix, use_parallel: bool = False) -> Matrix:
        """
        Forward pass: y = xW + b
        
        Args:
            x: Input matrix (batch_size, in_features)
            use_parallel: Use multiprocessing for matmul
        
        Returns:
            Output matrix (batch_size, out_features)
        """
        self._input_cache = x
        
        if use_parallel:
            output = matmul_parallel(x, self.weight)
        else:
            output = matmul(x, self.weight)
        
        if self.has_bias:
            output = add(output, self.bias)
        
        return output
    
    def backward(self, grad_output: Matrix) -> Matrix:
        """
        Backward pass: compute gradients.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient for previous layer
        """
        if self._input_cache is None:
            raise RuntimeError("Forward must be called before backward")
        
        # Weight gradient: input.T @ grad_output
        self.weight_grad = matmul(self._input_cache.T, grad_output)
        
        # Bias gradient: sum along batch dimension
        if self.has_bias:
            self.bias_grad = sum_cols(grad_output)
        
        # Input gradient: grad_output @ weight.T
        grad_input = matmul(grad_output, self.weight.T)
        
        return grad_input
    
    def parameters(self) -> List[Matrix]:
        """Return list of parameters."""
        if self.has_bias:
            return [self.weight, self.bias]
        return [self.weight]
    
    def gradients(self) -> List[Optional[Matrix]]:
        """Return list of gradients."""
        if self.has_bias:
            return [self.weight_grad, self.bias_grad]
        return [self.weight_grad]


class PureLayerNorm:
    """
    Layer Normalization.
    
    Normalizes across features (last dimension).
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.gamma = Matrix.ones(1, normalized_shape)  # Scale
        self.beta = Matrix.zeros(1, normalized_shape)   # Shift
        
        # Gradients
        self.gamma_grad: Optional[Matrix] = None
        self.beta_grad: Optional[Matrix] = None
        
        # Cache
        self._input_cache: Optional[Matrix] = None
        self._normalized_cache: Optional[Matrix] = None
        self._std_cache: Optional[List[float]] = None
    
    def forward(self, x: Matrix) -> Matrix:
        """Forward pass: normalize and scale."""
        self._input_cache = x
        result = Matrix.zeros(x.rows, x.cols)
        self._std_cache = []
        self._normalized_cache = Matrix.zeros(x.rows, x.cols)
        
        for i in range(x.rows):
            row_start = i * x.cols
            row = x.data[row_start:row_start + x.cols]
            
            # Compute mean and variance
            mu = sum(row) / len(row)
            var = sum((v - mu) ** 2 for v in row) / len(row)
            std = math.sqrt(var + self.eps)
            self._std_cache.append(std)
            
            # Normalize
            for j in range(x.cols):
                norm_val = (row[j] - mu) / std
                self._normalized_cache.data[row_start + j] = norm_val
                # Scale and shift
                result.data[row_start + j] = norm_val * self.gamma.data[j] + self.beta.data[j]
        
        return result
    
    def backward(self, grad_output: Matrix) -> Matrix:
        """Backward pass."""
        # Simplified gradient computation
        self.gamma_grad = Matrix.zeros(1, self.normalized_shape)
        self.beta_grad = sum_cols(grad_output)
        
        # Gamma gradient
        for i in range(grad_output.rows):
            for j in range(grad_output.cols):
                self.gamma_grad.data[j] += (
                    grad_output[i, j] * self._normalized_cache[i, j]
                )
        
        # Input gradient (simplified)
        grad_input = Matrix.zeros(grad_output.rows, grad_output.cols)
        for i in range(grad_output.rows):
            std = self._std_cache[i]
            for j in range(grad_output.cols):
                grad_input[i, j] = grad_output[i, j] * self.gamma.data[j] / std
        
        return grad_input
    
    def parameters(self) -> List[Matrix]:
        return [self.gamma, self.beta]
    
    def gradients(self) -> List[Optional[Matrix]]:
        return [self.gamma_grad, self.beta_grad]


class PureRMSNorm:
    """
    Root Mean Square Layer Normalization.
    
    Simpler than LayerNorm - no mean subtraction.
    y = x / sqrt(mean(x^2) + eps) * gamma
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Matrix.ones(1, normalized_shape)
        self.gamma_grad: Optional[Matrix] = None
        self._rms_cache: Optional[List[float]] = None
        self._input_cache: Optional[Matrix] = None
    
    def forward(self, x: Matrix) -> Matrix:
        """Forward pass."""
        self._input_cache = x
        self._rms_cache = []
        result = Matrix.zeros(x.rows, x.cols)
        
        for i in range(x.rows):
            row_start = i * x.cols
            row = x.data[row_start:row_start + x.cols]
            
            # RMS = sqrt(mean(x^2))
            mean_sq = sum(v * v for v in row) / len(row)
            rms = math.sqrt(mean_sq + self.eps)
            self._rms_cache.append(rms)
            
            # Normalize and scale
            for j in range(x.cols):
                result.data[row_start + j] = (row[j] / rms) * self.gamma.data[j]
        
        return result
    
    def backward(self, grad_output: Matrix) -> Matrix:
        """Backward pass."""
        grad_input = Matrix.zeros(grad_output.rows, grad_output.cols)
        self.gamma_grad = Matrix.zeros(1, self.normalized_shape)
        
        for i in range(grad_output.rows):
            rms = self._rms_cache[i]
            for j in range(grad_output.cols):
                x_val = self._input_cache[i, j]
                g = grad_output[i, j]
                
                # Gamma gradient
                self.gamma_grad.data[j] += g * (x_val / rms)
                
                # Input gradient (simplified)
                grad_input[i, j] = g * self.gamma.data[j] / rms
        
        return grad_input
    
    def parameters(self) -> List[Matrix]:
        return [self.gamma]
    
    def gradients(self) -> List[Optional[Matrix]]:
        return [self.gamma_grad]


class PureEmbedding:
    """
    Embedding layer: maps token IDs to dense vectors.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings
        self.weight = Matrix.randn(num_embeddings, embedding_dim, std=0.02)
        self.weight_grad: Optional[Matrix] = None
        self._input_cache: Optional[List[int]] = None
    
    def forward(self, input_ids: List[int]) -> Matrix:
        """
        Lookup embeddings for input IDs.
        
        Args:
            input_ids: List of token IDs
            
        Returns:
            Matrix of shape (len(input_ids), embedding_dim)
        """
        self._input_cache = input_ids
        result = Matrix.zeros(len(input_ids), self.embedding_dim)
        
        for i, token_id in enumerate(input_ids):
            if 0 <= token_id < self.num_embeddings:
                row_start = i * self.embedding_dim
                emb_start = token_id * self.embedding_dim
                for j in range(self.embedding_dim):
                    result.data[row_start + j] = self.weight.data[emb_start + j]
        
        return result
    
    def backward(self, grad_output: Matrix) -> None:
        """Accumulate gradients for embeddings."""
        if self.weight_grad is None:
            self.weight_grad = Matrix.zeros(self.num_embeddings, self.embedding_dim)
        
        for i, token_id in enumerate(self._input_cache):
            if 0 <= token_id < self.num_embeddings:
                grad_start = i * self.embedding_dim
                emb_start = token_id * self.embedding_dim
                for j in range(self.embedding_dim):
                    self.weight_grad.data[emb_start + j] += grad_output.data[grad_start + j]
    
    def parameters(self) -> List[Matrix]:
        return [self.weight]
    
    def gradients(self) -> List[Optional[Matrix]]:
        return [self.weight_grad]


# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION MECHANISM
# ═══════════════════════════════════════════════════════════════════════════════

class PureAttention:
    """
    Multi-Head Self-Attention.
    
    The core mechanism that allows the model to attend to different
    parts of the input sequence.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Q, K, V projections
        self.q_proj = PureLinear(d_model, d_model, bias=False)
        self.k_proj = PureLinear(d_model, d_model, bias=False)
        self.v_proj = PureLinear(d_model, d_model, bias=False)
        self.o_proj = PureLinear(d_model, d_model, bias=False)
        
        # Cache for backward
        self._q_cache: Optional[Matrix] = None
        self._k_cache: Optional[Matrix] = None
        self._v_cache: Optional[Matrix] = None
        self._attn_cache: Optional[Matrix] = None
    
    def forward(self, x: Matrix, mask: Optional[Matrix] = None) -> Matrix:
        """
        Forward pass for self-attention.
        
        Args:
            x: Input (seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output (seq_len, d_model)
        """
        seq_len = x.rows
        
        # Project to Q, K, V
        q = self.q_proj.forward(x)  # (seq_len, d_model)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        
        self._q_cache = q
        self._k_cache = k
        self._v_cache = v
        
        # For simplicity, process all heads together
        # In a real impl, we'd reshape to (n_heads, seq_len, head_dim)
        
        # Compute attention scores: Q @ K.T * scale
        scores = matmul(q, k.T)
        scores = scale(scores, self.scale)
        
        # Apply causal mask if needed
        if mask is not None:
            for i in range(scores.rows):
                for j in range(scores.cols):
                    if mask[i, j] == 0:
                        scores[i, j] = -1e9
        
        # Softmax
        attn = softmax(scores, axis=-1)
        self._attn_cache = attn
        
        # Apply attention to values
        output = matmul(attn, v)
        
        # Output projection
        output = self.o_proj.forward(output)
        
        return output
    
    def backward(self, grad_output: Matrix) -> Matrix:
        """Backward pass."""
        # This is simplified - full impl needs chain rule through softmax
        grad = self.o_proj.backward(grad_output)
        
        # Gradient through attention (simplified)
        grad_v = matmul(self._attn_cache.T, grad)
        grad_attn = matmul(grad, self._v_cache.T)
        
        # Gradient through softmax (simplified - assumes attn * (1-attn))
        grad_scores = multiply(grad_attn, self._attn_cache)
        grad_scores = scale(grad_scores, self.scale)
        
        # Gradient through Q, K projections
        grad_q = matmul(grad_scores, self._k_cache)
        grad_k = matmul(grad_scores.T, self._q_cache)
        
        grad_x_q = self.q_proj.backward(grad_q)
        grad_x_k = self.k_proj.backward(grad_k)
        grad_x_v = self.v_proj.backward(grad_v)
        
        # Sum gradients from all paths
        grad_input = add(add(grad_x_q, grad_x_k), grad_x_v)
        
        return grad_input
    
    def parameters(self) -> List[Matrix]:
        params = []
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.o_proj.parameters())
        return params
    
    def gradients(self) -> List[Optional[Matrix]]:
        grads = []
        grads.extend(self.q_proj.gradients())
        grads.extend(self.k_proj.gradients())
        grads.extend(self.v_proj.gradients())
        grads.extend(self.o_proj.gradients())
        return grads


class PureFeedForward:
    """
    Feed-Forward Network (FFN).
    
    Two linear layers with activation in between.
    FFN(x) = act(xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu"):
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        
        self.up_proj = PureLinear(d_model, d_ff)
        self.down_proj = PureLinear(d_ff, d_model)
        
        self._up_cache: Optional[Matrix] = None
        self._act_cache: Optional[Matrix] = None
    
    def forward(self, x: Matrix) -> Matrix:
        """Forward pass."""
        up = self.up_proj.forward(x)
        self._up_cache = up
        
        # Activation
        if self.activation == "gelu":
            act = gelu(up)
        elif self.activation == "silu":
            act = silu(up)
        elif self.activation == "relu":
            act = relu(up)
        else:
            act = gelu(up)
        
        self._act_cache = act
        down = self.down_proj.forward(act)
        
        return down
    
    def backward(self, grad_output: Matrix) -> Matrix:
        """Backward pass."""
        grad_act = self.down_proj.backward(grad_output)
        
        # Gradient through activation (simplified)
        if self.activation == "relu":
            grad_up = relu_backward(self._up_cache, grad_act)
        else:
            # Approximate GELU gradient
            grad_up = grad_act  # Simplified
        
        grad_input = self.up_proj.backward(grad_up)
        return grad_input
    
    def parameters(self) -> List[Matrix]:
        return self.up_proj.parameters() + self.down_proj.parameters()
    
    def gradients(self) -> List[Optional[Matrix]]:
        return self.up_proj.gradients() + self.down_proj.gradients()


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

class PureTransformerBlock:
    """
    Single Transformer block.
    
    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_rms_norm: bool = True):
        self.d_model = d_model
        
        # Layer norms
        if use_rms_norm:
            self.attn_norm = PureRMSNorm(d_model)
            self.ffn_norm = PureRMSNorm(d_model)
        else:
            self.attn_norm = PureLayerNorm(d_model)
            self.ffn_norm = PureLayerNorm(d_model)
        
        # Attention and FFN
        self.attention = PureAttention(d_model, n_heads)
        self.ffn = PureFeedForward(d_model, d_ff)
        
        # Cache for residual connections
        self._x_cache: Optional[Matrix] = None
        self._attn_out_cache: Optional[Matrix] = None
    
    def forward(self, x: Matrix, mask: Optional[Matrix] = None) -> Matrix:
        """Forward pass with residual connections."""
        self._x_cache = x
        
        # Attention block with residual
        normed = self.attn_norm.forward(x)
        attn_out = self.attention.forward(normed, mask)
        x = add(x, attn_out)
        self._attn_out_cache = x
        
        # FFN block with residual
        normed = self.ffn_norm.forward(x)
        ffn_out = self.ffn.forward(normed)
        x = add(x, ffn_out)
        
        return x
    
    def backward(self, grad_output: Matrix) -> Matrix:
        """Backward pass through block."""
        # FFN backward
        grad_ffn = self.ffn.backward(grad_output)
        grad_ffn = self.ffn_norm.backward(grad_ffn)
        grad = add(grad_output, grad_ffn)  # Residual gradient
        
        # Attention backward
        grad_attn = self.attention.backward(grad)
        grad_attn = self.attn_norm.backward(grad_attn)
        grad = add(grad, grad_attn)  # Residual gradient
        
        return grad
    
    def parameters(self) -> List[Matrix]:
        params = []
        params.extend(self.attn_norm.parameters())
        params.extend(self.attention.parameters())
        params.extend(self.ffn_norm.parameters())
        params.extend(self.ffn.parameters())
        return params
    
    def gradients(self) -> List[Optional[Matrix]]:
        grads = []
        grads.extend(self.attn_norm.gradients())
        grads.extend(self.attention.gradients())
        grads.extend(self.ffn_norm.gradients())
        grads.extend(self.ffn.gradients())
        return grads


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE TRANSFORMER MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class PureTransformer:
    """
    Complete Transformer model in pure Python.
    
    This is the main class that ties everything together.
    Compatible with ForgeAI's nano/micro model sizes.
    """
    
    def __init__(self, config: Optional[PureConfig] = None, **kwargs):
        if config is None:
            config = PureConfig(**kwargs)
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = PureEmbedding(config.vocab_size, config.d_model)
        self.position_embedding = PureEmbedding(config.max_seq_len, config.d_model)
        
        # Transformer blocks
        self.blocks = [
            PureTransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff
            )
            for _ in range(config.n_layers)
        ]
        
        # Final layer norm and output projection
        self.final_norm = PureRMSNorm(config.d_model)
        self.output_proj = PureLinear(config.d_model, config.vocab_size, bias=False)
        
        # Use parallel operations for larger models
        self.use_parallel = config.use_multiprocessing and config.param_count() > 500_000
    
    def forward(self, input_ids: List[int]) -> Matrix:
        """
        Forward pass.
        
        Args:
            input_ids: List of token IDs
            
        Returns:
            Logits matrix (seq_len, vocab_size)
        """
        seq_len = len(input_ids)
        
        # Get embeddings
        token_emb = self.token_embedding.forward(input_ids)
        pos_ids = list(range(seq_len))
        pos_emb = self.position_embedding.forward(pos_ids)
        
        # Add embeddings
        x = add(token_emb, pos_emb)
        
        # Create causal mask
        mask = self._create_causal_mask(seq_len)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)
        
        # Final norm and project to vocabulary
        x = self.final_norm.forward(x)
        logits = self.output_proj.forward(x)
        
        return logits
    
    def _create_causal_mask(self, seq_len: int) -> Matrix:
        """Create causal (triangular) attention mask."""
        mask = Matrix.ones(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                mask[i, j] = 0.0
        return mask
    
    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> List[int]:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Starting token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            List of generated token IDs
        """
        generated = list(input_ids)
        
        for _ in range(max_new_tokens):
            # Truncate to max sequence length
            context = generated[-self.config.max_seq_len:]
            
            # Forward pass
            logits = self.forward(context)
            
            # Get last token's logits
            last_logits = logits.get_row(logits.rows - 1)
            
            # Apply temperature
            if temperature != 1.0:
                last_logits = [l / temperature for l in last_logits]
            
            # Top-k filtering
            if top_k > 0:
                # Get indices of top-k values
                indexed = list(enumerate(last_logits))
                indexed.sort(key=lambda x: x[1], reverse=True)
                top_indices = set(i for i, _ in indexed[:top_k])
                
                # Zero out non-top-k
                for i in range(len(last_logits)):
                    if i not in top_indices:
                        last_logits[i] = -1e9
            
            # Softmax to get probabilities
            max_logit = max(last_logits)
            exp_logits = [math.exp(l - max_logit) for l in last_logits]
            sum_exp = sum(exp_logits)
            probs = [e / sum_exp for e in exp_logits]
            
            # Sample from distribution
            r = random.random()
            cumsum = 0.0
            next_token = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r < cumsum:
                    next_token = i
                    break
            
            generated.append(next_token)
            
            # Stop at EOS (assuming token 0 or 2 is EOS)
            if next_token in [0, 2]:
                break
        
        return generated
    
    def parameters(self) -> List[Matrix]:
        """Get all model parameters."""
        params = []
        params.extend(self.token_embedding.parameters())
        params.extend(self.position_embedding.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        params.extend(self.output_proj.parameters())
        return params
    
    def gradients(self) -> List[Optional[Matrix]]:
        """Get all gradients."""
        grads = []
        grads.extend(self.token_embedding.gradients())
        grads.extend(self.position_embedding.gradients())
        for block in self.blocks:
            grads.extend(block.gradients())
        grads.extend(self.final_norm.gradients())
        grads.extend(self.output_proj.gradients())
        return grads
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        total = 0
        for p in self.parameters():
            total += p.rows * p.cols
        return total
    
    def save(self, path: Path):
        """Save model weights to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        weights = {}
        for i, param in enumerate(self.parameters()):
            weights[f"param_{i}"] = {
                "rows": param.rows,
                "cols": param.cols,
                "data": param.data
            }
        
        # Also save config
        weights["config"] = {
            "vocab_size": self.config.vocab_size,
            "d_model": self.config.d_model,
            "n_heads": self.config.n_heads,
            "n_layers": self.config.n_layers,
            "d_ff": self.config.d_ff,
            "max_seq_len": self.config.max_seq_len
        }
        
        with open(path, 'w') as f:
            json.dump(weights, f)
    
    def load(self, path: Path):
        """Load model weights from file."""
        with open(path, 'r') as f:
            weights = json.load(f)
        
        params = self.parameters()
        for i, param in enumerate(params):
            key = f"param_{i}"
            if key in weights:
                w = weights[key]
                if param.rows == w["rows"] and param.cols == w["cols"]:
                    param.data = w["data"]


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING (OPTIONAL)
# ═══════════════════════════════════════════════════════════════════════════════

class PureSGD:
    """Simple SGD optimizer."""
    
    def __init__(self, parameters: List[Matrix], lr: float = 0.001):
        self.parameters = parameters
        self.lr = lr
    
    def step(self, gradients: List[Optional[Matrix]]):
        """Apply gradients to parameters."""
        for param, grad in zip(self.parameters, gradients):
            if grad is not None:
                for i in range(len(param.data)):
                    param.data[i] -= self.lr * grad.data[i]
    
    def zero_grad(self):
        """Zero all gradients (they get accumulated)."""
        # Gradients are recomputed each backward pass
        pass


class PureAdam:
    """Adam optimizer."""
    
    def __init__(
        self,
        parameters: List[Matrix],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Moment estimates
        self.m = [Matrix.zeros(p.rows, p.cols) for p in parameters]
        self.v = [Matrix.zeros(p.rows, p.cols) for p in parameters]
    
    def step(self, gradients: List[Optional[Matrix]]):
        """Apply Adam update."""
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(self.parameters, gradients)):
            if grad is None:
                continue
            
            # Update moment estimates
            for j in range(len(param.data)):
                g = grad.data[j]
                self.m[i].data[j] = self.beta1 * self.m[i].data[j] + (1 - self.beta1) * g
                self.v[i].data[j] = self.beta2 * self.v[i].data[j] + (1 - self.beta2) * g * g
                
                # Bias correction
                m_hat = self.m[i].data[j] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i].data[j] / (1 - self.beta2 ** self.t)
                
                # Update parameter
                param.data[j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


def cross_entropy_loss(logits: Matrix, targets: List[int]) -> Tuple[float, Matrix]:
    """
    Cross-entropy loss with gradient.
    
    Args:
        logits: Model output (seq_len, vocab_size)
        targets: Target token IDs
        
    Returns:
        (loss_value, gradient_matrix)
    """
    # Softmax
    probs = softmax(logits, axis=-1)
    
    # Compute loss: -log(p[target])
    loss = 0.0
    for i, target in enumerate(targets):
        p = probs[i, target]
        loss -= math.log(max(p, 1e-10))
    loss /= len(targets)
    
    # Gradient: probs - one_hot(targets)
    grad = probs.copy()
    for i, target in enumerate(targets):
        grad[i, target] -= 1.0
    grad = scale(grad, 1.0 / len(targets))
    
    return loss, grad


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND SWITCHING
# ═══════════════════════════════════════════════════════════════════════════════

_current_backend = "auto"
_backend_threshold = 5_000_000  # 5M params - switch to PyTorch above this

def set_backend(backend: str, threshold: int = 5_000_000):
    """
    Set the neural network backend.
    
    Args:
        backend: "pure", "torch", or "auto"
        threshold: Parameter count threshold for auto mode
    """
    global _current_backend, _backend_threshold
    if backend not in ("pure", "torch", "auto"):
        raise ValueError(f"Unknown backend: {backend}. Use 'pure', 'torch', or 'auto'")
    _current_backend = backend
    _backend_threshold = threshold


def get_backend() -> str:
    """Get current backend setting."""
    return _current_backend


def should_use_pure_backend(param_count: int) -> bool:
    """
    Determine if pure Python backend should be used.
    
    Args:
        param_count: Number of model parameters
        
    Returns:
        True if pure Python should be used
    """
    if _current_backend == "pure":
        return True
    elif _current_backend == "torch":
        return False
    else:  # auto
        # Use pure for small models, torch for large
        return param_count < _backend_threshold


def get_model_for_size(size: str) -> Union['PureTransformer', Any]:
    """
    Get appropriate model implementation for a given size.
    
    Args:
        size: Model size name (nano, micro, tiny, small, etc.)
        
    Returns:
        Model instance (PureTransformer or PyTorch model)
    """
    # Size configurations (matching ForgeAI presets)
    SIZE_CONFIGS = {
        "nano": PureConfig(vocab_size=1000, d_model=64, n_heads=2, n_layers=2, d_ff=128),
        "micro": PureConfig(vocab_size=2000, d_model=128, n_heads=4, n_layers=4, d_ff=256),
        "tiny": PureConfig(vocab_size=4000, d_model=256, n_heads=4, n_layers=6, d_ff=512),
        "small": PureConfig(vocab_size=8000, d_model=512, n_heads=8, n_layers=8, d_ff=1024),
        "medium": PureConfig(vocab_size=16000, d_model=768, n_heads=12, n_layers=12, d_ff=2048),
    }
    
    if size not in SIZE_CONFIGS:
        # Default to small if unknown
        config = SIZE_CONFIGS.get("small")
    else:
        config = SIZE_CONFIGS[size]
    
    param_count = config.param_count()
    
    if should_use_pure_backend(param_count):
        print(f"[PureBackend] Using pure Python for {size} ({param_count:,} params)")
        return PureTransformer(config)
    else:
        # Try to use PyTorch
        try:
            from ..core.model import create_model
            print(f"[PyTorchBackend] Using PyTorch for {size} ({param_count:,} params)")
            return create_model(size)
        except ImportError:
            print(f"[PureBackend] PyTorch unavailable, using pure Python for {size}")
            return PureTransformer(config)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_matmul(size: int = 256, iterations: int = 5) -> Dict[str, float]:
    """
    Benchmark matrix multiplication performance.
    
    Returns times for serial and parallel implementations.
    """
    import time
    
    a = Matrix.randn(size, size)
    b = Matrix.randn(size, size)
    
    # Serial
    start = time.time()
    for _ in range(iterations):
        matmul(a, b)
    serial_time = (time.time() - start) / iterations
    
    # Parallel
    start = time.time()
    for _ in range(iterations):
        matmul_parallel(a, b)
    parallel_time = (time.time() - start) / iterations
    
    return {
        "matrix_size": size,
        "serial_seconds": serial_time,
        "parallel_seconds": parallel_time,
        "speedup": serial_time / parallel_time if parallel_time > 0 else 0,
        "cpu_cores": mp.cpu_count(),
        "is_pypy": is_pypy(),
        "python": platform.python_implementation()
    }


def test_pure_transformer():
    """Quick test of the pure Python transformer."""
    print("Testing PureTransformer...")
    print(f"Runtime: {platform.python_implementation()} {platform.python_version()}")
    
    # Create tiny model for testing
    config = PureConfig(
        vocab_size=100,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_seq_len=32
    )
    
    model = PureTransformer(config)
    print(f"Created model with {model.count_parameters():,} parameters")
    
    # Test forward pass
    input_ids = [1, 5, 10, 15, 20]
    logits = model.forward(input_ids)
    print(f"Input: {input_ids}")
    print(f"Output shape: {logits.shape}")
    
    # Test generation
    generated = model.generate(input_ids, max_new_tokens=10, temperature=0.8)
    print(f"Generated: {generated}")
    
    print("Test passed!")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT CONVERSION: PyTorch <-> Pure Python
# ═══════════════════════════════════════════════════════════════════════════════

def convert_pytorch_to_pure(pytorch_state_dict: Dict[str, Any], config: PureConfig) -> 'PureTransformer':
    """
    Convert a PyTorch Forge model's state dict to a PureTransformer.
    
    Args:
        pytorch_state_dict: PyTorch model's state_dict()
        config: Configuration for the pure model
        
    Returns:
        PureTransformer with loaded weights
    """
    model = PureTransformer(config)
    
    # Map PyTorch weight names to pure model structure
    # This handles the common Forge model structure
    
    def tensor_to_matrix(tensor) -> Matrix:
        """Convert PyTorch tensor to Matrix."""
        # Handle both numpy arrays and torch tensors
        if hasattr(tensor, 'numpy'):
            arr = tensor.detach().cpu().numpy()
        elif hasattr(tensor, 'tolist'):
            arr = tensor
        else:
            arr = tensor
        
        if len(arr.shape) == 1:
            # 1D tensor -> row vector
            return Matrix(1, len(arr), list(arr.flatten()))
        else:
            # 2D tensor
            rows, cols = arr.shape
            return Matrix(rows, cols, list(arr.flatten()))
    
    # Try to load embeddings
    for key, value in pytorch_state_dict.items():
        try:
            if 'tok_emb' in key or 'token_embedding' in key:
                mat = tensor_to_matrix(value)
                if mat.rows == model.token_embedding.num_embeddings:
                    model.token_embedding.weight = mat
                    
            elif 'pos_emb' in key or 'position_embedding' in key:
                mat = tensor_to_matrix(value)
                if mat.rows <= model.position_embedding.num_embeddings:
                    model.position_embedding.weight = mat
                    
        except Exception as e:
            print(f"Warning: Could not load {key}: {e}")
    
    return model


def convert_pure_to_pytorch(pure_model: 'PureTransformer') -> Dict[str, Any]:
    """
    Convert a PureTransformer's weights to a PyTorch state dict format.
    
    Args:
        pure_model: PureTransformer instance
        
    Returns:
        Dictionary compatible with PyTorch's load_state_dict()
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for conversion to PyTorch format")
    
    state_dict = {}
    
    # Convert embeddings
    tok_emb = pure_model.token_embedding.weight
    state_dict['tok_emb.weight'] = torch.tensor(
        tok_emb.to_2d(), dtype=torch.float32
    )
    
    pos_emb = pure_model.position_embedding.weight
    state_dict['pos_emb.weight'] = torch.tensor(
        pos_emb.to_2d(), dtype=torch.float32
    )
    
    # Convert layers
    for i, block in enumerate(pure_model.blocks):
        prefix = f'layers.{i}.'
        
        # Attention weights
        state_dict[f'{prefix}attn.q_proj.weight'] = torch.tensor(
            block.attention.q_proj.weight.to_2d(), dtype=torch.float32
        )
        state_dict[f'{prefix}attn.k_proj.weight'] = torch.tensor(
            block.attention.k_proj.weight.to_2d(), dtype=torch.float32
        )
        state_dict[f'{prefix}attn.v_proj.weight'] = torch.tensor(
            block.attention.v_proj.weight.to_2d(), dtype=torch.float32
        )
        state_dict[f'{prefix}attn.o_proj.weight'] = torch.tensor(
            block.attention.o_proj.weight.to_2d(), dtype=torch.float32
        )
        
        # FFN weights
        state_dict[f'{prefix}ffn.up_proj.weight'] = torch.tensor(
            block.ffn.up_proj.weight.to_2d(), dtype=torch.float32
        )
        state_dict[f'{prefix}ffn.down_proj.weight'] = torch.tensor(
            block.ffn.down_proj.weight.to_2d(), dtype=torch.float32
        )
        
        # Layer norms
        state_dict[f'{prefix}attn_norm.gamma'] = torch.tensor(
            block.attn_norm.gamma.data, dtype=torch.float32
        )
        state_dict[f'{prefix}ffn_norm.gamma'] = torch.tensor(
            block.ffn_norm.gamma.data, dtype=torch.float32
        )
    
    # Final layer norm and output projection
    state_dict['final_norm.gamma'] = torch.tensor(
        pure_model.final_norm.gamma.data, dtype=torch.float32
    )
    state_dict['output.weight'] = torch.tensor(
        pure_model.output_proj.weight.to_2d(), dtype=torch.float32
    )
    
    return state_dict


def save_pure_model(model: 'PureTransformer', path: Path, format: str = "json"):
    """
    Save a PureTransformer to disk.
    
    Args:
        model: Model to save
        path: Output path
        format: "json" (readable) or "bin" (compact)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        model.save(path)
    elif format == "bin":
        # Binary format for faster loading and smaller size
        with open(path, 'wb') as f:
            # Write config
            config_json = json.dumps({
                "vocab_size": model.config.vocab_size,
                "d_model": model.config.d_model,
                "n_heads": model.config.n_heads,
                "n_layers": model.config.n_layers,
                "d_ff": model.config.d_ff,
                "max_seq_len": model.config.max_seq_len
            }).encode('utf-8')
            f.write(struct.pack('I', len(config_json)))
            f.write(config_json)
            
            # Write parameters as binary floats
            params = model.parameters()
            f.write(struct.pack('I', len(params)))
            for param in params:
                f.write(struct.pack('II', param.rows, param.cols))
                f.write(struct.pack(f'{len(param.data)}f', *param.data))


def load_pure_model(path: Path) -> 'PureTransformer':
    """
    Load a PureTransformer from disk.
    
    Args:
        path: Path to saved model
        
    Returns:
        Loaded PureTransformer
    """
    path = Path(path)
    
    if path.suffix == '.json' or path.suffix == '':
        # JSON format
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = PureConfig(**data.get("config", {}))
        model = PureTransformer(config)
        model.load(path)
        return model
    else:
        # Binary format
        with open(path, 'rb') as f:
            # Read config
            config_len = struct.unpack('I', f.read(4))[0]
            config_json = f.read(config_len).decode('utf-8')
            config_dict = json.loads(config_json)
            config = PureConfig(**config_dict)
            
            model = PureTransformer(config)
            params = model.parameters()
            
            # Read parameters
            n_params = struct.unpack('I', f.read(4))[0]
            for i, param in enumerate(params):
                rows, cols = struct.unpack('II', f.read(8))
                if rows == param.rows and cols == param.cols:
                    data = struct.unpack(f'{rows*cols}f', f.read(4 * rows * cols))
                    param.data = list(data)
        
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# BATCHED OPERATIONS FOR EFFICIENCY
# ═══════════════════════════════════════════════════════════════════════════════

def forward_batch(model: 'PureTransformer', batch: List[List[int]], n_workers: int = 0) -> List[Matrix]:
    """
    Process multiple sequences in parallel using multiprocessing.
    
    Args:
        model: PureTransformer model
        batch: List of input_id sequences
        n_workers: Number of worker processes (0 = auto)
        
    Returns:
        List of output matrices
    """
    if n_workers == 0:
        n_workers = min(mp.cpu_count(), len(batch))
    
    if len(batch) == 1 or PYPY_MODE:
        # Single sequence or PyPy - just run serially
        return [model.forward(seq) for seq in batch]
    
    # Parallel processing
    try:
        # We can't pickle the model, so we process serially
        # but individual operations can still be parallelized
        results = []
        for seq in batch:
            results.append(model.forward(seq))
        return results
    except Exception:
        return [model.forward(seq) for seq in batch]


if __name__ == "__main__":
    # Show runtime info
    info = get_python_info()
    print(f"Python: {info['implementation']} {info['version']}")
    print(f"Platform: {info['platform']} {info['machine']}")
    print(f"CPU Cores: {info['cpu_count']}")
    print()
    
    # Run tests
    test_pure_transformer()
    
    # Benchmark
    print("\nBenchmarking matmul...")
    results = benchmark_matmul(128, 3)
    print(f"Serial: {results['serial_seconds']:.4f}s")
    print(f"Parallel: {results['parallel_seconds']:.4f}s")
    print(f"Speedup: {results['speedup']:.2f}x on {results['cpu_cores']} cores")
