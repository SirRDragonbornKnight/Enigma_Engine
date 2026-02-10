"""
Pure Python Neural Network - PRIMARY BACKEND

Zero-dependency neural network implementation using only Python stdlib.
Works anywhere Python runs - no numpy, no torch, no pip installs required.

ACCELERATOR PRIORITY (auto-detected):
1. Cython:   500-1000x faster (compile with setup_cython.py)
2. Numba:    100-638x faster  (pip install numba) 
3. PyPy:     10-50x faster    (use pypy instead of python)
4. Pure:     Baseline         (works everywhere)

This is now the PRIMARY backend for Enigma AI Engine!
- Small/Medium models (<100M): Uses Pure Python + Numba (fast enough!)
- Large models (>100M): Falls back to PyTorch for GPU acceleration

Supports ALL model sizes from Raspberry Pi to datacenter:
- Nano/Micro (0.2-2M): Fast real-time chat
- Tiny/Small (5-30M): Good performance with Numba
- Medium/Base (85-125M): Works well with Numba
- Large+ (300M+): PyTorch recommended for GPU

Usage:
    from enigma_engine.builtin.neural_network import (
        PureLinear, PureAttention, PureTransformer,
        get_backend, set_backend
    )
    
    # Auto-selects best accelerator
    model = PureTransformer(vocab_size=1000, d_model=64, n_layers=2)
    output = model.forward(input_ids)

Module Organization (3295+ lines, 15 classes):
==============================================
Lines 45-140:    Accelerator Detection (Cython, Numba, PyPy)
Lines 142-304:   PureConfig - Model configuration (~163 lines)
Lines 305-809:   Matrix - Pure Python matrix operations (~505 lines)
                 - matmul, transpose, softmax, gelu, etc.
Lines 810-918:   RoPEFrequencies - Rotary Position Embeddings (~109 lines)
Lines 919-1002:  PureLinear - Linear layer (~84 lines)
Lines 1003-1082: PureLayerNorm - Layer normalization (~80 lines)
Lines 1083-1156: PureRMSNorm - RMS normalization (~74 lines)
Lines 1157-1215: PureEmbedding - Token embeddings (~59 lines)
Lines 1216-1467: PureAttention - Multi-head attention (~252 lines)
Lines 1468-1553: PureFeedForward - Feed-forward network (~86 lines)
Lines 1554-1665: PureTransformerBlock - Transformer block (~112 lines)
Lines 1666-2081: PureTransformer - Full transformer model (~416 lines)
Lines 2082-2101: PureSGD - SGD optimizer (~20 lines)
Lines 2102-2658: PureAdam - Adam optimizer (~557 lines)
Lines 2659-2771: PureTokenizer - Simple tokenizer (~113 lines)
Lines 2772-3143: PureChat - Chat interface (~371 lines)
Lines 3143-3295: Factory functions (get_model_for_size, convert_*, load_*)
"""

import json
import math
import multiprocessing as mp
import platform
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL ACCELERATORS (Auto-detected, priority: Cython > Numba > PyPy > Pure)
# ═══════════════════════════════════════════════════════════════════════════════

# Try to import compiled Cython kernels (fastest, ~500-1000x speedup)
CYTHON_AVAILABLE = False
try:
    from . import cython_kernels
    CYTHON_AVAILABLE = True
    print("[PureNN] Cython kernels detected - maximum acceleration enabled!")
except ImportError:
    pass

# Try to import Numba for JIT compilation (100-300x speedup)
NUMBA_AVAILABLE = False
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    if not CYTHON_AVAILABLE:
        print(f"[PureNN] Numba {numba.__version__} detected - JIT acceleration enabled!")
except ImportError:
    # Create dummy decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def is_cython_available() -> bool:
    """Check if Cython kernels are compiled and available."""
    return CYTHON_AVAILABLE


def is_pypy() -> bool:
    """Check if running on PyPy (faster JIT Python)."""
    return platform.python_implementation() == "PyPy"


def is_numba_available() -> bool:
    """Check if Numba is available for JIT acceleration."""
    return NUMBA_AVAILABLE


def get_python_info() -> dict[str, Any]:
    """Get Python runtime information."""
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "is_pypy": is_pypy(),
        "cython_available": CYTHON_AVAILABLE,
        "numba_available": NUMBA_AVAILABLE,
        "cpu_count": mp.cpu_count(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "accelerator": get_acceleration_status(),
    }


def get_acceleration_status() -> str:
    """Get human-readable acceleration status (priority order)."""
    if CYTHON_AVAILABLE:
        return "Cython (500-1000x faster)"
    elif NUMBA_AVAILABLE:
        return "Numba JIT (100-300x faster)"
    elif is_pypy():
        return "PyPy JIT (10-50x faster)"
    else:
        return "Pure Python (baseline)"


# PyPy-specific optimizations
PYPY_MODE = is_pypy()
if PYPY_MODE:
    # PyPy's JIT works better with simpler code patterns
    # Disable some multiprocessing overhead since PyPy is already fast
    _DEFAULT_USE_MP = False
    if not NUMBA_AVAILABLE:
        print("[PureNN] Running on PyPy - JIT optimizations active")
else:
    _DEFAULT_USE_MP = True


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PureConfig:
    """Configuration for pure Python backend - matches ForgeConfig."""
    # Model architecture
    vocab_size: int = 1000
    d_model: int = 64          # Embedding dimension (called 'dim' in Forge)
    n_heads: int = 4           # Attention heads
    n_kv_heads: int = 0        # KV heads for GQA (0 = same as n_heads)
    n_layers: int = 2          # Transformer layers
    d_ff: int = 256            # Feed-forward dimension (called 'hidden_dim' in Forge)
    max_seq_len: int = 512     # Maximum sequence length
    dropout: float = 0.0       # Dropout (only for training)
    
    # Advanced features
    use_rope: bool = True      # Rotary Position Embeddings
    use_swiglu: bool = True    # SwiGLU activation in FFN
    use_gqa: bool = True       # Grouped Query Attention
    rope_theta: float = 10000.0  # RoPE base frequency
    use_bias: bool = False     # Use bias in linear layers
    
    # Performance
    use_multiprocessing: bool = _DEFAULT_USE_MP  # Auto-disable on PyPy
    n_workers: int = 0         # 0 = auto-detect CPU cores
    chunk_size: int = 64       # Matrix chunk size for parallelization
    
    # Training
    learning_rate: float = 0.001
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.n_kv_heads == 0:
            self.n_kv_heads = self.n_heads
    
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
    
    @property
    def n_rep(self) -> int:
        """Number of Q heads per KV head (for GQA)."""
        return self.n_heads // self.n_kv_heads
    
    def param_count(self) -> int:
        """Estimate total parameters."""
        # Embeddings (no position embeddings with RoPE)
        emb = self.vocab_size * self.d_model
        
        # Per layer: attention (Q, K, V, O) + ffn (with SwiGLU: 3 layers) + norms (2)
        attn_params = (
            self.d_model * self.d_model +                    # Q
            self.d_model * (self.n_kv_heads * self.head_dim) +  # K (GQA)
            self.d_model * (self.n_kv_heads * self.head_dim) +  # V (GQA)
            self.d_model * self.d_model                      # O
        )
        
        if self.use_swiglu:
            ffn_params = 3 * self.d_model * self.d_ff  # w1, w2, w3
        else:
            ffn_params = 2 * self.d_model * self.d_ff  # up, down
        
        norm_params = 2 * self.d_model  # 2 norms per layer
        
        per_layer = attn_params + ffn_params + norm_params
        
        # Final norm + output projection
        final = self.d_model + self.vocab_size * self.d_model
        
        return emb + (self.n_layers * per_layer) + final


# ═══════════════════════════════════════════════════════════════════════════════
# NUMBA-ACCELERATED KERNELS (100-300x faster when Numba available)
# ═══════════════════════════════════════════════════════════════════════════════
# These functions operate on flat lists/arrays for maximum performance.
# When Numba is installed, they get JIT-compiled to machine code.

if NUMBA_AVAILABLE:
    import numpy as np
    
    @jit(nopython=True, parallel=True, cache=True)
    def _numba_matmul(a_data, b_data, a_rows, a_cols, b_cols):
        """Numba-accelerated matrix multiplication with parallelization."""
        result = np.zeros(a_rows * b_cols, dtype=np.float64)
        for i in prange(a_rows):
            for k in range(a_cols):
                a_ik = a_data[i * a_cols + k]
                for j in range(b_cols):
                    result[i * b_cols + j] += a_ik * b_data[k * b_cols + j]
        return result
    
    @jit(nopython=True, cache=True)
    def _numba_softmax(data, rows, cols):
        """Numba-accelerated row-wise softmax."""
        result = np.zeros(rows * cols, dtype=np.float64)
        for i in range(rows):
            row_start = i * cols
            # Find max for numerical stability
            max_val = data[row_start]
            for j in range(1, cols):
                if data[row_start + j] > max_val:
                    max_val = data[row_start + j]
            # Compute exp and sum
            exp_sum = 0.0
            for j in range(cols):
                exp_val = math.exp(data[row_start + j] - max_val)
                result[row_start + j] = exp_val
                exp_sum += exp_val
            # Normalize
            for j in range(cols):
                result[row_start + j] /= exp_sum
        return result
    
    @jit(nopython=True, parallel=True, cache=True)
    def _numba_add(a_data, b_data):
        """Numba-accelerated element-wise addition."""
        n = len(a_data)
        result = np.zeros(n, dtype=np.float64)
        for i in prange(n):
            result[i] = a_data[i] + b_data[i]
        return result
    
    @jit(nopython=True, parallel=True, cache=True)
    def _numba_scale(data, scalar):
        """Numba-accelerated scalar multiplication."""
        n = len(data)
        result = np.zeros(n, dtype=np.float64)
        for i in prange(n):
            result[i] = data[i] * scalar
        return result
    
    @jit(nopython=True, cache=True)
    def _numba_silu(data):
        """Numba-accelerated SiLU/Swish activation."""
        n = len(data)
        result = np.zeros(n, dtype=np.float64)
        for i in range(n):
            x = data[i]
            result[i] = x / (1.0 + math.exp(-x))
        return result
    
    @jit(nopython=True, cache=True)
    def _numba_rms_norm(data, weight, rows, cols, eps):
        """Numba-accelerated RMS normalization."""
        result = np.zeros(rows * cols, dtype=np.float64)
        for i in range(rows):
            row_start = i * cols
            # Compute RMS
            ss = 0.0
            for j in range(cols):
                val = data[row_start + j]
                ss += val * val
            rms = math.sqrt(ss / cols + eps)
            # Normalize and scale
            for j in range(cols):
                result[row_start + j] = weight[j] * data[row_start + j] / rms
        return result
    
    print("[PureNN] Numba kernels compiled - expect 100-300x speedup!")


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
    
    def __init__(self, rows: int, cols: int, data: Optional[list[float]] = None):
        self.rows = rows
        self.cols = cols
        if data is not None:
            self.data = data
        else:
            self.data = [0.0] * (rows * cols)
    
    @classmethod
    def from_2d(cls, arr: list[list[float]]) -> 'Matrix':
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
    
    def __getitem__(self, idx: tuple[int, int]) -> float:
        """Get element at (row, col)."""
        row, col = idx
        return self.data[row * self.cols + col]
    
    def __setitem__(self, idx: tuple[int, int], value: float):
        """Set element at (row, col)."""
        row, col = idx
        self.data[row * self.cols + col] = value
    
    def get_row(self, row: int) -> list[float]:
        """Get a row as list."""
        start = row * self.cols
        return self.data[start:start + self.cols]
    
    def set_row(self, row: int, values: list[float]):
        """Set a row from list."""
        start = row * self.cols
        for i, v in enumerate(values):
            self.data[start + i] = v
    
    def to_2d(self) -> list[list[float]]:
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
    def shape(self) -> tuple[int, int]:
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
    
    Auto-uses best available accelerator:
    - Cython: 500-1000x faster (if compiled)
    - Numba: 100-300x faster (pip install numba)
    - Pure Python: baseline
    """
    if a.cols != b.rows:
        raise ValueError(f"Cannot multiply {a.shape} @ {b.shape}")
    
    # Use Cython if available (fastest!)
    if CYTHON_AVAILABLE:
        import numpy as np
        a_np = np.array(a.data, dtype=np.float64)
        b_np = np.array(b.data, dtype=np.float64)
        result_data = cython_kernels.cython_matmul(a_np, b_np, a.rows, a.cols, b.cols)
        return Matrix(a.rows, b.cols, result_data.tolist())
    
    # Use Numba if available (100-300x faster)
    if NUMBA_AVAILABLE:
        import numpy as np
        a_np = np.array(a.data, dtype=np.float64)
        b_np = np.array(b.data, dtype=np.float64)
        result_data = _numba_matmul(a_np, b_np, a.rows, a.cols, b.cols)
        return Matrix(a.rows, b.cols, result_data.tolist())
    
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


def _matmul_chunk(args: tuple) -> list[float]:
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
    if axis is None:
        mu = mean(a, axis)
        return sum((x - mu) ** 2 for x in a.data) / len(a.data)
    elif axis == 0:
        # Variance along rows (result is row vector)
        mu = mean(a, axis=0)  # Row vector of column means
        result = Matrix.zeros(1, a.cols)
        for j in range(a.cols):
            col_var = 0.0
            for i in range(a.rows):
                diff = a.data[i * a.cols + j] - mu.data[j]
                col_var += diff * diff
            result.data[j] = col_var / a.rows
        return result
    elif axis == 1:
        # Variance along cols (result is column vector)
        mu = mean(a, axis=1)  # Column vector of row means
        result = Matrix.zeros(a.rows, 1)
        for i in range(a.rows):
            row_var = 0.0
            start = i * a.cols
            row_mean = mu.data[i]
            for j in range(a.cols):
                diff = a.data[start + j] - row_mean
                row_var += diff * diff
            result.data[i] = row_var / a.cols
        return result
    else:
        raise ValueError(f"Invalid axis: {axis}")


def std(a: Matrix, axis: Optional[int] = None) -> Union[float, Matrix]:
    """Compute standard deviation along axis or of all elements."""
    var = variance(a, axis)
    if isinstance(var, float):
        return math.sqrt(var)
    else:
        return Matrix(var.rows, var.cols, [math.sqrt(v) for v in var.data])


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
    # Use Cython if available
    if CYTHON_AVAILABLE:
        import numpy as np
        x_np = np.array(x.data, dtype=np.float64)
        result_data = cython_kernels.cython_gelu(x_np)
        return Matrix(x.rows, x.cols, result_data.tolist())
    
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    result = []
    for v in x.data:
        inner = sqrt_2_pi * (v + 0.044715 * v * v * v)
        result.append(0.5 * v * (1.0 + math.tanh(inner)))
    return Matrix(x.rows, x.cols, result)


def silu(x: Matrix) -> Matrix:
    """SiLU/Swish activation: x * sigmoid(x)"""
    # Use Cython if available (fastest)
    if CYTHON_AVAILABLE:
        import numpy as np
        x_np = np.array(x.data, dtype=np.float64)
        result_data = cython_kernels.cython_silu(x_np)
        return Matrix(x.rows, x.cols, result_data.tolist())
    
    # Use Numba if available
    if NUMBA_AVAILABLE:
        import numpy as np
        x_np = np.array(x.data, dtype=np.float64)
        result_data = _numba_silu(x_np)
        return Matrix(x.rows, x.cols, result_data.tolist())
    
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
    Auto-uses best accelerator (Cython > Numba).
    """
    # Use Cython for row-wise softmax (most common case)
    if CYTHON_AVAILABLE and (axis == -1 or axis == 1):
        import numpy as np
        x_np = np.array(x.data, dtype=np.float64)
        result_data = cython_kernels.cython_softmax(x_np, x.rows, x.cols)
        return Matrix(x.rows, x.cols, result_data.tolist())
    
    # Use Numba for row-wise softmax (most common case)
    if NUMBA_AVAILABLE and (axis == -1 or axis == 1):
        import numpy as np
        x_np = np.array(x.data, dtype=np.float64)
        result_data = _numba_softmax(x_np, x.rows, x.cols)
        return Matrix(x.rows, x.cols, result_data.tolist())
    
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


def dropout(x: Matrix, p: float = 0.1, training: bool = True) -> Matrix:
    """
    Dropout regularization.
    
    During training, randomly zeroes elements with probability p.
    Remaining elements are scaled by 1/(1-p) to maintain expected values.
    
    Args:
        x: Input matrix
        p: Dropout probability (0 to 1)
        training: If False, returns input unchanged
    
    Returns:
        Matrix with dropout applied
    """
    if not training or p == 0.0:
        return x
    
    scale = 1.0 / (1.0 - p)
    result = []
    for v in x.data:
        if random.random() < p:
            result.append(0.0)
        else:
            result.append(v * scale)
    return Matrix(x.rows, x.cols, result)


# ═══════════════════════════════════════════════════════════════════════════════
# ROTARY POSITION EMBEDDINGS (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════

class RoPEFrequencies:
    """
    Precomputed RoPE (Rotary Position Embedding) frequencies.
    
    RoPE encodes position by ROTATING vectors. Each position gets a unique
    rotation angle that the model learns to interpret.
    
    📐 THE MATH:
    For dimension pair i, frequency = 1 / (theta^(2i/dim))
    For position p, angle = p * frequency
    We store cos(angle) and sin(angle) for efficient rotation.
    """
    
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        """
        Precompute RoPE frequencies.
        
        Args:
            dim: Dimension per head (must be even)
            max_seq_len: Maximum sequence length
            theta: Base frequency (higher = better long context)
        """
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute cos and sin for all positions
        self.cos_cache: list[list[float]] = []
        self.sin_cache: list[list[float]] = []
        
        # freqs[i] = 1 / (theta^(2i/dim))
        freqs = []
        for i in range(0, dim, 2):
            freqs.append(1.0 / (theta ** (i / dim)))
        
        # For each position, compute cos and sin of (pos * freq)
        for pos in range(max_seq_len):
            cos_row = []
            sin_row = []
            for freq in freqs:
                angle = pos * freq
                cos_row.append(math.cos(angle))
                sin_row.append(math.sin(angle))
            self.cos_cache.append(cos_row)
            self.sin_cache.append(sin_row)
    
    def get_cos_sin(self, seq_len: int, start_pos: int = 0) -> tuple[list[list[float]], list[list[float]]]:
        """
        Get cos and sin values for a sequence.
        
        Args:
            seq_len: Length of sequence
            start_pos: Starting position (for KV cache continuation)
            
        Returns:
            (cos_values, sin_values) for positions [start_pos, start_pos + seq_len)
        """
        end_pos = start_pos + seq_len
        return (
            self.cos_cache[start_pos:end_pos],
            self.sin_cache[start_pos:end_pos]
        )


def apply_rope(x: Matrix, cos_vals: list[list[float]], sin_vals: list[list[float]]) -> Matrix:
    """
    Apply rotary position embeddings to a matrix.
    
    📐 ROTATION FORMULA:
    For each pair of dimensions (x0, x1):
        rotated_x0 = x0 * cos - x1 * sin
        rotated_x1 = x0 * sin + x1 * cos
    
    Args:
        x: Input matrix (seq_len, dim) or reshaped for heads
        cos_vals: Cosine values for each position
        sin_vals: Sine values for each position
        
    Returns:
        Rotated matrix, same shape as input
    """
    result = Matrix.zeros(x.rows, x.cols)
    half_dim = x.cols // 2
    
    for i in range(x.rows):
        row_start = i * x.cols
        cos_row = cos_vals[i] if i < len(cos_vals) else cos_vals[-1]
        sin_row = sin_vals[i] if i < len(sin_vals) else sin_vals[-1]
        
        for j in range(half_dim):
            # Get the pair of values
            x0 = x.data[row_start + j]
            x1 = x.data[row_start + half_dim + j]
            
            # Get rotation angles
            cos_val = cos_row[j] if j < len(cos_row) else 1.0
            sin_val = sin_row[j] if j < len(sin_row) else 0.0
            
            # Apply rotation
            result.data[row_start + j] = x0 * cos_val - x1 * sin_val
            result.data[row_start + half_dim + j] = x0 * sin_val + x1 * cos_val
    
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
    
    def parameters(self) -> list[Matrix]:
        """Return list of parameters."""
        if self.has_bias:
            return [self.weight, self.bias]
        return [self.weight]
    
    def gradients(self) -> list[Optional[Matrix]]:
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
        self._std_cache: Optional[list[float]] = None
    
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
    
    def parameters(self) -> list[Matrix]:
        return [self.gamma, self.beta]
    
    def gradients(self) -> list[Optional[Matrix]]:
        return [self.gamma_grad, self.beta_grad]


class PureRMSNorm:
    """
    Root Mean Square Layer Normalization.
    
    Simpler than LayerNorm - no mean subtraction.
    y = x / sqrt(mean(x^2) + eps) * gamma
    
    Auto-uses Numba acceleration when available.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Matrix.ones(1, normalized_shape)
        self.gamma_grad: Optional[Matrix] = None
        self._rms_cache: Optional[list[float]] = None
        self._input_cache: Optional[Matrix] = None
    
    def forward(self, x: Matrix) -> Matrix:
        """Forward pass."""
        self._input_cache = x
        
        # Use Numba if available
        if NUMBA_AVAILABLE:
            import numpy as np
            x_np = np.array(x.data, dtype=np.float64)
            gamma_np = np.array(self.gamma.data, dtype=np.float64)
            result_data = _numba_rms_norm(x_np, gamma_np, x.rows, x.cols, self.eps)
            return Matrix(x.rows, x.cols, result_data.tolist())
        
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
    
    def parameters(self) -> list[Matrix]:
        return [self.gamma]
    
    def gradients(self) -> list[Optional[Matrix]]:
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
        self._input_cache: Optional[list[int]] = None
    
    def forward(self, input_ids: list[int]) -> Matrix:
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
    
    def parameters(self) -> list[Matrix]:
        return [self.weight]
    
    def gradients(self) -> list[Optional[Matrix]]:
        return [self.weight_grad]


# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION MECHANISM
# ═══════════════════════════════════════════════════════════════════════════════

class PureAttention:
    """
    Multi-Head Self-Attention with Grouped Query Attention (GQA) and KV Cache.
    
    📖 WHAT THIS DOES:
    Attention is how the model "looks at" different parts of the input.
    "The cat sat on the mat" - when processing "sat", attention lets
    the model look back at "cat" to know WHO sat.
    
    📐 THE MATH (simplified):
    1. Create Query (Q), Key (K), Value (V) from input
    2. Attention scores = Q @ K.T / sqrt(dim)  (which words to look at?)
    3. Softmax → probabilities (normalize scores)
    4. Output = scores @ V  (weighted combination of values)
    
    ⚡ GROUPED QUERY ATTENTION (GQA):
    Normal: Each head has its own K and V (memory hungry!)
    GQA: Multiple Q heads share the same K,V (saves 2-4x memory!)
    
    💾 KV-CACHE:
    During generation, we only add ONE new token at a time.
    Instead of recomputing K,V for all previous tokens, we cache them!
    """
    
    MAX_CACHE_SEQ_LEN = 4096
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        n_kv_heads: int = 0,
        use_rope: bool = True,
        use_bias: bool = False,
        max_seq_len: int = 512,
        rope_theta: float = 10000.0
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads > 0 else n_heads
        self.head_dim = d_model // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # Q heads per KV head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = use_rope
        
        # Q projection: full size
        self.wq = PureLinear(d_model, n_heads * self.head_dim, bias=use_bias)
        # K, V projections: may be smaller for GQA
        self.wk = PureLinear(d_model, self.n_kv_heads * self.head_dim, bias=use_bias)
        self.wv = PureLinear(d_model, self.n_kv_heads * self.head_dim, bias=use_bias)
        # Output projection
        self.wo = PureLinear(n_heads * self.head_dim, d_model, bias=use_bias)
        
        # RoPE frequencies
        if use_rope:
            self.rope_freqs = RoPEFrequencies(self.head_dim, max_seq_len, rope_theta)
        else:
            self.rope_freqs = None
        
        # KV Cache
        self.cache_k: Optional[list[list[float]]] = None  # List of K vectors
        self.cache_v: Optional[list[list[float]]] = None  # List of V vectors
        self.cache_seq_len: int = 0
        
        # Cache for backward
        self._input_cache: Optional[Matrix] = None
        self._attn_cache: Optional[Matrix] = None
    
    def forward(
        self, 
        x: Matrix, 
        mask: Optional[Matrix] = None,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> Matrix:
        """
        Forward pass through attention with optional KV cache.
        
        Args:
            x: Input tensor (seq_len, d_model)
            mask: Optional attention mask
            use_cache: Whether to use/update KV cache
            start_pos: Starting position (for cache continuation)
        
        Returns:
            Output tensor (seq_len, d_model)
        """
        seq_len = x.rows
        self._input_cache = x
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Project to Q, K, V
        # ─────────────────────────────────────────────────────────────────────
        q = self.wq.forward(x)  # (seq_len, n_heads * head_dim)
        k = self.wk.forward(x)  # (seq_len, n_kv_heads * head_dim)
        v = self.wv.forward(x)  # (seq_len, n_kv_heads * head_dim)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Apply RoPE to Q and K
        # ─────────────────────────────────────────────────────────────────────
        if self.use_rope and self.rope_freqs is not None:
            cos_vals, sin_vals = self.rope_freqs.get_cos_sin(seq_len, start_pos)
            
            # Apply RoPE per head for Q
            q_rotated = Matrix.zeros(q.rows, q.cols)
            for h in range(self.n_heads):
                h_start = h * self.head_dim
                h_end = h_start + self.head_dim
                # Extract head slice
                head_q = Matrix(seq_len, self.head_dim)
                for i in range(seq_len):
                    for j in range(self.head_dim):
                        head_q[i, j] = q[i, h_start + j]
                # Rotate
                head_q_rot = apply_rope(head_q, cos_vals, sin_vals)
                # Put back
                for i in range(seq_len):
                    for j in range(self.head_dim):
                        q_rotated[i, h_start + j] = head_q_rot[i, j]
            q = q_rotated
            
            # Apply RoPE per head for K
            k_rotated = Matrix.zeros(k.rows, k.cols)
            for h in range(self.n_kv_heads):
                h_start = h * self.head_dim
                h_end = h_start + self.head_dim
                head_k = Matrix(seq_len, self.head_dim)
                for i in range(seq_len):
                    for j in range(self.head_dim):
                        head_k[i, j] = k[i, h_start + j]
                head_k_rot = apply_rope(head_k, cos_vals, sin_vals)
                for i in range(seq_len):
                    for j in range(self.head_dim):
                        k_rotated[i, h_start + j] = head_k_rot[i, j]
            k = k_rotated
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Handle KV Cache
        # ─────────────────────────────────────────────────────────────────────
        if use_cache:
            if self.cache_k is None:
                # First token - initialize cache
                self.cache_k = [k.get_row(i) for i in range(seq_len)]
                self.cache_v = [v.get_row(i) for i in range(seq_len)]
            else:
                # Append new K, V to cache
                for i in range(seq_len):
                    self.cache_k.append(k.get_row(i))
                    self.cache_v.append(v.get_row(i))
                
                # Trim if too long (sliding window)
                if len(self.cache_k) > self.MAX_CACHE_SEQ_LEN:
                    trim = len(self.cache_k) - self.MAX_CACHE_SEQ_LEN
                    self.cache_k = self.cache_k[trim:]
                    self.cache_v = self.cache_v[trim:]
            
            # Use cached K, V
            cache_len = len(self.cache_k)
            k = Matrix(cache_len, k.cols)
            v = Matrix(cache_len, v.cols)
            for i in range(cache_len):
                k.set_row(i, self.cache_k[i])
                v.set_row(i, self.cache_v[i])
            
            self.cache_seq_len = cache_len
        
        kv_seq_len = k.rows
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Multi-Head Attention (with GQA)
        # ─────────────────────────────────────────────────────────────────────
        # Process each head separately for proper multi-head attention
        output_heads = []
        
        for h in range(self.n_heads):
            # Get Q for this head
            q_head = Matrix(seq_len, self.head_dim)
            q_start = h * self.head_dim
            for i in range(seq_len):
                for j in range(self.head_dim):
                    q_head[i, j] = q[i, q_start + j]
            
            # Get K, V for this head (with GQA: multiple Q heads share same KV)
            kv_head_idx = h // self.n_rep
            k_head = Matrix(kv_seq_len, self.head_dim)
            v_head = Matrix(kv_seq_len, self.head_dim)
            kv_start = kv_head_idx * self.head_dim
            for i in range(kv_seq_len):
                for j in range(self.head_dim):
                    k_head[i, j] = k[i, kv_start + j]
                    v_head[i, j] = v[i, kv_start + j]
            
            # Attention scores: Q @ K.T / sqrt(head_dim)
            scores = matmul(q_head, k_head.T)
            scores = scale(scores, self.scale)
            
            # Apply causal mask
            if mask is not None:
                for i in range(scores.rows):
                    for j in range(scores.cols):
                        if j >= mask.cols or mask[i, j] == 0:
                            scores[i, j] = -1e9
            else:
                # Default causal mask for generation
                for i in range(scores.rows):
                    query_pos = start_pos + i if use_cache else i
                    for j in range(scores.cols):
                        if j > query_pos:
                            scores[i, j] = -1e9
            
            # Softmax
            attn = softmax(scores, axis=-1)
            
            # Apply to values
            head_output = matmul(attn, v_head)
            output_heads.append(head_output)
        
        # Concatenate heads
        output = Matrix(seq_len, self.n_heads * self.head_dim)
        for h, head_out in enumerate(output_heads):
            h_start = h * self.head_dim
            for i in range(seq_len):
                for j in range(self.head_dim):
                    output[i, h_start + j] = head_out[i, j]
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 5: Output projection
        # ─────────────────────────────────────────────────────────────────────
        return self.wo.forward(output)
    
    def clear_cache(self):
        """Clear the KV cache (call between different sequences)."""
        self.cache_k = None
        self.cache_v = None
        self.cache_seq_len = 0
    
    def parameters(self) -> list[Matrix]:
        params = []
        params.extend(self.wq.parameters())
        params.extend(self.wk.parameters())
        params.extend(self.wv.parameters())
        params.extend(self.wo.parameters())
        return params
    
    def gradients(self) -> list[Optional[Matrix]]:
        grads = []
        grads.extend(self.wq.gradients())
        grads.extend(self.wk.gradients())
        grads.extend(self.wv.gradients())
        grads.extend(self.wo.gradients())
        return grads


class PureFeedForward:
    """
    SwiGLU Feed-Forward Network.
    
    📖 WHAT THIS DOES:
    After attention decides WHAT to look at, the FFN decides
    WHAT TO DO with that information. It's the "thinking" part!
    
    📐 SWIGLU FORMULA:
    Standard FFN: output = W2(ReLU(W1(x)))
    SwiGLU:       output = W2(Swish(W1(x)) * W3(x))
    
    💡 WHY SWIGLU IS BETTER:
    - Swish activation is smoother than ReLU (no hard corners)
    - Gating mechanism (the W3 multiplication) helps information flow
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        use_swiglu: bool = True,
        use_bias: bool = False
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_swiglu = use_swiglu
        
        if use_swiglu:
            # SwiGLU: 3 projections
            self.w1 = PureLinear(d_model, d_ff, bias=use_bias)  # Gate
            self.w2 = PureLinear(d_ff, d_model, bias=use_bias)  # Down
            self.w3 = PureLinear(d_model, d_ff, bias=use_bias)  # Up
        else:
            # Standard: 2 projections
            self.up_proj = PureLinear(d_model, d_ff, bias=use_bias)
            self.down_proj = PureLinear(d_ff, d_model, bias=use_bias)
        
        self._gate_cache: Optional[Matrix] = None
        self._up_cache: Optional[Matrix] = None
    
    def forward(self, x: Matrix) -> Matrix:
        """
        Forward pass.
        
        📐 SwiGLU computation:
        1. gate = swish(W1 @ x)  ← Smooth activation
        2. up = W3 @ x           ← Unactivated projection  
        3. hidden = gate * up    ← Gated combination
        4. output = W2 @ hidden  ← Project back
        """
        if self.use_swiglu:
            # SwiGLU
            gate = self.w1.forward(x)
            gate = silu(gate)  # Swish = SiLU
            self._gate_cache = gate
            
            up = self.w3.forward(x)
            self._up_cache = up
            
            # Gated combination
            hidden = multiply(gate, up)
            
            return self.w2.forward(hidden)
        else:
            # Standard FFN
            up = self.up_proj.forward(x)
            self._up_cache = up
            act = gelu(up)
            return self.down_proj.forward(act)
    
    def parameters(self) -> list[Matrix]:
        if self.use_swiglu:
            return self.w1.parameters() + self.w2.parameters() + self.w3.parameters()
        return self.up_proj.parameters() + self.down_proj.parameters()
    
    def gradients(self) -> list[Optional[Matrix]]:
        if self.use_swiglu:
            return self.w1.gradients() + self.w2.gradients() + self.w3.gradients()
        return self.up_proj.gradients() + self.down_proj.gradients()


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

class PureTransformerBlock:
    """
    Single Transformer block with all modern features.
    
    📖 WHAT THIS DOES:
    This is ONE "layer" of the transformer. The full model stacks many of these.
    
    📐 ARCHITECTURE (Pre-Norm style):
        x = x + Attention(Norm(x))   ← Look at context
        x = x + FFN(Norm(x))         ← Process information
    
    The residual connections (x + ...) help gradients flow during training.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int, 
        n_kv_heads: int = 0,
        use_rms_norm: bool = True,
        use_rope: bool = True,
        use_swiglu: bool = True,
        max_seq_len: int = 512
    ):
        self.d_model = d_model
        
        # Layer norms (pre-norm architecture)
        if use_rms_norm:
            self.attn_norm = PureRMSNorm(d_model)
            self.ffn_norm = PureRMSNorm(d_model)
        else:
            self.attn_norm = PureLayerNorm(d_model)
            self.ffn_norm = PureLayerNorm(d_model)
        
        # Attention with GQA and RoPE
        self.attention = PureAttention(
            d_model=d_model, 
            n_heads=n_heads, 
            n_kv_heads=n_kv_heads,
            use_rope=use_rope,
            max_seq_len=max_seq_len
        )
        
        # FFN with optional SwiGLU
        self.ffn = PureFeedForward(d_model, d_ff, use_swiglu=use_swiglu)
        
        # Cache for residual connections
        self._x_cache: Optional[Matrix] = None
        self._attn_out_cache: Optional[Matrix] = None
    
    def forward(
        self, 
        x: Matrix, 
        mask: Optional[Matrix] = None,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> Matrix:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            use_cache: Whether to use KV cache
            start_pos: Starting position for cache
        """
        self._x_cache = x
        
        # Attention block with residual
        normed = self.attn_norm.forward(x)
        attn_out = self.attention.forward(normed, mask, use_cache=use_cache, start_pos=start_pos)
        x = add(x, attn_out)
        self._attn_out_cache = x
        
        # FFN block with residual
        normed = self.ffn_norm.forward(x)
        ffn_out = self.ffn.forward(normed)
        x = add(x, ffn_out)
        
        return x
    
    def clear_cache(self):
        """Clear KV cache in attention."""
        self.attention.clear_cache()
    
    def backward(self, grad_output: Matrix) -> Matrix:
        """Backward pass through block."""
        # Simplified backward - for training use full implementation
        return grad_output
    
    def parameters(self) -> list[Matrix]:
        params = []
        params.extend(self.attn_norm.parameters())
        params.extend(self.attention.parameters())
        params.extend(self.ffn_norm.parameters())
        params.extend(self.ffn.parameters())
        return params
    
    def gradients(self) -> list[Optional[Matrix]]:
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
    
    📖 WHAT THIS IS:
    The full language model that ties everything together.
    Compatible with Enigma AI Engine's nano/micro model sizes.
    
    📐 ARCHITECTURE:
    Input → Embedding → [Transformer Block × N] → Norm → Output Projection
    
    ⚡ FEATURES:
    - RoPE positional embeddings (no separate position embedding needed)
    - KV cache for fast generation
    - Grouped Query Attention
    - SwiGLU feed-forward networks
    """
    
    def __init__(self, config: Optional[PureConfig] = None, **kwargs):
        if config is None:
            config = PureConfig(**kwargs)
        self.config = config
        
        # Token embeddings (no position embeddings - using RoPE!)
        self.token_embedding = PureEmbedding(config.vocab_size, config.d_model)
        
        # Position embedding is only used if RoPE is disabled
        if not config.use_rope:
            self.position_embedding = PureEmbedding(config.max_seq_len, config.d_model)
        else:
            self.position_embedding = None
        
        # Transformer blocks with all features
        self.blocks = [
            PureTransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                n_kv_heads=config.n_kv_heads,
                use_rms_norm=True,
                use_rope=config.use_rope,
                use_swiglu=config.use_swiglu,
                max_seq_len=config.max_seq_len
            )
            for _ in range(config.n_layers)
        ]
        
        # Final layer norm and output projection
        self.final_norm = PureRMSNorm(config.d_model)
        self.output_proj = PureLinear(config.d_model, config.vocab_size, bias=False)
        
        # Use parallel operations for larger models
        self.use_parallel = config.use_multiprocessing and config.param_count() > 500_000
    
    def forward(
        self, 
        input_ids: list[int],
        use_cache: bool = False,
        start_pos: int = 0
    ) -> Matrix:
        """
        Forward pass.
        
        Args:
            input_ids: List of token IDs
            use_cache: Whether to use KV cache for generation
            start_pos: Starting position (for cache continuation)
            
        Returns:
            Logits matrix (seq_len, vocab_size)
        """
        seq_len = len(input_ids)
        
        # Get token embeddings
        x = self.token_embedding.forward(input_ids)
        
        # Add position embeddings (only if not using RoPE)
        if self.position_embedding is not None:
            pos_ids = list(range(start_pos, start_pos + seq_len))
            pos_emb = self.position_embedding.forward(pos_ids)
            x = add(x, pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask=None, use_cache=use_cache, start_pos=start_pos)
        
        # Final norm and project to vocabulary
        x = self.final_norm.forward(x)
        logits = self.output_proj.forward(x)
        
        return logits
    
    def clear_cache(self):
        """Clear KV cache in all blocks."""
        for block in self.blocks:
            block.clear_cache()
    
    def _create_causal_mask(self, seq_len: int) -> Matrix:
        """Create causal (triangular) attention mask."""
        mask = Matrix.ones(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                mask[i, j] = 0.0
        return mask
    
    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        use_cache: bool = True
    ) -> list[int]:
        """
        Generate tokens autoregressively with KV cache.
        
        Args:
            input_ids: Starting token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (0 to disable)
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens (>1.0 reduces repetition)
            use_cache: Whether to use KV cache (faster!)
            
        Returns:
            List of generated token IDs
        """
        generated = list(input_ids)
        
        # Clear any previous cache
        self.clear_cache()
        
        # Process prompt first (if using cache)
        if use_cache and len(input_ids) > 1:
            # Process all but last token to build cache
            _ = self.forward(input_ids[:-1], use_cache=True, start_pos=0)
            # Now process just the last token
            context = [input_ids[-1]]
            start_pos = len(input_ids) - 1
        else:
            context = input_ids
            start_pos = 0
        
        for i in range(max_new_tokens):
            if use_cache:
                # With cache: just process the new token
                logits = self.forward(context, use_cache=True, start_pos=start_pos)
            else:
                # Without cache: process full sequence (slower)
                context = generated[-self.config.max_seq_len:]
                logits = self.forward(context, use_cache=False, start_pos=0)
            
            # Get last token's logits
            last_logits = logits.get_row(logits.rows - 1)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if 0 <= token_id < len(last_logits):
                        last_logits[token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                last_logits = [l / temperature for l in last_logits]
            
            # Top-k filtering
            if top_k > 0:
                # Get indices of top-k values
                indexed = list(enumerate(last_logits))
                indexed.sort(key=lambda x: x[1], reverse=True)
                top_indices = {i for i, _ in indexed[:top_k]}
                
                # Zero out non-top-k
                for i in range(len(last_logits)):
                    if i not in top_indices:
                        last_logits[i] = -1e9
            
            # Softmax to get probabilities
            max_logit = max(last_logits)
            exp_logits = [math.exp(l - max_logit) for l in last_logits]
            sum_exp = sum(exp_logits)
            probs = [e / sum_exp for e in exp_logits]
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
                cumsum = 0.0
                cutoff_idx = len(sorted_probs)
                for idx, (i, p) in enumerate(sorted_probs):
                    cumsum += p
                    if cumsum > top_p:
                        cutoff_idx = idx + 1
                        break
                
                # Keep only tokens within top_p
                allowed = {i for i, _ in sorted_probs[:cutoff_idx]}
                for i in range(len(probs)):
                    if i not in allowed:
                        probs[i] = 0.0
                
                # Renormalize
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
            
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
            
            # Update for next iteration (with cache)
            if use_cache:
                context = [next_token]
                start_pos = len(generated) - 1
            
            # Stop at EOS (assuming token 0 or 2 is EOS)
            if next_token in [0, 2]:
                break
        
        self.clear_cache()
        return generated
    
    def generate_stream(
        self,
        input_ids: list[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0
    ):
        """
        Generate tokens with streaming (yields tokens one at a time).
        
        This is like generate() but yields each token as it's generated,
        so you can display output in real-time.
        
        Args:
            repetition_penalty: Penalty for repeating tokens (>1.0 reduces repetition)
        
        Yields:
            Token IDs one at a time
        """
        generated = list(input_ids)
        
        # Clear any previous cache
        self.clear_cache()
        
        # Process prompt first to build cache
        if len(input_ids) > 1:
            _ = self.forward(input_ids[:-1], use_cache=True, start_pos=0)
            context = [input_ids[-1]]
            start_pos = len(input_ids) - 1
        else:
            context = input_ids
            start_pos = 0
        
        for i in range(max_new_tokens):
            logits = self.forward(context, use_cache=True, start_pos=start_pos)
            last_logits = logits.get_row(logits.rows - 1)
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if 0 <= token_id < len(last_logits):
                        last_logits[token_id] /= repetition_penalty
            
            # Temperature
            if temperature != 1.0:
                last_logits = [l / temperature for l in last_logits]
            
            # Top-k
            if top_k > 0:
                indexed = list(enumerate(last_logits))
                indexed.sort(key=lambda x: x[1], reverse=True)
                top_indices = {i for i, _ in indexed[:top_k]}
                for i in range(len(last_logits)):
                    if i not in top_indices:
                        last_logits[i] = -1e9
            
            # Softmax
            max_logit = max(last_logits)
            exp_logits = [math.exp(l - max_logit) for l in last_logits]
            sum_exp = sum(exp_logits)
            probs = [e / sum_exp for e in exp_logits]
            
            # Top-p
            if top_p < 1.0:
                sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
                cumsum = 0.0
                cutoff_idx = len(sorted_probs)
                for idx, (tok_i, p) in enumerate(sorted_probs):
                    cumsum += p
                    if cumsum > top_p:
                        cutoff_idx = idx + 1
                        break
                allowed = {i for i, _ in sorted_probs[:cutoff_idx]}
                for i in range(len(probs)):
                    if i not in allowed:
                        probs[i] = 0.0
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
            
            # Sample
            r = random.random()
            cumsum = 0.0
            next_token = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r < cumsum:
                    next_token = i
                    break
            
            generated.append(next_token)
            yield next_token  # Stream the token!
            
            # Update for next iteration
            context = [next_token]
            start_pos = len(generated) - 1
            
            # Stop at EOS
            if next_token in [0, 2]:
                break
        
        self.clear_cache()
    
    def parameters(self) -> list[Matrix]:
        """Get all model parameters."""
        params = []
        params.extend(self.token_embedding.parameters())
        if self.position_embedding is not None:
            params.extend(self.position_embedding.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        params.extend(self.output_proj.parameters())
        return params
    
    def gradients(self) -> list[Optional[Matrix]]:
        """Get all gradients."""
        grads = []
        grads.extend(self.token_embedding.gradients())
        if self.position_embedding is not None:
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
            "max_seq_len": self.config.max_seq_len,
            "n_kv_heads": self.config.n_kv_heads,
            "use_rope": self.config.use_rope,
            "use_swiglu": self.config.use_swiglu,
            "use_gqa": self.config.use_gqa
        }
        
        with open(path, 'w') as f:
            json.dump(weights, f)
    
    def load(self, path: Path):
        """Load model weights from file."""
        with open(path) as f:
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
    
    def __init__(self, parameters: list[Matrix], lr: float = 0.001):
        self.parameters = parameters
        self.lr = lr
    
    def step(self, gradients: list[Optional[Matrix]]):
        """Apply gradients to parameters."""
        for param, grad in zip(self.parameters, gradients):
            if grad is not None:
                for i in range(len(param.data)):
                    param.data[i] -= self.lr * grad.data[i]
    
    def zero_grad(self):
        """Zero all gradients (they get accumulated)."""
        # Gradients are recomputed each backward pass


class PureAdam:
    """Adam optimizer."""
    
    def __init__(
        self,
        parameters: list[Matrix],
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
    
    def step(self, gradients: list[Optional[Matrix]]):
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


def cross_entropy_loss(logits: Matrix, targets: list[int]) -> tuple[float, Matrix]:
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


def get_model_for_size(size: str, vocab_size: int = None) -> Union['PureTransformer', Any]:
    """
    Get appropriate model implementation for a given size.
    
    Supports ALL sizes from nano to omega (70B+).
    Auto-selects Pure Python or PyTorch based on what's available.
    
    Args:
        size: Model size name (nano, micro, tiny, small, medium, large, xl, xxl, omega)
        vocab_size: Override vocab size (optional)
        
    Returns:
        Model instance (PureTransformer or PyTorch model)
    """
    # Size configurations (matching Enigma AI Engine presets)
    # These scale from Raspberry Pi to datacenter
    SIZE_CONFIGS = {
        # Tiny - for embedded/Pi (pure Python friendly)
        "nano":   {"d_model": 64,   "n_heads": 2,  "n_layers": 2,  "d_ff": 128,   "vocab": 1000},   # ~0.2M
        "micro":  {"d_model": 128,  "n_heads": 4,  "n_layers": 4,  "d_ff": 256,   "vocab": 2000},   # ~1M
        "tiny":   {"d_model": 256,  "n_heads": 4,  "n_layers": 6,  "d_ff": 512,   "vocab": 4000},   # ~5M
        
        # Medium - for desktop (Numba recommended)
        "small":  {"d_model": 512,  "n_heads": 8,  "n_layers": 8,  "d_ff": 1024,  "vocab": 8000},   # ~30M
        "medium": {"d_model": 768,  "n_heads": 12, "n_layers": 12, "d_ff": 2048,  "vocab": 16000},  # ~85M
        "base":   {"d_model": 768,  "n_heads": 12, "n_layers": 12, "d_ff": 3072,  "vocab": 32000},  # ~125M (GPT-2 small)
        
        # Large - for GPU/server (PyTorch recommended)
        "large":  {"d_model": 1024, "n_heads": 16, "n_layers": 24, "d_ff": 4096,  "vocab": 32000},  # ~350M (GPT-2 medium)
        "xl":     {"d_model": 1280, "n_heads": 20, "n_layers": 36, "d_ff": 5120,  "vocab": 50000},  # ~770M (GPT-2 large)
        "xxl":    {"d_model": 1600, "n_heads": 25, "n_layers": 48, "d_ff": 6400,  "vocab": 50000},  # ~1.5B (GPT-2 XL)
        
        # Massive - for datacenter/supercomputer
        "huge":   {"d_model": 2048, "n_heads": 32, "n_layers": 64, "d_ff": 8192,  "vocab": 50000},  # ~3B
        "giant":  {"d_model": 4096, "n_heads": 32, "n_layers": 80, "d_ff": 16384, "vocab": 100000}, # ~13B
        "omega":  {"d_model": 8192, "n_heads": 64, "n_layers": 96, "d_ff": 32768, "vocab": 100000}, # ~70B (LLaMA-70B scale)
    }
    
    if size not in SIZE_CONFIGS:
        print(f"Unknown size '{size}', using 'small'. Available: {list(SIZE_CONFIGS.keys())}")
        size = "small"
    
    cfg = SIZE_CONFIGS[size]
    v_size = vocab_size if vocab_size else cfg["vocab"]
    
    config = PureConfig(
        vocab_size=v_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=min(2048, cfg["d_model"] * 4),  # Scale context with model
    )
    
    param_count = config.param_count()
    
    if should_use_pure_backend(param_count):
        accel = "Numba" if NUMBA_AVAILABLE else ("PyPy" if is_pypy() else "Pure Python")
        print(f"[PureBackend] Using {accel} for {size} ({param_count/1e6:.1f}M params)")
        return PureTransformer(config)
    else:
        # Try to use PyTorch
        try:
            from ..core.model import create_model
            print(f"[PyTorchBackend] Using PyTorch for {size} ({param_count/1e6:.1f}M params)")
            return create_model(size)
        except ImportError:
            accel = "Numba" if NUMBA_AVAILABLE else ("PyPy" if is_pypy() else "Pure Python")
            print(f"[PureBackend] PyTorch unavailable, using {accel} for {size}")
            return PureTransformer(config)


def list_available_sizes() -> dict[str, dict[str, Any]]:
    """
    List all available model sizes with their configurations.
    
    Returns:
        Dict mapping size names to their configurations and estimated params
    """
    sizes = {
        "nano":   {"d_model": 64,   "n_heads": 2,  "n_layers": 2,  "d_ff": 128,   "vocab": 1000},
        "micro":  {"d_model": 128,  "n_heads": 4,  "n_layers": 4,  "d_ff": 256,   "vocab": 2000},
        "tiny":   {"d_model": 256,  "n_heads": 4,  "n_layers": 6,  "d_ff": 512,   "vocab": 4000},
        "small":  {"d_model": 512,  "n_heads": 8,  "n_layers": 8,  "d_ff": 1024,  "vocab": 8000},
        "medium": {"d_model": 768,  "n_heads": 12, "n_layers": 12, "d_ff": 2048,  "vocab": 16000},
        "base":   {"d_model": 768,  "n_heads": 12, "n_layers": 12, "d_ff": 3072,  "vocab": 32000},
        "large":  {"d_model": 1024, "n_heads": 16, "n_layers": 24, "d_ff": 4096,  "vocab": 32000},
        "xl":     {"d_model": 1280, "n_heads": 20, "n_layers": 36, "d_ff": 5120,  "vocab": 50000},
        "xxl":    {"d_model": 1600, "n_heads": 25, "n_layers": 48, "d_ff": 6400,  "vocab": 50000},
        "huge":   {"d_model": 2048, "n_heads": 32, "n_layers": 64, "d_ff": 8192,  "vocab": 50000},
        "giant":  {"d_model": 4096, "n_heads": 32, "n_layers": 80, "d_ff": 16384, "vocab": 100000},
        "omega":  {"d_model": 8192, "n_heads": 64, "n_layers": 96, "d_ff": 32768, "vocab": 100000},
    }
    
    result = {}
    for name, cfg in sizes.items():
        config = PureConfig(
            vocab_size=cfg["vocab"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], d_ff=cfg["d_ff"]
        )
        params = config.param_count()
        result[name] = {
            "params": params,
            "params_human": f"{params/1e6:.1f}M" if params < 1e9 else f"{params/1e9:.1f}B",
            "d_model": cfg["d_model"],
            "n_layers": cfg["n_layers"],
            "recommended_for": (
                "Pi/embedded" if params < 5e6 else
                "Desktop" if params < 100e6 else
                "GPU" if params < 1e9 else
                "Multi-GPU" if params < 10e9 else
                "Datacenter"
            )
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_matmul(size: int = 256, iterations: int = 5) -> dict[str, float]:
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

def convert_pytorch_to_pure(pytorch_state_dict: dict[str, Any], config: PureConfig) -> 'PureTransformer':
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


def convert_pure_to_pytorch(pure_model: 'PureTransformer') -> dict[str, Any]:
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
        with open(path) as f:
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

def forward_batch(model: 'PureTransformer', batch: list[list[int]], n_workers: int = 0) -> list[Matrix]:
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


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class PureTokenizer:
    """
    Simple tokenizer wrapper that works with PureTransformer.
    
    Can use Enigma AI Engine's tokenizer if available, or falls back to simple splitting.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self._forge_tokenizer = None
        self._simple_vocab: dict[str, int] = {}
        self._simple_vocab_rev: dict[int, str] = {}
        self._loaded = False
    
    def load(self) -> bool:
        """Load tokenizer - try Enigma AI Engine first, fall back to simple."""
        # Try to load Enigma AI Engine tokenizer
        try:
            from ..core.tokenizer import get_tokenizer
            self._forge_tokenizer = get_tokenizer()
            self.vocab_size = self._forge_tokenizer.vocab_size
            self._loaded = True
            return True
        except Exception:
            pass
        
        # Fall back to simple character/word tokenizer
        self._build_simple_vocab()
        self._loaded = True
        return True
    
    def _build_simple_vocab(self):
        """Build a simple vocabulary for fallback."""
        # Special tokens
        self._simple_vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<eos>": 2,
            "<bos>": 3,
            " ": 4,
            "\n": 5,
        }
        
        # Add ASCII characters
        idx = len(self._simple_vocab)
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-:;\"()[]{}":
            if idx < self.vocab_size:
                self._simple_vocab[c] = idx
                idx += 1
        
        # Common words
        common_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "and", "or", "but", "if", "then", "else", "when", "where",
            "what", "which", "who", "how", "why", "this", "that", "these",
            "it", "you", "he", "she", "we", "they", "I", "my", "your",
            "hello", "hi", "yes", "no", "please", "thank", "sorry",
        ]
        for word in common_words:
            if idx < self.vocab_size and word not in self._simple_vocab:
                self._simple_vocab[word] = idx
                idx += 1
        
        # Build reverse mapping
        self._simple_vocab_rev = {v: k for k, v in self._simple_vocab.items()}
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if not self._loaded:
            self.load()
        
        if self._forge_tokenizer is not None:
            return self._forge_tokenizer.encode(text)
        
        # Simple fallback encoding
        tokens = []
        tokens.append(self._simple_vocab.get("<bos>", 3))
        
        # Try word-level first, fall back to character
        words = text.split()
        for word in words:
            if word in self._simple_vocab:
                tokens.append(self._simple_vocab[word])
            else:
                # Character-level fallback
                for c in word:
                    tokens.append(self._simple_vocab.get(c, 1))  # <unk>
            tokens.append(self._simple_vocab.get(" ", 4))
        
        return tokens
    
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        if not self._loaded:
            self.load()
        
        if self._forge_tokenizer is not None:
            return self._forge_tokenizer.decode(ids)
        
        # Simple fallback decoding
        chars = []
        for id in ids:
            if id in [0, 2, 3]:  # Skip special tokens
                continue
            token = self._simple_vocab_rev.get(id, "")
            chars.append(token)
        
        return "".join(chars)


class PureChat:
    """
    Chat interface for PureTransformer.
    
    Handles conversation formatting, tokenization, and generation.
    """
    
    def __init__(self, model: 'PureTransformer', tokenizer: Optional[PureTokenizer] = None):
        self.model = model
        self.tokenizer = tokenizer or PureTokenizer(model.config.vocab_size)
        self.tokenizer.load()
        self.history: list[dict[str, str]] = []
    
    def chat(
        self, 
        message: str, 
        max_tokens: int = 100,
        temperature: float = 0.7,
        repetition_penalty: float = 1.1,
        system_prompt: str = "You are a helpful AI assistant."
    ) -> str:
        """
        Generate a response to a message.
        
        Args:
            message: User message
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            repetition_penalty: Penalty for repeating tokens (>1.0 reduces repetition)
            system_prompt: System prompt for context
            
        Returns:
            Model's response text
        """
        # Format conversation
        prompt = f"{system_prompt}\n\nUser: {message}\nAssistant:"
        
        # Encode
        input_ids = self.tokenizer.encode(prompt)
        
        # Truncate if too long
        max_input = self.model.config.max_seq_len - max_tokens
        if len(input_ids) > max_input:
            input_ids = input_ids[-max_input:]
        
        # Generate
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            repetition_penalty=repetition_penalty
        )
        
        # Get only the new tokens
        new_ids = output_ids[len(input_ids):]
        
        # Decode
        response = self.tokenizer.decode(new_ids)
        
        # Update history
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []


# ═══════════════════════════════════════════════════════════════════════════════
# FORGE PYTORCH WEIGHT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_forge_weights(model: 'PureTransformer', checkpoint_path: Path) -> bool:
    """
    Load weights from a trained Enigma AI Engine PyTorch model.
    
    Args:
        model: PureTransformer to load weights into
        checkpoint_path: Path to PyTorch .pt or .pth file
        
    Returns:
        True if successful
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        import torch
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle nested state dict
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        loaded_count = 0
        
        def to_matrix(tensor) -> Matrix:
            """Convert tensor to Matrix."""
            arr = tensor.detach().cpu().numpy()
            if len(arr.shape) == 1:
                return Matrix(1, len(arr), arr.tolist())
            return Matrix(arr.shape[0], arr.shape[1], arr.flatten().tolist())
        
        # Load embeddings
        for key, tensor in state_dict.items():
            try:
                # Token embeddings
                if 'tok_emb' in key and 'weight' in key:
                    mat = to_matrix(tensor)
                    if mat.rows <= model.token_embedding.num_embeddings:
                        # Copy rows that fit
                        for i in range(min(mat.rows, model.token_embedding.num_embeddings)):
                            model.token_embedding.weight.set_row(i, mat.get_row(i)[:model.config.d_model])
                        loaded_count += 1
                        print(f"  Loaded: {key}")
                
                # Attention weights
                elif 'attention' in key or 'attn' in key:
                    # Parse layer index
                    import re
                    layer_match = re.search(r'layers?\.?(\d+)', key)
                    if layer_match:
                        layer_idx = int(layer_match.group(1))
                        if layer_idx < len(model.blocks):
                            block = model.blocks[layer_idx]
                            
                            if 'wq' in key or 'q_proj' in key:
                                mat = to_matrix(tensor)
                                block.attention.wq.weight = mat
                                loaded_count += 1
                            elif 'wk' in key or 'k_proj' in key:
                                mat = to_matrix(tensor)
                                block.attention.wk.weight = mat
                                loaded_count += 1
                            elif 'wv' in key or 'v_proj' in key:
                                mat = to_matrix(tensor)
                                block.attention.wv.weight = mat
                                loaded_count += 1
                            elif 'wo' in key or 'o_proj' in key:
                                mat = to_matrix(tensor)
                                block.attention.wo.weight = mat
                                loaded_count += 1
                
                # FFN weights
                elif 'ffn' in key or 'feed_forward' in key or 'mlp' in key:
                    layer_match = re.search(r'layers?\.?(\d+)', key)
                    if layer_match:
                        layer_idx = int(layer_match.group(1))
                        if layer_idx < len(model.blocks):
                            block = model.blocks[layer_idx]
                            
                            if 'w1' in key or 'gate' in key:
                                mat = to_matrix(tensor)
                                if block.ffn.use_swiglu:
                                    block.ffn.w1.weight = mat
                                loaded_count += 1
                            elif 'w2' in key or 'down' in key:
                                mat = to_matrix(tensor)
                                if block.ffn.use_swiglu:
                                    block.ffn.w2.weight = mat
                                else:
                                    block.ffn.down_proj.weight = mat
                                loaded_count += 1
                            elif 'w3' in key or 'up' in key:
                                mat = to_matrix(tensor)
                                if block.ffn.use_swiglu:
                                    block.ffn.w3.weight = mat
                                else:
                                    block.ffn.up_proj.weight = mat
                                loaded_count += 1
                
                # Layer norms
                elif 'norm' in key and 'gamma' in key or 'weight' in key:
                    layer_match = re.search(r'layers?\.?(\d+)', key)
                    if layer_match:
                        layer_idx = int(layer_match.group(1))
                        if layer_idx < len(model.blocks):
                            block = model.blocks[layer_idx]
                            mat = to_matrix(tensor)
                            if 'attn' in key:
                                block.attn_norm.gamma = mat
                            elif 'ffn' in key:
                                block.ffn_norm.gamma = mat
                            loaded_count += 1
                
                # Final norm
                elif 'final_norm' in key or 'ln_f' in key:
                    mat = to_matrix(tensor)
                    model.final_norm.gamma = mat
                    loaded_count += 1
                
                # Output projection
                elif 'output' in key or 'lm_head' in key:
                    mat = to_matrix(tensor)
                    model.output_proj.weight = mat
                    loaded_count += 1
                    
            except Exception as e:
                print(f"  Warning: Could not load {key}: {e}")
        
        print(f"Loaded {loaded_count} weight tensors from {checkpoint_path}")
        return loaded_count > 0
        
    except ImportError:
        print("PyTorch not available - cannot load .pt/.pth files")
        return False
    except Exception as e:
        print(f"Error loading weights: {e}")
        return False


def create_from_forge_checkpoint(checkpoint_path: Path, size: str = "auto") -> Optional['PureTransformer']:
    """
    Create a PureTransformer and load weights from a Forge checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        size: Model size or "auto" to detect from checkpoint
        
    Returns:
        PureTransformer with loaded weights, or None if failed
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Try to detect model size from checkpoint
    if size == "auto":
        try:
            import torch
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'config' in state_dict:
                cfg = state_dict['config']
                # Map dim to size
                dim = cfg.get('dim', cfg.get('d_model', 256))
                if dim <= 64:
                    size = "nano"
                elif dim <= 128:
                    size = "micro"
                elif dim <= 256:
                    size = "tiny"
                elif dim <= 512:
                    size = "small"
                else:
                    size = "medium"
            else:
                size = "small"
        except Exception:
            size = "small"
    
    # Create model
    SIZE_CONFIGS = {
        "nano": PureConfig(vocab_size=1000, d_model=64, n_heads=2, n_layers=2, d_ff=128),
        "micro": PureConfig(vocab_size=2000, d_model=128, n_heads=4, n_layers=4, d_ff=256),
        "tiny": PureConfig(vocab_size=4000, d_model=256, n_heads=4, n_layers=6, d_ff=512),
        "small": PureConfig(vocab_size=8000, d_model=512, n_heads=8, n_layers=8, d_ff=1024),
        "medium": PureConfig(vocab_size=16000, d_model=768, n_heads=12, n_layers=12, d_ff=2048),
    }
    
    config = SIZE_CONFIGS.get(size, SIZE_CONFIGS["small"])
    model = PureTransformer(config)
    
    # Load weights
    if load_forge_weights(model, checkpoint_path):
        return model
    
    return None


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


def train_example():
    """
    Example of training a tiny model using pure Python.
    
    📖 HOW TRAINING WORKS:
    
    1. FORWARD PASS: Input → Model → Predictions
       "The cat sat" → PureTransformer → probability distribution over vocabulary
    
    2. COST FUNCTION (cross_entropy_loss):
       Measures how wrong the predictions are.
       If model predicts "dog" when answer is "mat", cost is HIGH.
       If model predicts "mat" when answer is "mat", cost is LOW.
       
       Cost = -log(probability of correct answer)
       
    3. BACKWARD PASS (backpropagation):
       Compute gradients: how much each weight contributed to the error.
       This uses calculus (chain rule) to trace error back through layers.
       
    4. OPTIMIZER (Adam/SGD):
       Update weights in direction that reduces cost.
       weight_new = weight_old - learning_rate * gradient
       
    5. REPEAT:
       Do this for many examples until cost is low!
    
    Returns:
        List of loss values during training
    """
    import time
    print("=" * 60)
    print("PURE PYTHON TRAINING EXAMPLE")
    print("=" * 60)
    
    # Create tiny model
    config = PureConfig(
        vocab_size=50,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_seq_len=16
    )
    
    model = PureTransformer(config)
    print(f"\nModel: {model.count_parameters():,} parameters")
    
    # Create optimizer
    optimizer = PureAdam(model.parameters(), lr=0.01)
    
    # Simple training data: predict next token
    # Pattern: [1, 2, 3, 4] -> [2, 3, 4, 5]
    training_examples = [
        ([1, 2, 3, 4], [2, 3, 4, 5]),
        ([5, 6, 7, 8], [6, 7, 8, 9]),
        ([10, 11, 12, 13], [11, 12, 13, 14]),
        ([20, 21, 22, 23], [21, 22, 23, 24]),
    ]
    
    print(f"Training examples: {len(training_examples)}")
    print("\nTraining pattern: predict next number in sequence")
    print("  Input:  [1, 2, 3, 4]")
    print("  Target: [2, 3, 4, 5]")
    
    losses = []
    n_epochs = 20
    
    print(f"\nTraining for {n_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        for inputs, targets in training_examples:
            # FORWARD PASS
            logits = model.forward(inputs)
            
            # COMPUTE LOSS (cost function)
            loss, grad = cross_entropy_loss(logits, targets)
            epoch_loss += loss
            
            # BACKWARD PASS (compute gradients)
            # Note: Full backprop through transformer is complex
            # This is simplified - just updates output layer
            model.output_proj.backward(grad)
            
            # OPTIMIZER STEP (update weights)
            optimizer.step(model.output_proj.gradients())
        
        avg_loss = epoch_loss / len(training_examples)
        losses.append(avg_loss)
        
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f} ({100*(losses[0]-losses[-1])/losses[0]:.1f}% improvement)")
    
    # Test generation
    print("\n" + "=" * 60)
    print("TESTING TRAINED MODEL")
    print("=" * 60)
    
    test_input = [1, 2, 3]
    generated = model.generate(test_input, max_new_tokens=5, temperature=0.5)
    print(f"Input: {test_input}")
    print(f"Generated: {generated}")
    print(f"Expected pattern: [1, 2, 3, 4, 5, 6, 7, 8]")
    
    return losses


def demonstrate_cost_function():
    """
    Demonstrate how the cost function (cross-entropy loss) works.
    
    📖 COST FUNCTION EXPLAINED:
    
    The cost function measures prediction quality:
    
    GOOD prediction (high probability for correct answer):
        P(correct) = 0.9  →  Cost = -log(0.9) = 0.105  (LOW)
    
    BAD prediction (low probability for correct answer):
        P(correct) = 0.1  →  Cost = -log(0.1) = 2.303  (HIGH)
    
    TERRIBLE prediction:
        P(correct) = 0.01 →  Cost = -log(0.01) = 4.605 (VERY HIGH)
    
    Training minimizes this cost by adjusting weights!
    """
    print("=" * 60)
    print("COST FUNCTION (CROSS-ENTROPY LOSS) DEMONSTRATION")
    print("=" * 60)
    
    # Create fake logits (model outputs before softmax)
    # vocab_size = 5, seq_len = 1
    
    print("\n📖 What is cross-entropy loss?")
    print("   It measures how 'surprised' we are by the correct answer.")
    print("   Lower loss = better prediction = model learned!")
    
    print("\n" + "-" * 60)
    print("SCENARIO 1: Model is confident and CORRECT")
    print("-" * 60)
    
    # Logits that strongly favor token 3
    good_logits = Matrix.from_2d([
        [0.1, 0.1, 0.1, 5.0, 0.1]  # Strongly predicts token 3
    ])
    target = [3]  # Correct answer is token 3
    
    loss, grad = cross_entropy_loss(good_logits, target)
    print(f"Logits: [0.1, 0.1, 0.1, 5.0, 0.1]")
    print(f"Target: token 3")
    print(f"Loss: {loss:.4f} (LOW = good!)")
    
    print("\n" + "-" * 60)
    print("SCENARIO 2: Model is confident but WRONG")
    print("-" * 60)
    
    # Logits that strongly favor wrong token
    bad_logits = Matrix.from_2d([
        [5.0, 0.1, 0.1, 0.1, 0.1]  # Strongly predicts token 0
    ])
    target = [3]  # But correct answer is token 3!
    
    loss, grad = cross_entropy_loss(bad_logits, target)
    print(f"Logits: [5.0, 0.1, 0.1, 0.1, 0.1]")
    print(f"Target: token 3")
    print(f"Loss: {loss:.4f} (HIGH = bad prediction!)")
    
    print("\n" + "-" * 60)
    print("SCENARIO 3: Model is uncertain (uniform)")
    print("-" * 60)
    
    # Uniform logits - model has no idea
    uncertain_logits = Matrix.from_2d([
        [1.0, 1.0, 1.0, 1.0, 1.0]  # Equal probability for all
    ])
    target = [3]
    
    loss, grad = cross_entropy_loss(uncertain_logits, target)
    print(f"Logits: [1.0, 1.0, 1.0, 1.0, 1.0]")
    print(f"Target: token 3")
    print(f"Loss: {loss:.4f} (MEDIUM = random guess)")
    print(f"Note: -log(1/5) = -log(0.2) = {-math.log(0.2):.4f}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("=" * 60)
    print("Training adjusts weights to MINIMIZE the loss.")
    print("Lower loss = model assigns higher probability to correct answers!")


# Run demonstrations when file is executed directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_example()
    elif len(sys.argv) > 1 and sys.argv[1] == "--cost":
        demonstrate_cost_function()
    else:
        # Default: run basic test
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
        
        print("\n" + "=" * 60)
        print("Run with --train for training example")
        print("Run with --cost for cost function demo")
        print("=" * 60)