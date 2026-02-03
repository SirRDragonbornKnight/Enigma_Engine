# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-accelerated kernels for Pure Python Neural Network.

To compile:
    pip install cython
    cd forge_ai/builtin
    cythonize -i cython_kernels.pyx

Or use setup.py:
    python setup.py build_ext --inplace

This provides ~500-1000x speedup over pure Python.
Auto-detected by neural_network.py when compiled.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt, tanh
from cython.parallel import prange

# Type definitions
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_matmul(
    np.ndarray[DTYPE_t, ndim=1] a_data,
    np.ndarray[DTYPE_t, ndim=1] b_data,
    int a_rows, int a_cols, int b_cols
):
    """
    Fast matrix multiplication: C = A @ B
    
    Args:
        a_data: Flattened A matrix (a_rows * a_cols)
        b_data: Flattened B matrix (a_cols * b_cols)
        a_rows, a_cols, b_cols: Matrix dimensions
    
    Returns:
        Flattened result matrix (a_rows * b_cols)
    """
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(a_rows * b_cols, dtype=np.float64)
    cdef int i, j, k
    cdef int a_row_start, b_row_start, result_row_start
    cdef double a_ik
    
    for i in prange(a_rows, nogil=True):
        a_row_start = i * a_cols
        result_row_start = i * b_cols
        for k in range(a_cols):
            a_ik = a_data[a_row_start + k]
            b_row_start = k * b_cols
            for j in range(b_cols):
                result[result_row_start + j] += a_ik * b_data[b_row_start + j]
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_softmax(
    np.ndarray[DTYPE_t, ndim=1] data,
    int rows, int cols
):
    """Row-wise softmax with numerical stability."""
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(rows * cols, dtype=np.float64)
    cdef int i, j, row_start
    cdef double max_val, exp_sum, val
    
    for i in range(rows):
        row_start = i * cols
        
        # Find max for stability
        max_val = data[row_start]
        for j in range(1, cols):
            if data[row_start + j] > max_val:
                max_val = data[row_start + j]
        
        # Compute exp and sum
        exp_sum = 0.0
        for j in range(cols):
            val = exp(data[row_start + j] - max_val)
            result[row_start + j] = val
            exp_sum += val
        
        # Normalize
        for j in range(cols):
            result[row_start + j] /= exp_sum
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_silu(np.ndarray[DTYPE_t, ndim=1] data):
    """SiLU/Swish activation: x * sigmoid(x)"""
    cdef int n = data.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i
    cdef double x, sig
    
    for i in prange(n, nogil=True):
        x = data[i]
        sig = 1.0 / (1.0 + exp(-x))
        result[i] = x * sig
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_rms_norm(
    np.ndarray[DTYPE_t, ndim=1] data,
    np.ndarray[DTYPE_t, ndim=1] weight,
    int rows, int cols, double eps
):
    """RMS Layer Normalization."""
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(rows * cols, dtype=np.float64)
    cdef int i, j, row_start
    cdef double ss, rms, val
    
    for i in range(rows):
        row_start = i * cols
        
        # Compute sum of squares
        ss = 0.0
        for j in range(cols):
            val = data[row_start + j]
            ss += val * val
        
        # RMS
        rms = sqrt(ss / cols + eps)
        
        # Normalize and scale
        for j in range(cols):
            result[row_start + j] = weight[j] * data[row_start + j] / rms
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_gelu(np.ndarray[DTYPE_t, ndim=1] data):
    """GELU activation approximation."""
    cdef int n = data.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i
    cdef double x, inner
    cdef double sqrt_2_pi = 0.7978845608028654  # sqrt(2/pi)
    
    for i in prange(n, nogil=True):
        x = data[i]
        inner = sqrt_2_pi * (x + 0.044715 * x * x * x)
        result[i] = 0.5 * x * (1.0 + tanh(inner))
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_add(
    np.ndarray[DTYPE_t, ndim=1] a,
    np.ndarray[DTYPE_t, ndim=1] b
):
    """Element-wise addition."""
    cdef int n = a.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i
    
    for i in prange(n, nogil=True):
        result[i] = a[i] + b[i]
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_hadamard(
    np.ndarray[DTYPE_t, ndim=1] a,
    np.ndarray[DTYPE_t, ndim=1] b
):
    """Element-wise multiplication (Hadamard product)."""
    cdef int n = a.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i
    
    for i in prange(n, nogil=True):
        result[i] = a[i] * b[i]
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_scale(np.ndarray[DTYPE_t, ndim=1] data, double scalar):
    """Scalar multiplication."""
    cdef int n = data.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i
    
    for i in prange(n, nogil=True):
        result[i] = data[i] * scalar
    
    return result
