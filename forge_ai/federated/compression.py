"""
================================================================================
UPDATE COMPRESSION - REDUCE BANDWIDTH FOR FEDERATED LEARNING
================================================================================

Compresses model updates for faster transfer over network.
Important for devices with slow connections.

FILE: forge_ai/federated/compression.py
TYPE: Update Compression
MAIN CLASS: UpdateCompressor

TECHNIQUES:
    - Quantization: Reduce floating point precision
    - Sparsification: Send only top-K% of changes
    - Gradient compression: Various compression methods

USAGE:
    compressor = UpdateCompressor()
    compressed = compressor.compress(update)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np

from .federation import ModelUpdate

logger = logging.getLogger(__name__)


@dataclass
class CompressedUpdate:
    """
    Compressed model update for efficient transfer.
    """
    device_id: str
    round_number: int
    compressed_deltas: Dict[str, 'CompressedTensor']
    num_samples: int
    loss: float
    compression_ratio: float = 0.0
    
    def to_model_update(self) -> ModelUpdate:
        """
        Decompress to full ModelUpdate.
        
        Returns:
            Decompressed ModelUpdate
        """
        weight_deltas = {}
        
        for layer_name, compressed in self.compressed_deltas.items():
            weight_deltas[layer_name] = compressed.decompress()
        
        return ModelUpdate(
            device_id=self.device_id,
            round_number=self.round_number,
            weight_deltas=weight_deltas,
            num_samples=self.num_samples,
            loss=self.loss,
        )


@dataclass
class CompressedTensor:
    """
    Compressed representation of a tensor.
    """
    data: np.ndarray
    shape: Tuple[int, ...]
    dtype: str
    compression_method: str
    metadata: Dict = field(default_factory=dict)
    
    def decompress(self) -> np.ndarray:
        """
        Decompress tensor to original representation.
        
        Returns:
            Decompressed numpy array
        """
        if self.compression_method == 'quantized':
            return self._dequantize()
        elif self.compression_method == 'sparse':
            return self._densify()
        else:
            return self.data.reshape(self.shape)
    
    def _dequantize(self) -> np.ndarray:
        """Dequantize tensor."""
        scale = self.metadata.get('scale', 1.0)
        zero_point = self.metadata.get('zero_point', 0)
        
        # Dequantize: real_value = (quantized_value - zero_point) * scale
        dequantized = (self.data.astype(np.float32) - zero_point) * scale
        return dequantized.reshape(self.shape)
    
    def _densify(self) -> np.ndarray:
        """Convert sparse representation to dense."""
        indices = self.metadata['indices']
        values = self.data
        
        # Create dense tensor
        dense = np.zeros(self.shape, dtype=np.float32)
        dense[indices] = values
        
        return dense


@dataclass
class SparseUpdate:
    """
    Sparse model update (only non-zero or top-K values).
    """
    indices: Tuple[np.ndarray, ...]
    values: np.ndarray
    shape: Tuple[int, ...]
    
    def to_dense(self) -> np.ndarray:
        """
        Convert to dense array.
        
        Returns:
            Dense numpy array
        """
        dense = np.zeros(self.shape, dtype=self.values.dtype)
        dense[self.indices] = self.values
        return dense


class UpdateCompressor:
    """
    Compress updates for faster transfer.
    
    Techniques:
    - Quantization (reduce precision)
    - Sparsification (only send large changes)
    - Gradient compression
    """
    
    def __init__(
        self,
        quantization_bits: int = 8,
        sparsity: float = 0.1,
        method: str = "auto"
    ):
        """
        Initialize update compressor.
        
        Args:
            quantization_bits: Bits for quantization (8, 16, 32)
            sparsity: Top-K% of values to keep (0.1 = top 10%)
            method: Compression method ("auto", "quantize", "sparsify", "both")
        """
        self.quantization_bits = quantization_bits
        self.sparsity = sparsity
        self.method = method
        
        logger.info(
            f"Initialized compressor: {quantization_bits}-bit quantization, "
            f"{sparsity*100}% sparsity, method={method}"
        )
    
    def compress(self, update: ModelUpdate) -> CompressedUpdate:
        """
        Compress weight updates.
        
        Args:
            update: Original model update
        
        Returns:
            Compressed update
        """
        compressed_deltas = {}
        original_size = 0
        compressed_size = 0
        
        for layer_name, weights in update.weight_deltas.items():
            original_size += weights.nbytes
            
            # Choose compression method
            if self.method == "auto":
                # Use sparsity for large layers, quantization for small
                if weights.size > 1000:
                    method = "sparsify"
                else:
                    method = "quantize"
            else:
                method = self.method
            
            # Compress
            if method == "quantize":
                compressed = self._quantize(weights)
            elif method == "sparsify":
                compressed = self._sparsify(weights)
            elif method == "both":
                # First sparsify, then quantize
                sparse = self._sparsify(weights)
                compressed = CompressedTensor(
                    data=self._quantize_array(sparse.data, self.quantization_bits),
                    shape=weights.shape,
                    dtype=str(weights.dtype),
                    compression_method='both',
                    metadata={
                        'indices': sparse.metadata['indices'],
                        **self._get_quantization_metadata(sparse.data),
                    },
                )
            else:
                # No compression
                compressed = CompressedTensor(
                    data=weights,
                    shape=weights.shape,
                    dtype=str(weights.dtype),
                    compression_method='none',
                )
            
            compressed_deltas[layer_name] = compressed
            compressed_size += compressed.data.nbytes
        
        compression_ratio = compressed_size / max(1, original_size)
        
        logger.info(
            f"Compressed update: {original_size} -> {compressed_size} bytes "
            f"({compression_ratio*100:.1f}% of original)"
        )
        
        return CompressedUpdate(
            device_id=update.device_id,
            round_number=update.round_number,
            compressed_deltas=compressed_deltas,
            num_samples=update.num_samples,
            loss=update.loss,
            compression_ratio=compression_ratio,
        )
    
    def _quantize(self, weights: np.ndarray) -> CompressedTensor:
        """
        Quantize weights to lower bit precision.
        
        Args:
            weights: Weight array
        
        Returns:
            Compressed tensor
        """
        quantized, metadata = self._quantize_array(weights, self.quantization_bits)
        
        return CompressedTensor(
            data=quantized,
            shape=weights.shape,
            dtype=str(weights.dtype),
            compression_method='quantized',
            metadata=metadata,
        )
    
    def _quantize_array(self, weights: np.ndarray, bits: int) -> Tuple[np.ndarray, Dict]:
        """
        Quantize array to specified bits.
        
        Args:
            weights: Weight array
            bits: Number of bits (8, 16, 32)
        
        Returns:
            Tuple of (quantized array, metadata)
        """
        # Calculate quantization parameters
        min_val = np.min(weights)
        max_val = np.max(weights)
        
        # Avoid division by zero
        if max_val == min_val:
            # All values are the same, no need to quantize
            return weights, {}
        
        # Calculate scale and zero point
        if bits == 8:
            qmin, qmax = 0, 255
            dtype = np.uint8
        elif bits == 16:
            qmin, qmax = 0, 65535
            dtype = np.uint16
        else:
            # 32-bit, no quantization needed
            return weights, {}
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        
        # Quantize
        quantized = np.round(weights / scale + zero_point).clip(qmin, qmax).astype(dtype)
        
        metadata = {
            'scale': scale,
            'zero_point': zero_point,
            'min_val': min_val,
            'max_val': max_val,
            'bits': bits,
        }
        
        return quantized, metadata
    
    def _get_quantization_metadata(self, values: np.ndarray) -> Dict:
        """Get quantization metadata for values."""
        _, metadata = self._quantize_array(values, self.quantization_bits)
        return metadata
    
    def _sparsify(self, weights: np.ndarray) -> CompressedTensor:
        """
        Only send top K% of changes by magnitude.
        
        Args:
            weights: Weight array
        
        Returns:
            Sparse compressed tensor
        """
        # Calculate threshold for top-K values
        abs_weights = np.abs(weights)
        threshold = np.percentile(abs_weights, (1 - self.sparsity) * 100)
        
        # Create mask for values above threshold
        mask = abs_weights > threshold
        
        # Get indices and values
        indices = np.where(mask)
        values = weights[mask]
        
        return CompressedTensor(
            data=values,
            shape=weights.shape,
            dtype=str(weights.dtype),
            compression_method='sparse',
            metadata={
                'indices': indices,
                'threshold': threshold,
                'sparsity': 1 - (len(values) / weights.size),
            },
        )
    
    def decompress(self, compressed: CompressedUpdate) -> ModelUpdate:
        """
        Decompress update back to full representation.
        
        Args:
            compressed: Compressed update
        
        Returns:
            Full model update
        """
        return compressed.to_model_update()
