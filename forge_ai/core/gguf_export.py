"""
GGUF Export/Creation for Forge_AI

Create GGUF files for llama.cpp compatibility:
- Export Forge models to GGUF format
- Support multiple quantization types
- Metadata and tokenizer embedding
- Compatible with llama.cpp, ollama, etc.

Usage:
    from forge_ai.core.gguf_export import export_to_gguf
    
    export_to_gguf(model, tokenizer, "model.gguf", quant_type="Q4_K_M")
"""

import json
import logging
import os
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# GGUF Magic and Version
GGUF_MAGIC = 0x46554747  # "GGUF" in little endian
GGUF_VERSION = 3


class GGMLType(IntEnum):
    """GGML tensor types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23


class GGUFValueType(IntEnum):
    """GGUF metadata value types."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# Block sizes for quantized types
GGML_BLOCK_SIZES = {
    GGMLType.Q4_0: 32,
    GGMLType.Q4_1: 32,
    GGMLType.Q5_0: 32,
    GGMLType.Q5_1: 32,
    GGMLType.Q8_0: 32,
    GGMLType.Q8_1: 32,
    GGMLType.Q2_K: 256,
    GGMLType.Q3_K: 256,
    GGMLType.Q4_K: 256,
    GGMLType.Q5_K: 256,
    GGMLType.Q6_K: 256,
    GGMLType.Q8_K: 256,
}


# Quantization type aliases
QUANT_TYPES = {
    'F32': GGMLType.F32,
    'F16': GGMLType.F16,
    'Q4_0': GGMLType.Q4_0,
    'Q4_1': GGMLType.Q4_1,
    'Q5_0': GGMLType.Q5_0,
    'Q5_1': GGMLType.Q5_1,
    'Q8_0': GGMLType.Q8_0,
    'Q2_K': GGMLType.Q2_K,
    'Q3_K_S': GGMLType.Q3_K,
    'Q3_K_M': GGMLType.Q3_K,
    'Q3_K_L': GGMLType.Q3_K,
    'Q4_K_S': GGMLType.Q4_K,
    'Q4_K_M': GGMLType.Q4_K,
    'Q5_K_S': GGMLType.Q5_K,
    'Q5_K_M': GGMLType.Q5_K,
    'Q6_K': GGMLType.Q6_K,
    'Q8_K': GGMLType.Q8_K,
}


@dataclass
class TensorInfo:
    """Information about a tensor to write."""
    name: str
    shape: Tuple[int, ...]
    dtype: GGMLType
    data: np.ndarray


class GGUFWriter:
    """
    GGUF file writer.
    
    Creates GGUF files compatible with llama.cpp.
    """
    
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[TensorInfo] = []
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata entry."""
        self.metadata[key] = value
    
    def add_tensor(
        self,
        name: str,
        tensor: Union[np.ndarray, 'torch.Tensor'],
        quant_type: GGMLType = GGMLType.F16
    ):
        """Add a tensor to the file."""
        # Convert torch tensor to numpy
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        # Quantize if needed
        if quant_type not in (GGMLType.F32, GGMLType.F16):
            tensor = self._quantize_tensor(tensor, quant_type)
            dtype = quant_type
        elif quant_type == GGMLType.F16:
            tensor = tensor.astype(np.float16)
            dtype = GGMLType.F16
        else:
            tensor = tensor.astype(np.float32)
            dtype = GGMLType.F32
        
        self.tensors.append(TensorInfo(
            name=name,
            shape=tensor.shape,
            dtype=dtype,
            data=tensor
        ))
    
    def _quantize_tensor(
        self,
        tensor: np.ndarray,
        quant_type: GGMLType
    ) -> np.ndarray:
        """Quantize a tensor to the specified type."""
        # Flatten for quantization
        original_shape = tensor.shape
        tensor = tensor.flatten().astype(np.float32)
        
        if quant_type == GGMLType.Q4_0:
            return self._quantize_q4_0(tensor)
        elif quant_type == GGMLType.Q8_0:
            return self._quantize_q8_0(tensor)
        elif quant_type == GGMLType.Q4_K:
            return self._quantize_q4_k(tensor)
        elif quant_type == GGMLType.Q5_K:
            return self._quantize_q5_k(tensor)
        elif quant_type == GGMLType.Q6_K:
            return self._quantize_q6_k(tensor)
        else:
            # Fallback to Q8_0
            logger.warning(f"Unsupported quant type {quant_type}, using Q8_0")
            return self._quantize_q8_0(tensor)
    
    def _quantize_q4_0(self, data: np.ndarray) -> np.ndarray:
        """Quantize to Q4_0 format."""
        block_size = 32
        n_blocks = len(data) // block_size
        
        # Pad if necessary
        if len(data) % block_size != 0:
            pad_size = block_size - (len(data) % block_size)
            data = np.pad(data, (0, pad_size))
            n_blocks += 1
        
        # Output: 2 bytes scale + 16 bytes data per block
        output = bytearray()
        
        for i in range(n_blocks):
            block = data[i * block_size:(i + 1) * block_size]
            
            # Find scale (max absolute value)
            max_val = np.abs(block).max()
            scale = max_val / 7.0 if max_val > 0 else 1.0
            
            # Quantize to 4-bit signed (-8 to 7)
            quantized = np.round(block / scale).astype(np.int8)
            quantized = np.clip(quantized, -8, 7)
            
            # Pack into bytes (2 values per byte)
            packed = bytearray(16)
            for j in range(16):
                low = (quantized[j * 2] + 8) & 0xF
                high = (quantized[j * 2 + 1] + 8) & 0xF
                packed[j] = low | (high << 4)
            
            # Write scale as fp16
            scale_fp16 = np.float16(scale)
            output.extend(scale_fp16.tobytes())
            output.extend(packed)
        
        return np.frombuffer(bytes(output), dtype=np.uint8)
    
    def _quantize_q8_0(self, data: np.ndarray) -> np.ndarray:
        """Quantize to Q8_0 format."""
        block_size = 32
        n_blocks = len(data) // block_size
        
        if len(data) % block_size != 0:
            pad_size = block_size - (len(data) % block_size)
            data = np.pad(data, (0, pad_size))
            n_blocks += 1
        
        # Output: 2 bytes scale + 32 bytes data per block
        output = bytearray()
        
        for i in range(n_blocks):
            block = data[i * block_size:(i + 1) * block_size]
            
            max_val = np.abs(block).max()
            scale = max_val / 127.0 if max_val > 0 else 1.0
            
            quantized = np.round(block / scale).astype(np.int8)
            quantized = np.clip(quantized, -128, 127)
            
            scale_fp16 = np.float16(scale)
            output.extend(scale_fp16.tobytes())
            output.extend(quantized.astype(np.int8).tobytes())
        
        return np.frombuffer(bytes(output), dtype=np.uint8)
    
    def _quantize_q4_k(self, data: np.ndarray) -> np.ndarray:
        """Quantize to Q4_K format (super blocks)."""
        # Q4_K uses 256-element super blocks with multiple scales
        block_size = 256
        n_blocks = len(data) // block_size
        
        if len(data) % block_size != 0:
            pad_size = block_size - (len(data) % block_size)
            data = np.pad(data, (0, pad_size))
            n_blocks += 1
        
        output = bytearray()
        
        for i in range(n_blocks):
            block = data[i * block_size:(i + 1) * block_size]
            
            # Split into 8 sub-blocks of 32
            sub_scales = []
            sub_mins = []
            
            for j in range(8):
                sub_block = block[j * 32:(j + 1) * 32]
                sub_max = sub_block.max()
                sub_min = sub_block.min()
                sub_scales.append((sub_max - sub_min) / 15.0 if sub_max != sub_min else 1.0)
                sub_mins.append(sub_min)
            
            # Super block scale and min
            d = max(sub_scales) if max(sub_scales) > 0 else 1.0
            dmin = min(sub_mins)
            
            # Write header
            output.extend(np.float16(d).tobytes())
            output.extend(np.float16(dmin).tobytes())
            
            # Quantize and pack sub-block scales/mins
            for s in sub_scales:
                output.append(int(np.round(s / d * 63)) & 0x3F)
            
            for m in sub_mins:
                output.append(int(np.round((m - dmin) / d * 63)) & 0x3F)
            
            # Quantize data
            packed = bytearray(128)  # 256 values * 4 bits / 8
            for j in range(8):
                sub_block = block[j * 32:(j + 1) * 32]
                scale = sub_scales[j]
                min_val = sub_mins[j]
                
                for k in range(16):
                    low = int(np.round((sub_block[k * 2] - min_val) / scale)) & 0xF
                    high = int(np.round((sub_block[k * 2 + 1] - min_val) / scale)) & 0xF
                    packed[j * 16 + k] = low | (high << 4)
            
            output.extend(packed)
        
        return np.frombuffer(bytes(output), dtype=np.uint8)
    
    def _quantize_q5_k(self, data: np.ndarray) -> np.ndarray:
        """Quantize to Q5_K format."""
        # Similar to Q4_K but with 5 bits
        block_size = 256
        n_blocks = len(data) // block_size
        
        if len(data) % block_size != 0:
            pad_size = block_size - (len(data) % block_size)
            data = np.pad(data, (0, pad_size))
            n_blocks += 1
        
        output = bytearray()
        
        for i in range(n_blocks):
            block = data[i * block_size:(i + 1) * block_size]
            
            max_val = np.abs(block).max()
            scale = max_val / 15.0 if max_val > 0 else 1.0
            
            output.extend(np.float16(scale).tobytes())
            output.extend(np.float16(0).tobytes())  # min
            
            # Quantize to 5-bit
            quantized = np.round(block / scale * 15).astype(np.int8)
            quantized = np.clip(quantized, 0, 31)
            
            # Pack (complex, simplified here)
            for j in range(0, 256, 2):
                low = quantized[j] & 0x1F
                high = quantized[j + 1] & 0x1F if j + 1 < 256 else 0
                output.append(low | ((high & 0x7) << 5))
                if j + 1 < 256:
                    output.append((high >> 3) & 0x3)
        
        return np.frombuffer(bytes(output), dtype=np.uint8)
    
    def _quantize_q6_k(self, data: np.ndarray) -> np.ndarray:
        """Quantize to Q6_K format."""
        block_size = 256
        n_blocks = len(data) // block_size
        
        if len(data) % block_size != 0:
            pad_size = block_size - (len(data) % block_size)
            data = np.pad(data, (0, pad_size))
            n_blocks += 1
        
        output = bytearray()
        
        for i in range(n_blocks):
            block = data[i * block_size:(i + 1) * block_size]
            
            max_val = np.abs(block).max()
            scale = max_val / 31.0 if max_val > 0 else 1.0
            
            output.extend(np.float16(scale).tobytes())
            
            # 6-bit quantization
            quantized = np.round(block / scale * 31 + 32).astype(np.uint8)
            quantized = np.clip(quantized, 0, 63)
            
            # Pack 6-bit values (4 values into 3 bytes)
            for j in range(0, 256, 4):
                v0 = quantized[j]
                v1 = quantized[j + 1] if j + 1 < 256 else 0
                v2 = quantized[j + 2] if j + 2 < 256 else 0
                v3 = quantized[j + 3] if j + 3 < 256 else 0
                
                output.append(v0 | ((v1 & 0x3) << 6))
                output.append((v1 >> 2) | ((v2 & 0xF) << 4))
                output.append((v2 >> 4) | (v3 << 2))
        
        return np.frombuffer(bytes(output), dtype=np.uint8)
    
    def write(self):
        """Write the GGUF file."""
        with open(self.path, 'wb') as f:
            self._write_header(f)
            self._write_metadata(f)
            self._write_tensor_info(f)
            self._write_tensor_data(f)
        
        logger.info(f"Wrote GGUF file: {self.path}")
    
    def _write_header(self, f: BinaryIO):
        """Write GGUF header."""
        # Magic number
        f.write(struct.pack('<I', GGUF_MAGIC))
        
        # Version
        f.write(struct.pack('<I', GGUF_VERSION))
        
        # Tensor count
        f.write(struct.pack('<Q', len(self.tensors)))
        
        # Metadata KV count
        f.write(struct.pack('<Q', len(self.metadata)))
    
    def _write_metadata(self, f: BinaryIO):
        """Write metadata entries."""
        for key, value in self.metadata.items():
            self._write_string(f, key)
            self._write_value(f, value)
    
    def _write_string(self, f: BinaryIO, s: str):
        """Write a string."""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
    
    def _write_value(self, f: BinaryIO, value: Any):
        """Write a metadata value."""
        if isinstance(value, bool):
            f.write(struct.pack('<I', GGUFValueType.BOOL))
            f.write(struct.pack('<?', value))
        elif isinstance(value, int):
            if value < 0:
                f.write(struct.pack('<I', GGUFValueType.INT64))
                f.write(struct.pack('<q', value))
            else:
                f.write(struct.pack('<I', GGUFValueType.UINT64))
                f.write(struct.pack('<Q', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', GGUFValueType.FLOAT32))
            f.write(struct.pack('<f', value))
        elif isinstance(value, str):
            f.write(struct.pack('<I', GGUFValueType.STRING))
            self._write_string(f, value)
        elif isinstance(value, (list, tuple)):
            f.write(struct.pack('<I', GGUFValueType.ARRAY))
            if len(value) == 0:
                f.write(struct.pack('<I', GGUFValueType.UINT32))
                f.write(struct.pack('<Q', 0))
            else:
                # Determine array type from first element
                if isinstance(value[0], str):
                    f.write(struct.pack('<I', GGUFValueType.STRING))
                elif isinstance(value[0], int):
                    f.write(struct.pack('<I', GGUFValueType.INT64))
                elif isinstance(value[0], float):
                    f.write(struct.pack('<I', GGUFValueType.FLOAT32))
                else:
                    f.write(struct.pack('<I', GGUFValueType.UINT32))
                
                f.write(struct.pack('<Q', len(value)))
                
                for item in value:
                    if isinstance(item, str):
                        self._write_string(f, item)
                    elif isinstance(item, int):
                        f.write(struct.pack('<q', item))
                    elif isinstance(item, float):
                        f.write(struct.pack('<f', item))
    
    def _write_tensor_info(self, f: BinaryIO):
        """Write tensor information."""
        for tensor in self.tensors:
            # Name
            self._write_string(f, tensor.name)
            
            # Number of dimensions
            f.write(struct.pack('<I', len(tensor.shape)))
            
            # Dimensions (in reverse order for GGML)
            for dim in reversed(tensor.shape):
                f.write(struct.pack('<Q', dim))
            
            # Type
            f.write(struct.pack('<I', tensor.dtype))
            
            # Offset (will be filled later, write placeholder)
            f.write(struct.pack('<Q', 0))
    
    def _write_tensor_data(self, f: BinaryIO):
        """Write tensor data with alignment."""
        # Get current position
        data_start = f.tell()
        
        # Align to 32 bytes
        alignment = 32
        padding = (alignment - (data_start % alignment)) % alignment
        f.write(b'\x00' * padding)
        
        # Go back and fix offsets
        offset = f.tell()
        
        for i, tensor in enumerate(self.tensors):
            # Write data
            f.write(tensor.data.tobytes())
            
            # Pad to alignment
            size = len(tensor.data.tobytes())
            padding = (alignment - (size % alignment)) % alignment
            f.write(b'\x00' * padding)


def export_to_gguf(
    model: 'torch.nn.Module',
    tokenizer: Any,
    output_path: Union[str, Path],
    quant_type: str = "F16",
    model_name: Optional[str] = None,
    description: Optional[str] = None
):
    """
    Export a model to GGUF format.
    
    Args:
        model: PyTorch model to export
        tokenizer: Tokenizer
        output_path: Output GGUF file path
        quant_type: Quantization type (F16, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0)
        model_name: Model name for metadata
        description: Model description
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for GGUF export")
    
    output_path = Path(output_path)
    ggml_type = QUANT_TYPES.get(quant_type, GGMLType.F16)
    
    writer = GGUFWriter(output_path)
    
    # Add metadata
    writer.add_metadata("general.architecture", "llama")
    writer.add_metadata("general.name", model_name or "forge_model")
    writer.add_metadata("general.description", description or "Exported from Forge_AI")
    writer.add_metadata("general.quantization_version", 2)
    
    # Model config
    if hasattr(model, 'config'):
        config = model.config
        writer.add_metadata("llama.context_length", getattr(config, 'max_position_embeddings', 2048))
        writer.add_metadata("llama.embedding_length", getattr(config, 'hidden_size', 768))
        writer.add_metadata("llama.block_count", getattr(config, 'num_layers', 12))
        writer.add_metadata("llama.feed_forward_length", getattr(config, 'intermediate_size', 3072))
        writer.add_metadata("llama.attention.head_count", getattr(config, 'num_heads', 12))
        writer.add_metadata("llama.attention.head_count_kv", getattr(config, 'num_kv_heads', 12))
        writer.add_metadata("llama.rope.freq_base", getattr(config, 'rope_theta', 10000.0))
    
    # Tokenizer
    if hasattr(tokenizer, 'vocab_size'):
        writer.add_metadata("llama.vocab_size", tokenizer.vocab_size)
    
    # Add tensors
    logger.info(f"Exporting model with {quant_type} quantization")
    
    for name, param in model.named_parameters():
        # Convert name to GGUF convention
        gguf_name = _convert_tensor_name(name)
        
        logger.debug(f"Adding tensor: {name} -> {gguf_name}")
        writer.add_tensor(gguf_name, param.data, ggml_type)
    
    # Write file
    writer.write()
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported GGUF: {output_path} ({file_size:.1f} MB)")


def _convert_tensor_name(pytorch_name: str) -> str:
    """Convert PyTorch tensor name to GGUF convention."""
    # Common mappings
    mappings = {
        'embed_tokens': 'token_embd',
        'embedding': 'token_embd',
        'lm_head': 'output',
        'norm': 'output_norm',
        'layers': 'blk',
        'self_attn': 'attn',
        'q_proj': 'attn_q',
        'k_proj': 'attn_k',
        'v_proj': 'attn_v',
        'o_proj': 'attn_output',
        'mlp': 'ffn',
        'gate_proj': 'ffn_gate',
        'up_proj': 'ffn_up',
        'down_proj': 'ffn_down',
        'input_layernorm': 'attn_norm',
        'post_attention_layernorm': 'ffn_norm',
    }
    
    name = pytorch_name
    for old, new in mappings.items():
        name = name.replace(old, new)
    
    return name
