"""
GGUF Dequantization & Tensor Parsing
=====================================

Low-level routines for reading tensors out of a GGUF file and
dequantizing quantized weight blocks back to float32.

Moved out of ``gguf_loader.py`` to keep the user-facing API module
small.  Everything in this file is re-exported from ``gguf_loader``
for backward compatibility.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Check for torch (optional for some operations)
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    torch = None  # type: ignore


# ── tensor parsing ────────────────────────────────────────────────────────


def parse_gguf_tensors(
    f,
    header: dict,
    dequantize: bool = True
) -> dict[str, 'torch.Tensor']:
    """
    Parse and extract tensors from GGUF file.
    
    ⚠️ NOTE: Full dequantization of all GGUF quantization types is not yet
    implemented. This function will work for F32 and F16 tensors, but will
    raise NotImplementedError for quantized types unless the gguf library
    is available.
    
    Args:
        f: Open file handle (binary mode)
        header: Parsed header dictionary
        dequantize: If True, dequantize quantized tensors to float32
        
    Returns:
        Dictionary mapping tensor names to PyTorch tensors
        
    Raises:
        NotImplementedError: If quantized tensors are encountered without gguf library
    """
    import struct
    
    if not HAVE_TORCH:
        raise RuntimeError("torch required for tensor parsing")
    
    import numpy as np
    import torch
    
    tensors = {}
    
    # GGUF quantization types
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    
    # Read tensor info entries
    tensor_infos = []
    for _ in range(header['tensor_count']):
        # Read tensor name
        name_len = struct.unpack('<Q', f.read(8))[0]
        name = f.read(name_len).decode('utf-8', errors='replace')
        
        # Read number of dimensions
        n_dims = struct.unpack('<I', f.read(4))[0]
        
        # Read dimensions
        dims = []
        for _ in range(n_dims):
            dims.append(struct.unpack('<Q', f.read(8))[0])
        
        # Read tensor type
        tensor_type = struct.unpack('<I', f.read(4))[0]
        
        # Read offset
        offset = struct.unpack('<Q', f.read(8))[0]
        
        tensor_infos.append({
            'name': name,
            'dims': tuple(reversed(dims)),  # GGUF stores dims reversed
            'type': tensor_type,
            'offset': offset
        })
    
    # Align to tensor data section (GGUF aligns to 32-byte boundary)
    current_pos = f.tell()
    alignment = 32
    aligned_pos = ((current_pos + alignment - 1) // alignment) * alignment
    f.seek(aligned_pos)
    
    tensor_data_start = f.tell()
    
    # Read tensor data
    for info in tensor_infos:
        name = info['name']
        dims = info['dims']
        tensor_type = info['type']
        offset = info['offset']
        
        # Seek to tensor data
        f.seek(tensor_data_start + offset)
        
        # Calculate tensor size
        n_elements = 1
        for dim in dims:
            n_elements *= dim
        
        # Read and convert based on type
        if tensor_type == GGML_TYPE_F32:
            # Float32
            data = np.fromfile(f, dtype=np.float32, count=n_elements)
            tensor = torch.from_numpy(data.reshape(dims))
        elif tensor_type == GGML_TYPE_F16:
            # Float16
            data = np.fromfile(f, dtype=np.float16, count=n_elements)
            tensor = torch.from_numpy(data.reshape(dims))
            if dequantize:
                tensor = tensor.float()
        elif tensor_type == GGML_TYPE_Q4_0:
            # Q4_0 quantized
            if dequantize:
                block_size = 32
                bytes_per_block = 18
                n_blocks = (n_elements + block_size - 1) // block_size
                n_bytes = n_blocks * bytes_per_block
                raw_data = f.read(n_bytes)
                tensor = dequantize_q4_0(raw_data, tuple(dims))
            else:
                logger.warning(f"Skipping quantized tensor (no dequantize): {name}")
                continue
        elif tensor_type == GGML_TYPE_Q8_0:
            # Q8_0 quantized
            if dequantize:
                block_size = 32
                bytes_per_block = 34
                n_blocks = (n_elements + block_size - 1) // block_size
                n_bytes = n_blocks * bytes_per_block
                raw_data = f.read(n_bytes)
                tensor = dequantize_q8_0(raw_data, tuple(dims))
            else:
                logger.warning(f"Skipping quantized tensor (no dequantize): {name}")
                continue
        elif tensor_type in [GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1]:
            # Other quantized types - not yet implemented
            if dequantize:
                logger.warning(f"Dequantization of GGML type {tensor_type} not yet implemented, skipping: {name}")
                continue
            else:
                logger.warning(f"Skipping quantized tensor: {name}")
                continue
        else:
            logger.warning(f"Unknown tensor type {tensor_type} for {name}")
            continue
        
        tensors[name] = tensor
        logger.debug(f"Loaded tensor: {name}, shape: {tensor.shape}, type: {tensor_type}")
    
    return tensors


# ── config extraction ─────────────────────────────────────────────────────


def extract_config_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Extract Forge config parameters from GGUF metadata.
    
    Args:
        metadata: Parsed GGUF metadata dictionary
        
    Returns:
        Dictionary with config parameters
    """
    config = {}
    
    # Extract common LLaMA-style metadata
    if 'llama.embedding_length' in metadata:
        config['dim'] = metadata['llama.embedding_length']
    elif 'llama.embed_length' in metadata:
        config['dim'] = metadata['llama.embed_length']
    
    if 'llama.block_count' in metadata:
        config['n_layers'] = metadata['llama.block_count']
    
    if 'llama.attention.head_count' in metadata:
        config['n_heads'] = metadata['llama.attention.head_count']
    
    if 'llama.attention.head_count_kv' in metadata:
        config['n_kv_heads'] = metadata['llama.attention.head_count_kv']
    
    if 'llama.context_length' in metadata:
        config['max_seq_len'] = metadata['llama.context_length']
    
    # Try to get vocab size from tokenizer metadata
    if 'tokenizer.ggml.tokens' in metadata:
        tokens = metadata['tokenizer.ggml.tokens']
        if isinstance(tokens, list):
            config['vocab_size'] = len(tokens)
        elif isinstance(tokens, str) and '<array' in tokens:
            # Parse array size from string like "<array of 32000 items>"
            import re
            match = re.search(r'(\d+)\s+items', tokens)
            if match:
                config['vocab_size'] = int(match.group(1))
    
    # Set defaults for missing values
    if 'vocab_size' not in config:
        config['vocab_size'] = 32000  # Common default
    if 'dim' not in config:
        config['dim'] = 4096
    if 'n_layers' not in config:
        config['n_layers'] = 32
    if 'n_heads' not in config:
        config['n_heads'] = 32
    if 'max_seq_len' not in config:
        config['max_seq_len'] = 2048
    
    return config


# ── dequantization ────────────────────────────────────────────────────────


def dequantize_q4_0(data: bytes, shape: tuple) -> 'torch.Tensor':
    """
    Dequantize Q4_0 format (4-bit quantization, block size 32).
    
    Q4_0 format:
    - Block size: 32 elements
    - Each block: 1 float16 scale + 16 bytes (32 x 4-bit values)
    - Total: 18 bytes per block
    
    Args:
        data: Raw quantized bytes
        shape: Original tensor shape
        
    Returns:
        Dequantized PyTorch tensor
    """
    import numpy as np
    
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch required for dequantization")
    
    block_size = 32
    bytes_per_block = 18  # 2 (scale) + 16 (data)
    
    # Convert to numpy array
    data_array = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(data_array) // bytes_per_block
    
    # Calculate total elements
    n_elements = n_blocks * block_size
    output = np.zeros(n_elements, dtype=np.float32)
    
    for i in range(n_blocks):
        block_start = i * bytes_per_block
        
        # Extract scale (float16)
        scale_bytes = data_array[block_start:block_start + 2]
        scale = np.frombuffer(scale_bytes.tobytes(), dtype=np.float16)[0]
        
        # Extract packed 4-bit values
        packed = data_array[block_start + 2:block_start + 18]
        
        # Unpack 4-bit values (2 per byte)
        for j in range(16):
            byte_val = packed[j]
            low = (byte_val & 0xF) - 8  # Signed 4-bit (-8 to 7)
            high = ((byte_val >> 4) & 0xF) - 8
            
            output[i * block_size + j * 2] = float(scale) * low
            output[i * block_size + j * 2 + 1] = float(scale) * high
    
    # Reshape to original shape
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    output = output[:total_elements]
    return torch.from_numpy(output.reshape(shape))


def dequantize_q8_0(data: bytes, shape: tuple) -> 'torch.Tensor':
    """
    Dequantize Q8_0 format (8-bit quantization, block size 32).
    
    Q8_0 format:
    - Block size: 32 elements
    - Each block: 1 float16 scale + 32 bytes (32 x 8-bit values)
    - Total: 34 bytes per block
    
    Args:
        data: Raw quantized bytes
        shape: Original tensor shape
        
    Returns:
        Dequantized PyTorch tensor
    """
    import numpy as np
    
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch required for dequantization")
    
    block_size = 32
    bytes_per_block = 34  # 2 (scale) + 32 (data)
    
    # Convert to numpy array
    data_array = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(data_array) // bytes_per_block
    
    # Calculate total elements
    n_elements = n_blocks * block_size
    output = np.zeros(n_elements, dtype=np.float32)
    
    for i in range(n_blocks):
        block_start = i * bytes_per_block
        
        # Extract scale (float16)
        scale_bytes = data_array[block_start:block_start + 2]
        scale = np.frombuffer(scale_bytes.tobytes(), dtype=np.float16)[0]
        
        # Extract 8-bit signed values
        values = data_array[block_start + 2:block_start + 34].astype(np.int8)
        
        # Dequantize: value * scale
        output[i * block_size:(i + 1) * block_size] = values.astype(np.float32) * float(scale)
    
    # Reshape to original shape
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    output = output[:total_elements]
    return torch.from_numpy(output.reshape(shape))
