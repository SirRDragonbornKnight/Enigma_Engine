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

    # GGUF quantization types (subset; see ggml-quants.h ggml_type enum)
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q2_K = 10  # k-quant 2-bit (super-block of 256 elements)
    GGML_TYPE_Q3_K = 11  # k-quant 3-bit (super-block of 256 elements)
    GGML_TYPE_Q4_K = 12  # k-quant 4-bit (super-block of 256 elements)
    GGML_TYPE_Q5_K = 13  # k-quant 5-bit (super-block of 256 elements)
    GGML_TYPE_Q6_K = 14  # k-quant 6-bit (super-block of 256 elements)

    # Read tensor info entries
    tensor_infos = []
    tensor_count = header['tensor_count']
    if tensor_count < 0 or tensor_count > 100_000:
        logger.error("Invalid GGUF tensor_count: %s", tensor_count)
        return {}
    for _ in range(tensor_count):
        try:
            # Read tensor name
            name_len = struct.unpack('<Q', f.read(8))[0]
            if name_len > 1_000_000:
                logger.error("GGUF tensor name_len %d exceeds 1MB limit", name_len)
                return {}
            name = f.read(name_len).decode('utf-8', errors='replace')

            # Read number of dimensions
            n_dims = struct.unpack('<I', f.read(4))[0]
            if n_dims > 16:
                logger.error("GGUF tensor n_dims %d exceeds limit of 16", n_dims)
                return {}

            # Read dimensions
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack('<Q', f.read(8))[0])

            # Read tensor type
            tensor_type = struct.unpack('<I', f.read(4))[0]

            # Read offset
            offset = struct.unpack('<Q', f.read(8))[0]
        except struct.error as exc:
            logger.error("Corrupt GGUF tensor info at entry %d: %s",
                         len(tensor_infos), exc)
            break

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

    # Get file size for offset validation (S197)
    _saved_pos = f.tell()
    f.seek(0, 2)  # seek to end
    _file_size = f.tell()
    f.seek(_saved_pos)

    # Read tensor data
    for info in tensor_infos:
        name = info['name']
        dims = info['dims']
        tensor_type = info['type']
        offset = info['offset']

        # Seek to tensor data
        abs_offset = tensor_data_start + offset
        if abs_offset >= _file_size:
            logger.warning(
                "Tensor '%s' offset %d exceeds file size %d — skipping",
                name, abs_offset, _file_size,
            )
            continue
        f.seek(abs_offset)

        # Calculate tensor size with overflow guard
        n_elements = 1
        _MAX_ELEMENTS = 2**32  # ~4 billion elements, ~16 GB at fp32
        for dim in dims:
            if dim <= 0 or dim > _MAX_ELEMENTS:
                logger.warning(
                    "Tensor '%s' has invalid dimension %d — skipping",
                    name, dim,
                )
                n_elements = 0
                break
            n_elements *= dim
            if n_elements > _MAX_ELEMENTS:
                logger.warning(
                    "Tensor '%s' total elements %d exceeds safety "
                    "limit — skipping", name, n_elements,
                )
                n_elements = 0
                break
        if n_elements == 0:
            continue

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
                if len(raw_data) < n_bytes:
                    logger.warning(
                        "Truncated Q4_0 tensor '%s': expected %d bytes, "
                        "got %d", name, n_bytes, len(raw_data))
                    continue
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
                if len(raw_data) < n_bytes:
                    logger.warning(
                        "Truncated Q8_0 tensor '%s': expected %d bytes, "
                        "got %d", name, n_bytes, len(raw_data))
                    continue
                tensor = dequantize_q8_0(raw_data, tuple(dims))
            else:
                logger.warning(f"Skipping quantized tensor (no dequantize): {name}")
                continue
        elif tensor_type in (GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1):
            if not dequantize:
                logger.warning(f"Skipping quantized tensor (no dequantize): {name}")
                continue
            block_size = 32
            if tensor_type == GGML_TYPE_Q4_1:
                bytes_per_block = 20
                dequant_fn = dequantize_q4_1
                type_name = "Q4_1"
            elif tensor_type == GGML_TYPE_Q5_0:
                bytes_per_block = 22
                dequant_fn = dequantize_q5_0
                type_name = "Q5_0"
            else:  # GGML_TYPE_Q5_1
                bytes_per_block = 24
                dequant_fn = dequantize_q5_1
                type_name = "Q5_1"
            n_blocks = (n_elements + block_size - 1) // block_size
            n_bytes = n_blocks * bytes_per_block
            raw_data = f.read(n_bytes)
            if len(raw_data) < n_bytes:
                logger.warning(
                    "Truncated %s tensor '%s': expected %d bytes, got %d",
                    type_name, name, n_bytes, len(raw_data))
                continue
            tensor = dequant_fn(raw_data, tuple(dims))
        elif tensor_type in (GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K):
            # k-quants — super-block of 256 elements
            if not dequantize:
                logger.warning(f"Skipping quantized tensor (no dequantize): {name}")
                continue
            block_size = 256
            if tensor_type == GGML_TYPE_Q2_K:
                bytes_per_block = 84
                dequant_fn = dequantize_q2_K
                type_name = "Q2_K"
            elif tensor_type == GGML_TYPE_Q3_K:
                bytes_per_block = 110
                dequant_fn = dequantize_q3_K
                type_name = "Q3_K"
            elif tensor_type == GGML_TYPE_Q4_K:
                bytes_per_block = 144
                dequant_fn = dequantize_q4_K
                type_name = "Q4_K"
            elif tensor_type == GGML_TYPE_Q5_K:
                bytes_per_block = 176
                dequant_fn = dequantize_q5_K
                type_name = "Q5_K"
            else:  # GGML_TYPE_Q6_K
                bytes_per_block = 210
                dequant_fn = dequantize_q6_K
                type_name = "Q6_K"
            n_blocks = (n_elements + block_size - 1) // block_size
            n_bytes = n_blocks * bytes_per_block
            raw_data = f.read(n_bytes)
            if len(raw_data) < n_bytes:
                logger.warning(
                    "Truncated %s tensor '%s': expected %d bytes, got %d",
                    type_name, name, n_bytes, len(raw_data))
                continue
            tensor = dequant_fn(raw_data, tuple(dims))
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
    _used_fallback = False
    if 'vocab_size' not in config:
        config['vocab_size'] = 32000  # Common default
        _used_fallback = True
    if 'dim' not in config:
        config['dim'] = 4096
        _used_fallback = True
    if 'n_layers' not in config:
        config['n_layers'] = 32
        _used_fallback = True
    if 'n_heads' not in config:
        config['n_heads'] = 32
        _used_fallback = True
    if 'max_seq_len' not in config:
        config['max_seq_len'] = 2048
        _used_fallback = True
    if _used_fallback:
        logger.warning(
            "Using Llama-7B fallback defaults for missing GGUF metadata "
            "— model architecture may not match"
        )

    return config


# ── dequantization ────────────────────────────────────────────────────────


def dequantize_q4_0(data: bytes, shape: tuple) -> 'torch.Tensor':
    """
    Dequantize Q4_0 format (4-bit signed quantization, block size 32).

    Q4_0 format (matches ggml ``block_q4_0`` / ``dequantize_row_q4_0``):
    - Block size: 32 elements
    - Each block: fp16 ``d`` (scale) + 16 bytes ``qs`` (32 x 4-bit values)
    - Total: 18 bytes per block
    - Dequant: ``y = (q - 8) * d`` where ``q`` is unsigned 4-bit (0..15)
    - Layout: byte ``j`` of ``qs`` provides low-nibble for element ``j`` (low half 0..15)
              and high-nibble for element ``j + 16`` (high half 16..31)

    Args:
        data: Raw quantized bytes
        shape: Original tensor shape

    Returns:
        Dequantized PyTorch tensor (float32)
    """
    import numpy as np

    if not HAVE_TORCH:
        raise RuntimeError("PyTorch required for dequantization")

    bytes_per_block = 18  # 2 (scale) + 16 (qs)

    data_array = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(data_array) // bytes_per_block

    if n_blocks == 0:
        return torch.zeros(shape, dtype=torch.float32)

    blocks = data_array[: n_blocks * bytes_per_block].reshape(n_blocks, bytes_per_block)
    d = np.frombuffer(blocks[:, :2].tobytes(), dtype=np.float16).astype(np.float32)  # (n_blocks,)
    qs = blocks[:, 2:18]  # (n_blocks, 16) uint8

    # ggml Q4_0 layout (matches dequantize_row_q4_0 in ggml-quants.c):
    #   byte j holds element j (low nibble) and element j+16 (high nibble).
    # Both nibbles are signed 4-bit values: q - 8.
    low_nib = (qs & 0x0F).astype(np.int16) - 8
    high_nib = ((qs >> 4) & 0x0F).astype(np.int16) - 8

    out = np.empty((n_blocks, 32), dtype=np.float32)
    out[:, :16] = low_nib.astype(np.float32) * d[:, None]
    out[:, 16:] = high_nib.astype(np.float32) * d[:, None]

    flat = out.reshape(-1)
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    flat = flat[:total_elements]
    return torch.from_numpy(flat.reshape(shape))


def dequantize_q8_0(data: bytes, shape: tuple) -> 'torch.Tensor':
    """
    Dequantize Q8_0 format (8-bit quantization, block size 32).

    Q8_0 format (matches ggml ``block_q8_0``):
    - Block size: 32 elements
    - Each block: fp16 ``d`` (scale) + 32 bytes ``qs`` (32 x int8)
    - Total: 34 bytes per block
    - Dequant: ``y = q * d`` where ``q`` is signed int8 (-128..127)
    - Layout: byte j holds element j directly (no nibble splitting,
      no qh expansion — Q8_0 is layout-unambiguous).

    Args:
        data: Raw quantized bytes
        shape: Original tensor shape

    Returns:
        Dequantized PyTorch tensor
    """
    import numpy as np

    if not HAVE_TORCH:
        raise RuntimeError("PyTorch required for dequantization")

    bytes_per_block = 34  # 2 (scale) + 32 (qs)

    raw = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(raw) // bytes_per_block

    total_elements = 1
    for dim in shape:
        total_elements *= dim

    if n_blocks == 0:
        return torch.zeros(shape, dtype=torch.float32)

    raw = raw[:n_blocks * bytes_per_block].reshape(n_blocks, bytes_per_block)

    # Scale: fp16 stored in first 2 bytes of each block.
    d = np.frombuffer(raw[:, 0:2].copy().tobytes(),
                      dtype=np.float16).astype(np.float32)

    # Quantized values: int8 view of bytes 2..34.
    qs = raw[:, 2:34].view(np.int8)  # (n_blocks, 32)

    out = qs.astype(np.float32) * d[:, None]  # (n_blocks, 32)

    flat = out.reshape(-1)[:total_elements]
    return torch.from_numpy(flat.reshape(shape))


def dequantize_q4_1(data: bytes, shape: tuple) -> 'torch.Tensor':
    """
    Dequantize Q4_1 format (4-bit quantization with min, block size 32).

    Q4_1 format (matches ggml ``block_q4_1``):
    - Block size: 32 elements
    - Each block: fp16 ``d`` (scale) + fp16 ``m`` (min) + 16 bytes ``qs`` (32 x 4-bit unsigned)
    - Total: 20 bytes per block
    - Dequant: ``y = q * d + m`` where ``q`` is unsigned 4-bit (0..15)
    - Layout: byte ``j`` contains low nibble for element ``j`` and high nibble for element ``j + 16``

    Args:
        data: Raw quantized bytes
        shape: Original tensor shape

    Returns:
        Dequantized PyTorch tensor (float32)
    """
    import numpy as np

    if not HAVE_TORCH:
        raise RuntimeError("PyTorch required for dequantization")

    block_size = 32
    bytes_per_block = 20  # 2 (d) + 2 (m) + 16 (qs)

    raw = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(raw) // bytes_per_block
    if n_blocks == 0:
        return torch.zeros(shape, dtype=torch.float32)
    raw = raw[: n_blocks * bytes_per_block].reshape(n_blocks, bytes_per_block)

    # fp16 → fp32 (copy() because frombuffer gives a read-only view)
    d = np.frombuffer(raw[:, 0:2].copy().tobytes(), dtype=np.float16).astype(np.float32)
    m = np.frombuffer(raw[:, 2:4].copy().tobytes(), dtype=np.float16).astype(np.float32)
    qs = raw[:, 4:20]  # (n_blocks, 16)

    low = (qs & 0x0F).astype(np.float32)              # elements 0..15
    high = ((qs >> 4) & 0x0F).astype(np.float32)      # elements 16..31

    out = np.empty((n_blocks, block_size), dtype=np.float32)
    out[:, :16] = low * d[:, None] + m[:, None]
    out[:, 16:] = high * d[:, None] + m[:, None]
    out = out.reshape(-1)

    total_elements = 1
    for dim in shape:
        total_elements *= dim
    out = out[:total_elements]
    return torch.from_numpy(out.reshape(shape))


def _expand_qh_bits(qh_u32, n_blocks):
    """Expand per-block uint32 high-bit field into a (n_blocks, 32) uint8 5th-bit array.

    Convention from ggml ``block_q5_0`` / ``block_q5_1``: bit ``i`` of ``qh`` is the
    5th (high) bit of element ``i`` (i = 0..31).
    """
    import numpy as np
    del n_blocks  # shape inferred from qh_u32
    bit_idx = np.arange(32, dtype=np.uint32)
    return ((qh_u32[:, None] >> bit_idx[None, :]) & np.uint32(1)).astype(np.uint8)


def dequantize_q5_0(data: bytes, shape: tuple) -> 'torch.Tensor':
    """
    Dequantize Q5_0 format (5-bit signed quantization, block size 32).

    Q5_0 format (matches ggml ``block_q5_0``):
    - Block size: 32 elements
    - Each block: fp16 ``d`` (scale) + 4 bytes ``qh`` (uint32, 5th bit per element)
                  + 16 bytes ``qs`` (low 4 bits per element)
    - Total: 22 bytes per block
    - Dequant: ``y = (q - 16) * d`` where ``q`` is unsigned 5-bit (0..31)
    - Layout: byte ``j`` of ``qs`` provides low-nibble for element ``j`` (low half)
              and element ``j + 16`` (high half); bit ``i`` of ``qh`` provides 5th bit
              of element ``i``.

    Args:
        data: Raw quantized bytes
        shape: Original tensor shape

    Returns:
        Dequantized PyTorch tensor (float32)
    """
    import numpy as np

    if not HAVE_TORCH:
        raise RuntimeError("PyTorch required for dequantization")

    bytes_per_block = 22  # 2 (d) + 4 (qh) + 16 (qs)

    raw = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(raw) // bytes_per_block
    if n_blocks == 0:
        return torch.zeros(shape, dtype=torch.float32)
    raw = raw[: n_blocks * bytes_per_block].reshape(n_blocks, bytes_per_block)

    d = np.frombuffer(raw[:, 0:2].copy().tobytes(), dtype=np.float16).astype(np.float32)
    qh = np.frombuffer(raw[:, 2:6].copy().tobytes(), dtype=np.uint32)
    qs = raw[:, 6:22]  # (n_blocks, 16)

    fifth = _expand_qh_bits(qh, n_blocks)  # (n_blocks, 32)
    low_nib = (qs & 0x0F).astype(np.uint8)         # (n_blocks, 16) → elements 0..15
    high_nib = ((qs >> 4) & 0x0F).astype(np.uint8)  # (n_blocks, 16) → elements 16..31

    q = np.empty((n_blocks, 32), dtype=np.int16)
    q[:, :16] = low_nib | (fifth[:, :16] << 4)
    q[:, 16:] = high_nib | (fifth[:, 16:] << 4)
    q -= 16  # signed shift: 0..31 → -16..15

    out = q.astype(np.float32) * d[:, None]
    out = out.reshape(-1)

    total_elements = 1
    for dim in shape:
        total_elements *= dim
    out = out[:total_elements]
    return torch.from_numpy(out.reshape(shape))


def dequantize_q5_1(data: bytes, shape: tuple) -> 'torch.Tensor':
    """
    Dequantize Q5_1 format (5-bit quantization with min, block size 32).

    Q5_1 format (matches ggml ``block_q5_1``):
    - Block size: 32 elements
    - Each block: fp16 ``d`` (scale) + fp16 ``m`` (min) + 4 bytes ``qh`` (uint32)
                  + 16 bytes ``qs`` (low nibbles)
    - Total: 24 bytes per block
    - Dequant: ``y = q * d + m`` where ``q`` is unsigned 5-bit (0..31)

    Args:
        data: Raw quantized bytes
        shape: Original tensor shape

    Returns:
        Dequantized PyTorch tensor (float32)
    """
    import numpy as np

    if not HAVE_TORCH:
        raise RuntimeError("PyTorch required for dequantization")

    bytes_per_block = 24  # 2 (d) + 2 (m) + 4 (qh) + 16 (qs)

    raw = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(raw) // bytes_per_block
    if n_blocks == 0:
        return torch.zeros(shape, dtype=torch.float32)
    raw = raw[: n_blocks * bytes_per_block].reshape(n_blocks, bytes_per_block)

    d = np.frombuffer(raw[:, 0:2].copy().tobytes(), dtype=np.float16).astype(np.float32)
    m = np.frombuffer(raw[:, 2:4].copy().tobytes(), dtype=np.float16).astype(np.float32)
    qh = np.frombuffer(raw[:, 4:8].copy().tobytes(), dtype=np.uint32)
    qs = raw[:, 8:24]  # (n_blocks, 16)

    fifth = _expand_qh_bits(qh, n_blocks)  # (n_blocks, 32)
    low_nib = (qs & 0x0F).astype(np.uint8)
    high_nib = ((qs >> 4) & 0x0F).astype(np.uint8)

    q = np.empty((n_blocks, 32), dtype=np.uint8)
    q[:, :16] = low_nib | (fifth[:, :16] << 4)
    q[:, 16:] = high_nib | (fifth[:, 16:] << 4)

    out = q.astype(np.float32) * d[:, None] + m[:, None]
    out = out.reshape(-1)

    total_elements = 1
    for dim in shape:
        total_elements *= dim
    out = out[:total_elements]
    return torch.from_numpy(out.reshape(shape))


# ── k-quant helpers ───────────────────────────────────────────────────────


def _get_scale_min_k4(j: int, scales) -> tuple:
    """Decode the j-th 6-bit (scale, min) pair from a 12-byte ``scales``
    buffer used by Q4_K and Q5_K super-blocks.

    Pure helper (numpy ndarray in, tuple of ndarrays out, no I/O / no GUI)
    so the bit-packing contract is unit-testable in isolation.

    Layout (matches ``get_scale_min_k4`` in ggml-quants.c) — ``scales`` has
    shape ``(n_blocks, 12)`` of ``uint8``. There are 8 (scale, min) pairs
    per super-block; pairs 0..3 are simple low-6-bit reads, pairs 4..7
    are stitched from two source bytes:

        j < 4:  d = scales[:, j]   & 0x3F
                m = scales[:, j+4] & 0x3F
        j ≥ 4:  d = (scales[:, j+4] & 0x0F) | ((scales[:, j-4] >> 6) << 4)
                m = (scales[:, j+4] >>  4)  | ((scales[:, j  ] >> 6) << 4)

    Args:
        j: pair index in ``0..7``.
        scales: ``(n_blocks, 12)`` ``uint8`` ndarray.

    Returns:
        ``(d, m)`` — both ``(n_blocks,)`` ``uint8`` ndarrays.
    """
    import numpy as np
    if j < 4:
        d = scales[:, j] & 0x3F
        m = scales[:, j + 4] & 0x3F
    else:
        d = (scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)
        m = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    # Cast to plain uint8 so callers can do ``d.astype(np.float32)``
    # without surprise upcasts from numpy's broadcasting rules.
    return d.astype(np.uint8), m.astype(np.uint8)


def dequantize_q2_K(data: bytes, shape: tuple) -> 'torch.Tensor':
    """Dequantize Q2_K format (2-bit k-quant, super-block of 256 elements).

    Q2_K layout (matches ggml ``block_q2_K`` /
    ``dequantize_row_q2_K`` in ggml-quants.c):

    - Super-block: 256 elements
    - Each super-block:
        - ``scales[16]`` (each byte = 4-bit ``sc`` low + 4-bit ``mn`` high,
          one byte per 16-element sub-block) — 16 bytes
        - ``qs[64]``     (256 × 2-bit values, packed 4-per-byte across
          two halves of 128 elements) — 64 bytes
        - fp16 ``d``     (super-block scale-of-scales)   — 2 bytes
        - fp16 ``dmin``  (super-block scale-of-mins)     — 2 bytes
    - Total: 84 bytes per super-block.
    - Dequant per 16-element sub-block ``is`` (0..15):
        ``dl = d * (scales[is] & 0xF)``;  ``ml = dmin * (scales[is] >> 4)``
        ``y[k] = dl * q[k] - ml`` where ``q[k]`` is the unsigned 2-bit
        value at position ``k`` within the sub-block.
    - qs layout (256 elements split across 16 sub-blocks of 16 each):
      sub-block ``is`` maps to ``half = is // 8`` (qs region
      ``[half*32 .. half*32+32]``), ``j = (is // 2) % 4`` (shift = ``2*j``),
      ``nibble_half = is % 2`` (which 16-byte half of the 32-byte qs
      region: 0 = bytes ``0..16``, 1 = bytes ``16..32``). The 16 elements
      come from ``(qs_region[byte_off : byte_off+16] >> shift) & 0x03``.

    Args:
        data: Raw quantized bytes.
        shape: Original tensor shape.

    Returns:
        Dequantized ``torch.Tensor`` (float32).
    """
    import numpy as np

    if not HAVE_TORCH:
        raise RuntimeError("PyTorch required for dequantization")

    bytes_per_block = 84  # 16 + 64 + 2 + 2
    super_block_size = 256

    raw = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(raw) // bytes_per_block

    total_elements = 1
    for dim in shape:
        total_elements *= dim

    if n_blocks == 0:
        return torch.zeros(shape, dtype=torch.float32)

    raw = raw[: n_blocks * bytes_per_block].reshape(
        n_blocks, bytes_per_block)

    scales = raw[:, 0:16]                   # (n_blocks, 16) uint8
    qs = raw[:, 16:80]                      # (n_blocks, 64) uint8
    # fp16 d / dmin (.copy() because frombuffer is read-only)
    d = np.frombuffer(raw[:, 80:82].copy().tobytes(),
                      dtype=np.float16).astype(np.float32)
    dmin = np.frombuffer(raw[:, 82:84].copy().tobytes(),
                         dtype=np.float16).astype(np.float32)

    out = np.empty((n_blocks, super_block_size), dtype=np.float32)
    # 16 sub-blocks of 16 elements each; iterate is = 0..15 and place
    # output in sub-block-major order (matches the C reference).
    for is_idx in range(16):
        half = is_idx // 8                   # 0 or 1
        j = (is_idx // 2) % 4                # 0..3 → shift = 0/2/4/6
        nibble_half = is_idx % 2             # 0 = qs[0..16], 1 = qs[16..32]
        shift = 2 * j

        sc = scales[:, is_idx]
        sc_lo = (sc & 0x0F).astype(np.float32)
        mn_hi = (sc >> 4).astype(np.float32)

        qs_off = half * 32 + nibble_half * 16  # absolute offset into qs
        q_byte = qs[:, qs_off:qs_off + 16]     # (n_blocks, 16)
        q_val = ((q_byte >> shift) & 0x03).astype(np.float32)

        out_start = is_idx * 16
        out[:, out_start:out_start + 16] = (
            d[:, None] * sc_lo[:, None] * q_val
            - dmin[:, None] * mn_hi[:, None]
        )

    flat = out.reshape(-1)[:total_elements]
    return torch.from_numpy(flat.reshape(shape))


def dequantize_q3_K(data: bytes, shape: tuple) -> 'torch.Tensor':
    """Dequantize Q3_K format (3-bit k-quant, super-block of 256 elements).

    Q3_K layout (matches ggml ``block_q3_K`` /
    ``dequantize_row_q3_K`` in ggml-quants.c):

    - Super-block: 256 elements
    - Each super-block:
        - ``hmask[32]``  (256 × 1 high-bit values, packed 8-per-byte) — 32 bytes
        - ``qs[64]``     (256 × 2 low-bit values, packed 4-per-byte) — 64 bytes
        - ``scales[12]`` (16 × 6-bit SIGNED scales, bit-packed: each
          scale = 4-bit low nibble from ``scales[0..7]`` plus 2-bit high
          stitch from ``scales[8..11]``) — 12 bytes
        - fp16 ``d``     (super-block scale-of-scales) — 2 bytes
    - Total: 110 bytes per super-block.
    - Dequant per 16-element sub-block ``is`` (0..15):
        ``signed_scale = scale_packed - 32``  (range -32..31)
        ``dl = d * signed_scale``
        For each of 16 elements:
            ``q_low = (qs_byte >> shift) & 0x03``         (2-bit unsigned [0,3])
            ``q_full = q_low - (hm_bit_set ? 0 : 4)``     (3-bit signed [-4,3])
            ``y = dl * q_full``
    - Layout: ``half = is // 8`` (qs region), ``j = (is // 2) % 4``
      (``shift = 2*j``), ``nibble_half = is % 2`` (which 16-byte half of
      the 32-byte qs region AND which 16-byte half of hmask).
    - hmask bit position for sub-block ``is`` is ``is // 2`` — the SAME
      bit is consumed by two consecutive sub-blocks (one for low half of
      hmask, one for high half) before advancing to the next bit.
    - Scale unpacking (6-bit value = 4-bit low + 2-bit high stitch):
        ``low_byte_idx = (is % 4) + 4 * ((is // 4) % 2)``
        ``low_shift   = 4 if is >= 8 else 0``
        ``high_byte_idx = 8 + (is % 4)``
        ``high_shift  = 2 * (is // 4)``
        ``scale = (scales[low_byte_idx] >> low_shift) & 0x0F
                | ((scales[high_byte_idx] >> high_shift) & 0x03) << 4``

    Args:
        data: Raw quantized bytes.
        shape: Original tensor shape.

    Returns:
        Dequantized ``torch.Tensor`` (float32).
    """
    import numpy as np

    if not HAVE_TORCH:
        raise RuntimeError("PyTorch required for dequantization")

    bytes_per_block = 110  # 32 + 64 + 12 + 2
    super_block_size = 256

    raw = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(raw) // bytes_per_block

    total_elements = 1
    for dim in shape:
        total_elements *= dim

    if n_blocks == 0:
        return torch.zeros(shape, dtype=torch.float32)

    raw = raw[: n_blocks * bytes_per_block].reshape(
        n_blocks, bytes_per_block)

    hmask = raw[:, 0:32]                    # (n_blocks, 32)
    qs = raw[:, 32:96]                      # (n_blocks, 64)
    scales_bytes = raw[:, 96:108]           # (n_blocks, 12)
    d = np.frombuffer(raw[:, 108:110].copy().tobytes(),
                      dtype=np.float16).astype(np.float32)

    out = np.empty((n_blocks, super_block_size), dtype=np.float32)

    for is_idx in range(16):
        # 6-bit signed scale unpack.
        low_byte_idx = (is_idx % 4) + 4 * ((is_idx // 4) % 2)
        low_shift = 4 if is_idx >= 8 else 0
        high_byte_idx = 8 + (is_idx % 4)
        high_shift = 2 * (is_idx // 4)

        scale_low = (scales_bytes[:, low_byte_idx] >> low_shift) & 0x0F
        scale_high = (scales_bytes[:, high_byte_idx] >> high_shift) & 0x03
        scale_packed = (scale_low | (scale_high << 4)).astype(np.int32)
        signed_scale = scale_packed - 32                          # [-32, 31]
        dl = d * signed_scale.astype(np.float32)                  # (n_blocks,)

        # Layout into qs / hmask.
        half = is_idx // 8
        j = (is_idx // 2) % 4
        nibble_half = is_idx % 2
        shift = 2 * j

        qs_off = half * 32 + nibble_half * 16
        q_byte = qs[:, qs_off:qs_off + 16]                        # (n_blocks, 16)
        q_low = ((q_byte >> shift) & 0x03).astype(np.int32)

        bit_pos = is_idx // 2                                     # 0..7
        hmask_off = nibble_half * 16
        hm_byte = hmask[:, hmask_off:hmask_off + 16]              # (n_blocks, 16)
        hm_bit = ((hm_byte >> bit_pos) & 0x01).astype(np.int32)
        # hm_bit set → subtract 0; clear → subtract 4 (centering [0,7] → [-4,3]).
        q_full = q_low - (1 - hm_bit) * 4                         # int32 in [-4, 3]

        out_start = is_idx * 16
        out[:, out_start:out_start + 16] = (
            dl[:, None] * q_full.astype(np.float32)
        )

    flat = out.reshape(-1)[:total_elements]
    return torch.from_numpy(flat.reshape(shape))


def dequantize_q4_K(data: bytes, shape: tuple) -> 'torch.Tensor':
    """Dequantize Q4_K format (4-bit k-quant, super-block of 256 elements).

    Q4_K layout (matches ggml ``block_q4_K`` /
    ``dequantize_row_q4_K`` in ggml-quants.c):

    - Super-block: 256 elements
    - Each super-block:
        - fp16 ``d``      (super-block scale-of-scales) — 2 bytes
        - fp16 ``dmin``   (super-block scale-of-mins)   — 2 bytes
        - ``scales[12]``  (8 packed 6-bit (scale, min) pairs) — 12 bytes
        - ``qs[128]``     (256 × 4-bit values, packed 2-per-byte) — 128 bytes
    - Total: 144 bytes per super-block
    - Dequant per 32-element sub-block ``j`` (0..7):
        ``d_j = d * sc_j``;  ``m_j = dmin * mn_j``
        ``y[k] = d_j * q[k] - m_j`` for the 32 elements of the sub-block.
      ``sc_j``, ``mn_j`` come from ``_get_scale_min_k4(j, scales)``.
    - qs layout: byte ``b`` of the first 32 bytes provides the low nibble
      for sub-block 0 element ``b`` and the high nibble for sub-block 1
      element ``b``; bytes 32..63 cover sub-blocks 2/3, etc.

    Args:
        data: Raw quantized bytes.
        shape: Original tensor shape.

    Returns:
        Dequantized ``torch.Tensor`` (float32).
    """
    import numpy as np

    if not HAVE_TORCH:
        raise RuntimeError("PyTorch required for dequantization")

    bytes_per_block = 144  # 2 + 2 + 12 + 128
    super_block_size = 256

    raw = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(raw) // bytes_per_block

    total_elements = 1
    for dim in shape:
        total_elements *= dim

    if n_blocks == 0:
        return torch.zeros(shape, dtype=torch.float32)

    raw = raw[: n_blocks * bytes_per_block].reshape(
        n_blocks, bytes_per_block)

    # fp16 scales (.copy() because frombuffer is read-only)
    d = np.frombuffer(raw[:, 0:2].copy().tobytes(),
                      dtype=np.float16).astype(np.float32)
    dmin = np.frombuffer(raw[:, 2:4].copy().tobytes(),
                         dtype=np.float16).astype(np.float32)
    scales = raw[:, 4:16]   # (n_blocks, 12) uint8 — 8 packed (sc, mn) pairs
    qs = raw[:, 16:144]     # (n_blocks, 128) uint8 — 256 × 4-bit values

    out = np.empty((n_blocks, super_block_size), dtype=np.float32)
    # 8 sub-blocks of 32 elements each: pairs (0,1), (2,3), (4,5), (6,7)
    # share a 32-byte qs slice (low nibble = even sub-block, high nibble
    # = odd sub-block).
    for pair in range(4):
        sub_lo = 2 * pair          # even sub-block index
        sub_hi = sub_lo + 1
        sc_lo, mn_lo = _get_scale_min_k4(sub_lo, scales)
        sc_hi, mn_hi = _get_scale_min_k4(sub_hi, scales)
        d_lo = d * sc_lo.astype(np.float32)
        m_lo = dmin * mn_lo.astype(np.float32)
        d_hi = d * sc_hi.astype(np.float32)
        m_hi = dmin * mn_hi.astype(np.float32)

        qs_slice = qs[:, pair * 32:(pair + 1) * 32]  # (n_blocks, 32)
        q_lo = (qs_slice & 0x0F).astype(np.float32)
        q_hi = ((qs_slice >> 4) & 0x0F).astype(np.float32)

        out[:, sub_lo * 32:(sub_lo + 1) * 32] = (
            d_lo[:, None] * q_lo - m_lo[:, None])
        out[:, sub_hi * 32:(sub_hi + 1) * 32] = (
            d_hi[:, None] * q_hi - m_hi[:, None])

    flat = out.reshape(-1)[:total_elements]
    return torch.from_numpy(flat.reshape(shape))


def dequantize_q5_K(data: bytes, shape: tuple) -> 'torch.Tensor':
    """Dequantize Q5_K format (5-bit k-quant, super-block of 256 elements).

    Q5_K layout (matches ggml ``block_q5_K`` /
    ``dequantize_row_q5_K`` in ggml-quants.c):

    - Super-block: 256 elements
    - Each super-block:
        - fp16 ``d``      (super-block scale-of-scales) — 2 bytes
        - fp16 ``dmin``   (super-block scale-of-mins)   — 2 bytes
        - ``scales[12]``  (8 packed 6-bit (scale, min) pairs) — 12 bytes
        - ``qh[32]``      (one 5th-bit per element, 256 bits) — 32 bytes
        - ``qs[128]``     (256 × 4-bit low nibbles, packed 2-per-byte) — 128 bytes
    - Total: 176 bytes per super-block
    - 5-bit unsigned quant: ``q = (ql_nibble | (qh_bit << 4))`` in ``[0, 31]``
    - Dequant per 32-element sub-block: ``y = d_j * q - m_j`` where
      ``d_j = d * sc_j`` and ``m_j = dmin * mn_j`` come from
      ``_get_scale_min_k4(j, scales)`` (same packed-scales format as Q4_K).
    - Layout: 4 outer iterations covering ``j = 0, 64, 128, 192``. Each
      iteration consumes 32 qs bytes (low nibbles → 32 outputs at sub-block
      ``2*pair``, high nibbles → 32 outputs at sub-block ``2*pair+1``).
      The SAME 32 qh bytes are reused across all 4 iterations — bit
      ``2*pair`` of ``qh[l]`` is the 5th bit of the low-nibble path,
      bit ``2*pair+1`` of ``qh[l]`` is the 5th bit of the high-nibble path.
      So all 8 bits of every qh byte are consumed across the 8 sub-blocks.

    Args:
        data: Raw quantized bytes.
        shape: Original tensor shape.

    Returns:
        Dequantized ``torch.Tensor`` (float32).
    """
    import numpy as np

    if not HAVE_TORCH:
        raise RuntimeError("PyTorch required for dequantization")

    bytes_per_block = 176  # 2 + 2 + 12 + 32 + 128
    super_block_size = 256

    raw = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(raw) // bytes_per_block

    total_elements = 1
    for dim in shape:
        total_elements *= dim

    if n_blocks == 0:
        return torch.zeros(shape, dtype=torch.float32)

    raw = raw[: n_blocks * bytes_per_block].reshape(
        n_blocks, bytes_per_block)

    d = np.frombuffer(raw[:, 0:2].copy().tobytes(),
                      dtype=np.float16).astype(np.float32)
    dmin = np.frombuffer(raw[:, 2:4].copy().tobytes(),
                         dtype=np.float16).astype(np.float32)
    scales = raw[:, 4:16]    # (n_blocks, 12) uint8 — 8 packed (sc, mn) pairs
    qh = raw[:, 16:48]       # (n_blocks, 32) uint8 — 5th bits, reused 4× below
    qs = raw[:, 48:176]      # (n_blocks, 128) uint8 — 256 × 4-bit low nibbles

    out = np.empty((n_blocks, super_block_size), dtype=np.float32)
    for pair in range(4):
        bit_lo = pair * 2          # qh bit for the low-nibble path
        bit_hi = pair * 2 + 1      # qh bit for the high-nibble path
        sub_lo = 2 * pair          # even sub-block index
        sub_hi = sub_lo + 1
        sc_lo, mn_lo = _get_scale_min_k4(sub_lo, scales)
        sc_hi, mn_hi = _get_scale_min_k4(sub_hi, scales)
        d_lo = d * sc_lo.astype(np.float32)
        m_lo = dmin * mn_lo.astype(np.float32)
        d_hi = d * sc_hi.astype(np.float32)
        m_hi = dmin * mn_hi.astype(np.float32)

        qs_slice = qs[:, pair * 32:(pair + 1) * 32]  # (n_blocks, 32)
        # 5th-bit contribution: 0 or 16 added to the 4-bit base
        bit_lo_val = (((qh >> bit_lo) & 0x01).astype(np.float32)) * 16.0
        bit_hi_val = (((qh >> bit_hi) & 0x01).astype(np.float32)) * 16.0

        q_lo = (qs_slice & 0x0F).astype(np.float32) + bit_lo_val
        q_hi = ((qs_slice >> 4) & 0x0F).astype(np.float32) + bit_hi_val

        out[:, sub_lo * 32:(sub_lo + 1) * 32] = (
            d_lo[:, None] * q_lo - m_lo[:, None])
        out[:, sub_hi * 32:(sub_hi + 1) * 32] = (
            d_hi[:, None] * q_hi - m_hi[:, None])

    flat = out.reshape(-1)[:total_elements]
    return torch.from_numpy(flat.reshape(shape))


def dequantize_q6_K(data: bytes, shape: tuple) -> 'torch.Tensor':
    """Dequantize Q6_K format (6-bit k-quant, super-block of 256 elements).

    Q6_K layout (matches ggml ``block_q6_K`` /
    ``dequantize_row_q6_K`` in ggml-quants.c):

    - Super-block: 256 elements
    - Each super-block:
        - ``ql[128]``  (low 4 bits of every quant)              — 128 bytes
        - ``qh[64]``   (high 2 bits of every quant)             —  64 bytes
        - ``scales[16]`` (16 ``int8`` per-sub-block scales)      —  16 bytes
        - fp16 ``d``     (super-block scale-of-scales)          —   2 bytes
    - Total: 210 bytes per super-block
    - 6-bit quant is signed in [-32, 31]: ``q = (ql_nibble | (qh_bits << 4)) - 32``
    - Dequant: ``y = d * scale_j * q`` where ``scale_j`` is the int8
      per-sub-block scale (16 sub-blocks of 16 elements each).
    - Layout (per super-block, looping ``j = 0, 128`` in halves of 128
      output elements):
        for l in 0..32:                   # is = l // 16  → 0 or 1
            q1 = (ql[l]      & 0xF)       | ((qh[l] >> 0) & 3) << 4) - 32
            q2 = (ql[l + 32] & 0xF)       | ((qh[l] >> 2) & 3) << 4) - 32
            q3 = (ql[l]      >> 4)        | ((qh[l] >> 4) & 3) << 4) - 32
            q4 = (ql[l + 32] >> 4)        | ((qh[l] >> 6) & 3) << 4) - 32
            y[l +   0] = d * scales[is + 0] * q1   # scales[0] then scales[1]
            y[l +  32] = d * scales[is + 2] * q2   # scales[2] then scales[3]
            y[l +  64] = d * scales[is + 4] * q3   # scales[4] then scales[5]
            y[l +  96] = d * scales[is + 6] * q4   # scales[6] then scales[7]
        ql += 64; qh += 32; scales += 8

    Args:
        data: Raw quantized bytes.
        shape: Original tensor shape.

    Returns:
        Dequantized ``torch.Tensor`` (float32).
    """
    import numpy as np

    if not HAVE_TORCH:
        raise RuntimeError("PyTorch required for dequantization")

    bytes_per_block = 210  # 128 + 64 + 16 + 2
    super_block_size = 256

    raw = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(raw) // bytes_per_block

    total_elements = 1
    for dim in shape:
        total_elements *= dim

    if n_blocks == 0:
        return torch.zeros(shape, dtype=torch.float32)

    raw = raw[: n_blocks * bytes_per_block].reshape(
        n_blocks, bytes_per_block)

    ql = raw[:, 0:128]                                  # (n_blocks, 128)
    qh = raw[:, 128:192]                                # (n_blocks, 64)
    scales = raw[:, 192:208].view(np.int8)              # (n_blocks, 16) signed
    d = np.frombuffer(raw[:, 208:210].copy().tobytes(),
                      dtype=np.float16).astype(np.float32)  # (n_blocks,)

    out = np.empty((n_blocks, super_block_size), dtype=np.float32)
    # Two halves of 128 output elements each. Each half consumes 64 ql
    # bytes, 32 qh bytes, and 8 of the 16 scales (scales[half*8:(half+1)*8]).
    for half in range(2):
        ql_half = ql[:, half * 64:(half + 1) * 64]       # (n_blocks, 64)
        qh_half = qh[:, half * 32:(half + 1) * 32]       # (n_blocks, 32)
        sc = scales[:, half * 8:(half + 1) * 8].astype(np.float32)
        base = half * 128

        # ql_a covers elements 0..31 (l=0..31) of the four output regions
        # via low+high nibbles of the same 32 bytes; ql_b is the same
        # for elements 32..63 of the regions.
        ql_a = ql_half[:, :32]    # (n_blocks, 32)
        ql_b = ql_half[:, 32:64]  # (n_blocks, 32)

        # qh bits (0,1) → q1, (2,3) → q2, (4,5) → q3, (6,7) → q4.
        qh_b0 = (qh_half >> 0) & 0x03
        qh_b1 = (qh_half >> 2) & 0x03
        qh_b2 = (qh_half >> 4) & 0x03
        qh_b3 = (qh_half >> 6) & 0x03

        # Signed 6-bit values: (low4 | (high2 << 4)) - 32
        q1 = ((ql_a & 0x0F) | (qh_b0 << 4)).astype(np.int16) - 32  # (n_blocks, 32)
        q2 = ((ql_b & 0x0F) | (qh_b1 << 4)).astype(np.int16) - 32
        q3 = ((ql_a >> 4)   | (qh_b2 << 4)).astype(np.int16) - 32
        q4 = ((ql_b >> 4)   | (qh_b3 << 4)).astype(np.int16) - 32

        # Per-region scale array — l=0..15 uses sc[is+0], l=16..31 uses
        # sc[is+1] (and likewise for is+2/+3, +4/+5, +6/+7). Build a
        # (n_blocks, 32) mixer once and broadcast over q1..q4.
        s_q1 = np.empty((n_blocks, 32), dtype=np.float32)
        s_q1[:, :16] = sc[:, 0:1]
        s_q1[:, 16:] = sc[:, 1:2]
        s_q2 = np.empty((n_blocks, 32), dtype=np.float32)
        s_q2[:, :16] = sc[:, 2:3]
        s_q2[:, 16:] = sc[:, 3:4]
        s_q3 = np.empty((n_blocks, 32), dtype=np.float32)
        s_q3[:, :16] = sc[:, 4:5]
        s_q3[:, 16:] = sc[:, 5:6]
        s_q4 = np.empty((n_blocks, 32), dtype=np.float32)
        s_q4[:, :16] = sc[:, 6:7]
        s_q4[:, 16:] = sc[:, 7:8]

        out[:, base + 0:base + 32] = (
            d[:, None] * s_q1 * q1.astype(np.float32))
        out[:, base + 32:base + 64] = (
            d[:, None] * s_q2 * q2.astype(np.float32))
        out[:, base + 64:base + 96] = (
            d[:, None] * s_q3 * q3.astype(np.float32))
        out[:, base + 96:base + 128] = (
            d[:, None] * s_q4 * q4.astype(np.float32))

    flat = out.reshape(-1)[:total_elements]
    return torch.from_numpy(flat.reshape(shape))
