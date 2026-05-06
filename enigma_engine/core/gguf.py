"""
GGUF Export Module for Enigma AI Engine

Unified GGUF file creation for llama.cpp compatibility.
Supports multiple quantization types and metadata embedding.

Usage:
    from enigma_engine.core.gguf import export_to_gguf, GGUFExporter

    # Simple export
    export_to_gguf(model, tokenizer, "model.gguf", quant_type="Q4_K_M")

    # Advanced export
    exporter = GGUFExporter(quantization="q4_0")
    exporter.export(model, "model.gguf", metadata, tokenizer)

Note: This module consolidates gguf_export.py and gguf_exporter.py
"""

import logging
import re
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

logger = logging.getLogger(__name__)

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


# =============================================================================
# GGUF Constants
# =============================================================================

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3


# =============================================================================
# Enums
# =============================================================================

class GGMLType(IntEnum):
    """GGML tensor data types."""
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
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    BF16 = 29


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


# Alias for compatibility
GGUFMetadataType = GGUFValueType


# =============================================================================
# Quantization Mappings
# =============================================================================

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


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GGUFTensor:
    """Tensor to be written to GGUF file."""
    name: str
    data: Any  # np.ndarray
    type: GGMLType = GGMLType.F32
    n_dims: int = 0
    shape: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.n_dims == 0 and self.shape:
            self.n_dims = len(self.shape)


@dataclass
class GGUFMetadata:
    """GGUF file metadata."""
    general_architecture: str = "llama"
    general_name: str = "forge-ai"
    general_author: str = "Enigma AI Engine"
    general_version: str = "1.0"
    general_description: str = ""
    general_license: str = "Apache-2.0"
    general_file_type: int = 1  # F16

    # Model architecture
    context_length: int = 2048
    embedding_length: int = 4096
    block_count: int = 32
    feed_forward_length: int = 11008
    attention_head_count: int = 32
    attention_head_count_kv: int = 32
    rope_dimension_count: int = 128
    rope_freq_base: float = 10000.0
    # RMSNorm epsilon — REQUIRED by llama.cpp for the "llama" arch.
    # Loading without this key fails with:
    #   error loading model hyperparameters: key not found in model:
    #   llama.attention.layer_norm_rms_epsilon
    attention_layer_norm_rms_epsilon: float = 1e-6

    # Tokenizer
    tokenizer_model: str = "llama"
    vocab_size: int = 32000
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        arch = self.general_architecture
        return {
            "general.architecture": arch,
            "general.name": self.general_name,
            "general.author": self.general_author,
            "general.version": self.general_version,
            "general.description": self.general_description,
            "general.license": self.general_license,
            "general.file_type": self.general_file_type,
            f"{arch}.context_length": self.context_length,
            f"{arch}.embedding_length": self.embedding_length,
            f"{arch}.block_count": self.block_count,
            f"{arch}.feed_forward_length": self.feed_forward_length,
            f"{arch}.attention.head_count": self.attention_head_count,
            f"{arch}.attention.head_count_kv": self.attention_head_count_kv,
            f"{arch}.rope.dimension_count": self.rope_dimension_count,
            f"{arch}.rope.freq_base": self.rope_freq_base,
            f"{arch}.attention.layer_norm_rms_epsilon": self.attention_layer_norm_rms_epsilon,
            f"{arch}.vocab_size": self.vocab_size,
            "tokenizer.ggml.model": self.tokenizer_model,
            "tokenizer.ggml.bos_token_id": self.bos_token_id,
            "tokenizer.ggml.eos_token_id": self.eos_token_id,
            "tokenizer.ggml.padding_token_id": self.pad_token_id,
        }


# llama.cpp / GGUF spec: many integer keys are UINT32, not UINT64. Reading
# them as the wrong type triggers `GGML_ASSERT(ctx->kv[key_id].type ==
# GGUF_TYPE_UINT32) failed` in `ggml.c` and aborts the host process.
# Reference: llama.cpp's src/llama-arch.cpp (LLM_KV_* table).
#
# Architecture-prefixed keys (e.g. "llama.context_length") are stored
# without the prefix; the lookup adds the prefix at call time. Plain
# top-level keys (e.g. "general.file_type") are stored as-is.
_GGUF_UINT32_ARCH_SUFFIXES: tuple[str, ...] = (
    ".context_length",
    ".embedding_length",
    ".block_count",
    ".feed_forward_length",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".rope.dimension_count",
    ".vocab_size",
    ".expert_count",
    ".expert_used_count",
)
_GGUF_UINT32_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "general.file_type",
    "general.quantization_version",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.separator_token_id",
})
_GGUF_FLOAT32_ARCH_SUFFIXES: tuple[str, ...] = (
    ".rope.freq_base",
    ".rope.scaling.factor",
    ".attention.layer_norm_epsilon",
    ".attention.layer_norm_rms_epsilon",
)


def _gguf_scalar_type(key: str) -> Optional["GGUFValueType"]:
    """Return the GGUFValueType the spec requires for `key`, or None if
    the value type can be inferred from the Python value."""
    if key in _GGUF_UINT32_TOP_LEVEL_KEYS:
        return GGUFValueType.UINT32
    # Architecture-prefixed: split on first '.'.
    if "." in key:
        suffix = key[key.index("."):]
        for s in _GGUF_UINT32_ARCH_SUFFIXES:
            if suffix == s:
                return GGUFValueType.UINT32
        for s in _GGUF_FLOAT32_ARCH_SUFFIXES:
            if suffix == s:
                return GGUFValueType.FLOAT32
    return None


@dataclass
class _GGUFTypedArray:
    """Internal marker for an array value with an explicit GGUF element type.

    Used by `GGUFWriter.add_metadata_array` to override the writer's
    Python-type inference (which would otherwise pick INT64 / FLOAT32
    by default and produce arrays llama.cpp refuses).
    """
    values: Any
    element_type: "GGUFValueType"


# =============================================================================
# Weight Name Conversion
# =============================================================================
#
# ARCH-V1b (May 6, 2026): Replaced naive str.replace dict with an ordered
# regex pipeline. The previous dict had two structural bugs:
#   1. It only knew HF-style names (q_proj/gate_proj/mlp/self_attn) but
#      Enigma's state_dict is Llama-style (attention.wq, feed_forward.w1).
#      Every weight tensor landed under the wrong GGUF name and llama.cpp
#      hard-aborted on load.
#   2. str.replace substring-collisions: `norm -> output_norm` rewrote
#      the inner `norm` of `attention_norm`, producing `attn_output_norm`.
#
# Patterns are tried in order; first match wins. Each pattern uses a
# single-pass `re.sub` so substrings can't collide with later rules.
# Reference: https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp
# (LLM_TENSOR_NAMES for the "llama" architecture).

# Ordered list of (regex, replacement) for Llama-style state dicts.
# Regex flags: ASCII match, no IGNORECASE — tensor names are case-sensitive.
_TENSOR_NAME_RULES: list[tuple[str, str]] = [
    # --- per-block tensors (Llama-style: layers.{i}....) ---
    # Attention projections (must match BEFORE the bare `attention.` rule).
    (r'^layers\.(\d+)\.attention\.wq\.weight$',     r'blk.\1.attn_q.weight'),
    (r'^layers\.(\d+)\.attention\.wk\.weight$',     r'blk.\1.attn_k.weight'),
    (r'^layers\.(\d+)\.attention\.wv\.weight$',     r'blk.\1.attn_v.weight'),
    (r'^layers\.(\d+)\.attention\.wo\.weight$',     r'blk.\1.attn_output.weight'),
    # QK-norm (used by Qwen3-style models; Enigma has it).
    (r'^layers\.(\d+)\.attention\.q_norm\.weight$', r'blk.\1.attn_q_norm.weight'),
    (r'^layers\.(\d+)\.attention\.k_norm\.weight$', r'blk.\1.attn_k_norm.weight'),
    # FFN projections (Llama: w1 = gate, w2 = down, w3 = up).
    (r'^layers\.(\d+)\.feed_forward\.w1\.weight$',  r'blk.\1.ffn_gate.weight'),
    (r'^layers\.(\d+)\.feed_forward\.w2\.weight$',  r'blk.\1.ffn_down.weight'),
    (r'^layers\.(\d+)\.feed_forward\.w3\.weight$',  r'blk.\1.ffn_up.weight'),
    # Norms — `attention_norm` MUST match before any `norm` rule.
    (r'^layers\.(\d+)\.attention_norm\.weight$',    r'blk.\1.attn_norm.weight'),
    (r'^layers\.(\d+)\.ffn_norm\.weight$',          r'blk.\1.ffn_norm.weight'),

    # --- HF-style fallback (kept so HF state-dicts still convert) ---
    (r'^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$',  r'blk.\1.attn_q.weight'),
    (r'^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$',  r'blk.\1.attn_k.weight'),
    (r'^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$',  r'blk.\1.attn_v.weight'),
    (r'^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$',  r'blk.\1.attn_output.weight'),
    (r'^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$',     r'blk.\1.ffn_gate.weight'),
    (r'^model\.layers\.(\d+)\.mlp\.down_proj\.weight$',     r'blk.\1.ffn_down.weight'),
    (r'^model\.layers\.(\d+)\.mlp\.up_proj\.weight$',       r'blk.\1.ffn_up.weight'),
    (r'^model\.layers\.(\d+)\.input_layernorm\.weight$',    r'blk.\1.attn_norm.weight'),
    (r'^model\.layers\.(\d+)\.post_attention_layernorm\.weight$',
                                                            r'blk.\1.ffn_norm.weight'),

    # --- top-level tensors ---
    (r'^tok_embeddings\.weight$',       r'token_embd.weight'),
    (r'^model\.embed_tokens\.weight$',  r'token_embd.weight'),
    (r'^embedding\.weight$',            r'token_embd.weight'),
    (r'^lm_head\.weight$',              r'output.weight'),
    (r'^output\.weight$',               r'output.weight'),
    (r'^model\.norm\.weight$',          r'output_norm.weight'),
    (r'^norm\.weight$',                 r'output_norm.weight'),
    (r'^ln_f\.weight$',                 r'output_norm.weight'),
]

# Pre-compile for the hot path (one re.compile per export, not per tensor).
_TENSOR_NAME_RULES_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pat), repl) for pat, repl in _TENSOR_NAME_RULES
]

# Legacy dict — kept as a back-compat re-export. NOT used by
# convert_tensor_name anymore; consumers reading this dict would have
# inherited the substring-collision bug. New code should rely on
# convert_tensor_name (regex pipeline) instead.
WEIGHT_NAME_MAP = {
    'tok_embeddings': 'token_embd',
    'embed_tokens': 'token_embd',
    'embedding': 'token_embd',
    'lm_head': 'output',
    'output': 'output',
    'norm': 'output_norm',
    'ln_f': 'output_norm',
    'layers': 'blk',
    'attention.wq': 'attn_q',
    'attention.wk': 'attn_k',
    'attention.wv': 'attn_v',
    'attention.wo': 'attn_output',
    'attention.q_norm': 'attn_q_norm',
    'attention.k_norm': 'attn_k_norm',
    'feed_forward.w1': 'ffn_gate',
    'feed_forward.w2': 'ffn_down',
    'feed_forward.w3': 'ffn_up',
    'attention_norm': 'attn_norm',
    'ffn_norm': 'ffn_norm',
    # HF-style aliases
    'self_attn.q_proj': 'attn_q',
    'self_attn.k_proj': 'attn_k',
    'self_attn.v_proj': 'attn_v',
    'self_attn.o_proj': 'attn_output',
    'mlp.gate_proj': 'ffn_gate',
    'mlp.up_proj': 'ffn_up',
    'mlp.down_proj': 'ffn_down',
    'input_layernorm': 'attn_norm',
    'post_attention_layernorm': 'ffn_norm',
}


# =============================================================================
# GGUF Reading / Parsing  (shared by gguf_loader.py & ollama_loader.py)
# =============================================================================

def read_gguf_value(f: BinaryIO, value_type: int) -> Any:
    """
    Read a single GGUF metadata value from an open file handle.

    Args:
        f: Open file handle (binary mode), positioned at the value bytes.
        value_type: One of :class:`GGUFValueType` integer codes.

    Returns:
        The decoded Python value, or *None* for unknown types.
    """
    vt = GGUFValueType
    if value_type == vt.UINT8:
        return struct.unpack('<B', f.read(1))[0]
    if value_type == vt.INT8:
        return struct.unpack('<b', f.read(1))[0]
    if value_type == vt.UINT16:
        return struct.unpack('<H', f.read(2))[0]
    if value_type == vt.INT16:
        return struct.unpack('<h', f.read(2))[0]
    if value_type == vt.UINT32:
        return struct.unpack('<I', f.read(4))[0]
    if value_type == vt.INT32:
        return struct.unpack('<i', f.read(4))[0]
    if value_type == vt.FLOAT32:
        return struct.unpack('<f', f.read(4))[0]
    if value_type == vt.BOOL:
        return struct.unpack('<B', f.read(1))[0] != 0
    if value_type == vt.STRING:
        str_len = struct.unpack('<Q', f.read(8))[0]
        if str_len > 100_000_000:
            raise ValueError(f"GGUF string length {str_len} exceeds 100 MB limit")
        return f.read(str_len).decode('utf-8', errors='replace')
    if value_type == vt.ARRAY:
        array_type = struct.unpack('<I', f.read(4))[0]
        array_len = struct.unpack('<Q', f.read(8))[0]
        if array_len > 1_000_000:
            raise ValueError(f"GGUF array length {array_len} exceeds 1M element limit")
        return [read_gguf_value(f, array_type) for _ in range(array_len)]
    if value_type == vt.UINT64:
        return struct.unpack('<Q', f.read(8))[0]
    if value_type == vt.INT64:
        return struct.unpack('<q', f.read(8))[0]
    if value_type == vt.FLOAT64:
        return struct.unpack('<d', f.read(8))[0]
    logger.warning("Unknown GGUF value type %d", value_type)
    return None


def parse_gguf_header(f: BinaryIO) -> dict[str, Any]:
    """
    Parse a GGUF file header.

    Returns dict with keys *version*, *tensor_count*, *metadata_kv_count*.

    Raises:
        ValueError: If the file does not start with the GGUF magic bytes.
    """
    magic = f.read(4)
    if magic != b'GGUF':
        raise ValueError(f"Not a valid GGUF file (magic: {magic!r})")
    version = struct.unpack('<I', f.read(4))[0]
    tensor_count = struct.unpack('<Q', f.read(8))[0]
    metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
    logger.debug(
        "GGUF header: version=%d, tensors=%d, metadata_kvs=%d",
        version, tensor_count, metadata_kv_count,
    )
    return {
        'version': version,
        'tensor_count': tensor_count,
        'metadata_kv_count': metadata_kv_count,
    }


def parse_gguf_metadata(f: BinaryIO, header: dict[str, Any]) -> dict[str, Any]:
    """
    Parse all metadata key-value pairs that follow the header.

    Args:
        f: Open file handle positioned right after the header.
        header: Dict returned by :func:`parse_gguf_header`.

    Returns:
        ``{key: value}`` mapping for every metadata entry.
    """
    metadata: dict[str, Any] = {}
    for _ in range(header['metadata_kv_count']):
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8', errors='replace')
        value_type = struct.unpack('<I', f.read(4))[0]
        try:
            value = read_gguf_value(f, value_type)
        except Exception as exc:
            logger.warning("Failed to parse metadata value for %s: %s", key, exc)
            value = None
        metadata[key] = value
        logger.debug("Metadata: %s = %s", key, value)
    return metadata


# =============================================================================
# Tensor Name Mapping
# =============================================================================

def convert_tensor_name(pytorch_name: str) -> str:
    """Convert a PyTorch state_dict tensor name to GGUF / llama.cpp form.

    Tries each rule in `_TENSOR_NAME_RULES` in order; the first regex
    that fully matches wins. Returns the input unchanged if no rule
    matches (so unknown names propagate to llama.cpp where they will
    fail loudly, rather than being silently mangled).

    Examples
    --------
    >>> convert_tensor_name('layers.0.attention.wq.weight')
    'blk.0.attn_q.weight'
    >>> convert_tensor_name('layers.0.attention_norm.weight')
    'blk.0.attn_norm.weight'
    >>> convert_tensor_name('tok_embeddings.weight')
    'token_embd.weight'
    """
    for pattern, replacement in _TENSOR_NAME_RULES_COMPILED:
        new, n = pattern.subn(replacement, pytorch_name)
        if n:
            return new
    return pytorch_name


# =============================================================================
# Numpy-dependent implementations
# =============================================================================

if HAS_NUMPY:

    class GGUFQuantizer:
        """Quantize tensors for GGUF format."""

        BLOCK_SIZE = 32

        @classmethod
        def quantize_q4_0(cls, data: np.ndarray) -> tuple[np.ndarray, GGMLType]:
            """Quantize to Q4_0 format (4-bit with block-wise scaling)."""
            data = data.flatten().astype(np.float32)
            block_size = cls.BLOCK_SIZE

            # Pad to block size
            n = len(data)
            padded_n = ((n + block_size - 1) // block_size) * block_size
            if padded_n > n:
                data = np.pad(data, (0, padded_n - n))

            data = data.reshape(-1, block_size)
            n_blocks = len(data)

            # Compute scales
            max_vals = np.abs(data).max(axis=1)
            scales = np.where(max_vals > 0, max_vals / 7.0, 1.0)

            # Quantize
            quantized = np.round(data / scales[:, None]).astype(np.int8)
            quantized = np.clip(quantized, -8, 7)

            # Pack into 4-bit pairs
            packed = np.zeros((n_blocks, block_size // 2), dtype=np.uint8)
            for i in range(block_size // 2):
                low = (quantized[:, 2*i] + 8).astype(np.uint8)
                high = (quantized[:, 2*i + 1] + 8).astype(np.uint8)
                packed[:, i] = low | (high << 4)

            # Create output: [scale (fp16), packed data]
            scales_fp16 = scales.astype(np.float16)
            output = np.zeros(n_blocks * (2 + block_size // 2), dtype=np.uint8)

            for i in range(n_blocks):
                offset = i * (2 + block_size // 2)
                output[offset:offset+2] = scales_fp16[i].view(np.uint8)
                output[offset+2:offset+2+block_size//2] = packed[i]

            return output, GGMLType.Q4_0

        @classmethod
        def quantize_q8_0(cls, data: np.ndarray) -> tuple[np.ndarray, GGMLType]:
            """Quantize to Q8_0 format (8-bit with block-wise scaling)."""
            data = data.flatten().astype(np.float32)
            block_size = cls.BLOCK_SIZE

            # Pad to block size
            n = len(data)
            padded_n = ((n + block_size - 1) // block_size) * block_size
            if padded_n > n:
                data = np.pad(data, (0, padded_n - n))

            data = data.reshape(-1, block_size)
            n_blocks = len(data)

            # Compute scales
            max_vals = np.abs(data).max(axis=1)
            scales = np.where(max_vals > 0, max_vals / 127.0, 1.0)

            # Quantize
            quantized = np.round(data / scales[:, None]).astype(np.int8)
            quantized = np.clip(quantized, -127, 127)

            # Create output: [scale (fp16), quantized data]
            scales_fp16 = scales.astype(np.float16)
            output = np.zeros(n_blocks * (2 + block_size), dtype=np.uint8)

            for i in range(n_blocks):
                offset = i * (2 + block_size)
                output[offset:offset+2] = scales_fp16[i].view(np.uint8)
                output[offset+2:offset+2+block_size] = quantized[i].view(np.uint8)

            return output, GGMLType.Q8_0

        @classmethod
        def quantize_q4_k(cls, data: np.ndarray) -> tuple[np.ndarray, GGMLType]:
            """Quantize to Q4_K format (super blocks with multiple scales)."""
            data = data.flatten().astype(np.float32)
            block_size = 256

            # Pad to block size
            n = len(data)
            padded_n = ((n + block_size - 1) // block_size) * block_size
            if padded_n > n:
                data = np.pad(data, (0, padded_n - n))

            n_blocks = len(data) // block_size
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
                    scale = (sub_max - sub_min) / 15.0 if sub_max != sub_min else 1.0
                    sub_scales.append(scale)
                    sub_mins.append(sub_min)

                # Super block scale and min
                d = max(sub_scales) if max(sub_scales) > 0 else 1.0
                dmin = min(sub_mins)

                # Write header
                output.extend(np.float16(d).tobytes())
                output.extend(np.float16(dmin).tobytes())

                # Pack sub-block scales/mins
                for s in sub_scales:
                    output.append(int(np.round(s / d * 63)) & 0x3F)
                for m in sub_mins:
                    output.append(int(np.round((m - dmin) / d * 63)) & 0x3F)

                # Quantize data
                packed = bytearray(128)
                for j in range(8):
                    sub_block = block[j * 32:(j + 1) * 32]
                    scale = sub_scales[j]
                    min_val = sub_mins[j]

                    for k in range(16):
                        low = int(np.round((sub_block[k * 2] - min_val) / scale)) & 0xF
                        high = int(np.round((sub_block[k * 2 + 1] - min_val) / scale)) & 0xF
                        packed[j * 16 + k] = low | (high << 4)

                output.extend(packed)

            return np.frombuffer(bytes(output), dtype=np.uint8), GGMLType.Q4_K


    class GGUFWriter:
        """Write GGUF files compatible with llama.cpp."""

        def __init__(self, filepath: Union[str, Path]) -> None:
            self.filepath = Path(filepath)
            self.metadata: dict[str, Any] = {}
            self.tensors: list[GGUFTensor] = []
            self._file: Optional[BinaryIO] = None

        def add_metadata(self, key: str, value: Any) -> None:
            """Add metadata key-value pair."""
            self.metadata[key] = value

        def add_metadata_array(
            self,
            key: str,
            values: Any,
            elem_type: "GGUFValueType",
        ) -> None:
            """Add an array-valued metadata entry with explicit element type.

            Use this when the GGUF spec requires a specific array element
            type (e.g. `tokenizer.ggml.token_type` is INT32, not the
            INT64 that bare `int` lists would default to).
            """
            self.metadata[key] = _GGUFTypedArray(values, elem_type)

        def add_tensor(
            self,
            name: str,
            data: Union[np.ndarray, Any],
            tensor_type: GGMLType = GGMLType.F32
        ) -> None:
            """Add tensor to be written."""
            # Convert torch tensor if needed
            if HAS_TORCH and isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()

            self.tensors.append(GGUFTensor(
                name=name,
                data=data,
                type=tensor_type,
                shape=data.shape if hasattr(data, 'shape') else ()
            ))

        def write(self) -> None:
            """Write the GGUF file."""
            with open(self.filepath, 'wb') as f:
                self._file = f
                self._write_header()
                self._write_metadata()
                self._write_tensors_info()
                self._write_tensor_data()

            logger.info(f"Written GGUF file: {self.filepath}")

        def _write_header(self) -> None:
            """Write GGUF header."""
            self._file.write(struct.pack('<I', GGUF_MAGIC))
            self._file.write(struct.pack('<I', GGUF_VERSION))
            self._file.write(struct.pack('<Q', len(self.tensors)))
            self._file.write(struct.pack('<Q', len(self.metadata)))

        def _write_metadata(self) -> None:
            """Write metadata key-value pairs.

            Looks up the spec-required scalar type for each key via
            `_gguf_scalar_type`. Falls back to Python-type inference
            when the key is not in the table (custom / unknown keys).
            """
            for key, value in self.metadata.items():
                self._write_string(key)
                forced = _gguf_scalar_type(key)
                self._write_value(value, forced_type=forced)

        def _write_tensors_info(self) -> None:
            """Write tensor information."""
            self._offset_positions: list[int] = []
            for tensor in self.tensors:
                self._write_string(tensor.name)
                self._file.write(struct.pack('<I', tensor.n_dims))
                for dim in reversed(tensor.shape):
                    self._file.write(struct.pack('<Q', dim))
                self._file.write(struct.pack('<I', tensor.type.value))
                self._offset_positions.append(self._file.tell())
                self._file.write(struct.pack('<Q', 0))  # Offset placeholder

        def _write_tensor_data(self) -> None:
            """Write tensor data with alignment."""
            # Align to 32 bytes
            current_pos = self._file.tell()
            padding = (32 - (current_pos % 32)) % 32
            self._file.write(b'\x00' * padding)

            data_start = self._file.tell()

            for i, tensor in enumerate(self.tensors):
                # Record offset relative to data section start
                tensor_offset = self._file.tell() - data_start

                data = tensor.data

                # Convert to appropriate format
                if tensor.type == GGMLType.F32:
                    data = data.astype(np.float32)
                elif tensor.type == GGMLType.F16:
                    data = data.astype(np.float16)
                elif tensor.type == GGMLType.BF16:
                    data = data.astype(np.float32).view(np.uint32)
                    data = (data >> 16).astype(np.uint16)

                self._file.write(data.tobytes())

                # Pad to alignment
                size = len(data.tobytes())
                padding = (32 - (size % 32)) % 32
                self._file.write(b'\x00' * padding)

                # Seek back to fill in the offset placeholder
                end_pos = self._file.tell()
                self._file.seek(self._offset_positions[i])
                self._file.write(struct.pack('<Q', tensor_offset))
                self._file.seek(end_pos)

        def _write_string(self, s: str) -> None:
            """Write a string value."""
            encoded = s.encode('utf-8')
            self._file.write(struct.pack('<Q', len(encoded)))
            self._file.write(encoded)

        def _write_value(
            self,
            value: Any,
            forced_type: Optional[GGUFValueType] = None,
        ) -> None:
            """Write a typed metadata value.

            When `forced_type` is supplied (typically because the key
            is in the GGUF spec's UINT32/FLOAT32 list), it overrides
            Python-type inference. Without this, every positive int
            falls through to UINT64, and llama.cpp aborts when reading
            keys it expects as UINT32 (e.g. `llama.context_length`).
            """
            # Forced scalar types — bypass Python isinstance dispatch.
            if forced_type is GGUFValueType.UINT32:
                self._file.write(struct.pack('<I', GGUFValueType.UINT32.value))
                self._file.write(struct.pack('<I', int(value)))
                return
            if forced_type is GGUFValueType.INT32:
                self._file.write(struct.pack('<I', GGUFValueType.INT32.value))
                self._file.write(struct.pack('<i', int(value)))
                return
            if forced_type is GGUFValueType.FLOAT32:
                self._file.write(struct.pack('<I', GGUFValueType.FLOAT32.value))
                self._file.write(struct.pack('<f', float(value)))
                return

            if isinstance(value, bool):
                self._file.write(struct.pack('<I', GGUFValueType.BOOL.value))
                self._file.write(struct.pack('<?', value))
            elif isinstance(value, int):
                if value < 0:
                    self._file.write(struct.pack('<I', GGUFValueType.INT64.value))
                    self._file.write(struct.pack('<q', value))
                else:
                    self._file.write(struct.pack('<I', GGUFValueType.UINT64.value))
                    self._file.write(struct.pack('<Q', value))
            elif isinstance(value, float):
                self._file.write(struct.pack('<I', GGUFValueType.FLOAT32.value))
                self._file.write(struct.pack('<f', value))
            elif isinstance(value, str):
                self._file.write(struct.pack('<I', GGUFValueType.STRING.value))
                self._write_string(value)
            elif isinstance(value, _GGUFTypedArray):
                self._write_array(value.values, value.element_type)
            elif isinstance(value, (list, tuple)):
                # Default array element type inference (legacy path).
                if len(value) == 0:
                    elem_type = GGUFValueType.UINT32
                elif isinstance(value[0], str):
                    elem_type = GGUFValueType.STRING
                elif isinstance(value[0], int):
                    elem_type = GGUFValueType.INT32
                elif isinstance(value[0], float):
                    elem_type = GGUFValueType.FLOAT32
                else:
                    elem_type = GGUFValueType.UINT32
                self._write_array(value, elem_type)

        def _write_array(
            self,
            values: Any,
            elem_type: GGUFValueType,
        ) -> None:
            """Write a typed array value.

            llama.cpp reads `tokenizer.ggml.token_type` as INT32 and
            `tokenizer.ggml.scores` as FLOAT32; the legacy code wrote
            ints as INT64 here, which causes llama.cpp to refuse load
            even after the UINT32 scalar fix above.
            """
            self._file.write(struct.pack('<I', GGUFValueType.ARRAY.value))
            self._file.write(struct.pack('<I', elem_type.value))
            self._file.write(struct.pack('<Q', len(values)))
            for item in values:
                if elem_type == GGUFValueType.STRING:
                    self._write_string(item)
                elif elem_type == GGUFValueType.INT32:
                    self._file.write(struct.pack('<i', int(item)))
                elif elem_type == GGUFValueType.UINT32:
                    self._file.write(struct.pack('<I', int(item)))
                elif elem_type == GGUFValueType.INT64:
                    self._file.write(struct.pack('<q', int(item)))
                elif elem_type == GGUFValueType.UINT64:
                    self._file.write(struct.pack('<Q', int(item)))
                elif elem_type == GGUFValueType.FLOAT32:
                    self._file.write(struct.pack('<f', float(item)))
                elif elem_type == GGUFValueType.FLOAT64:
                    self._file.write(struct.pack('<d', float(item)))
                else:
                    raise ValueError(
                        f"Unsupported GGUF array element type: {elem_type!r}"
                    )


    class GGUFExporter:
        """Export Enigma AI Engine models to GGUF format."""

        def __init__(self, quantization: str = "f16") -> None:
            """
            Initialize exporter.

            Args:
                quantization: Quantization type (f32, f16, q8_0, q4_0, q4_k)
            """
            self.quantization = quantization.lower()

        def export(
            self,
            model: Any,
            output_path: str,
            metadata: GGUFMetadata = None,
            tokenizer: Any = None
        ) -> str:
            """
            Export model to GGUF format.

            Returns:
                Path to exported file
            """
            if not HAS_TORCH:
                raise ImportError("PyTorch required for model export")

            output_path = Path(output_path)
            if not output_path.suffix:
                output_path = output_path.with_suffix('.gguf')

            writer = GGUFWriter(str(output_path))

            # Add metadata
            if metadata is None:
                metadata = GGUFMetadata()
                metadata = self._infer_metadata(model, metadata)

            for key, value in metadata.to_dict().items():
                writer.add_metadata(key, value)

            # Add tokenizer vocab if available
            if tokenizer:
                self._add_tokenizer_metadata(writer, tokenizer)

            # Convert and add tensors
            state_dict = model.state_dict() if hasattr(model, 'state_dict') else model

            for name, tensor in state_dict.items():
                gguf_name = convert_tensor_name(name)
                data = tensor.detach().cpu().numpy()
                tensor_type = GGMLType.F32

                # Quantize if needed
                if self.quantization == "q4_0" and self._should_quantize(gguf_name):
                    data, tensor_type = GGUFQuantizer.quantize_q4_0(data)
                elif self.quantization == "q8_0" and self._should_quantize(gguf_name):
                    data, tensor_type = GGUFQuantizer.quantize_q8_0(data)
                elif self.quantization == "q4_k" and self._should_quantize(gguf_name):
                    data, tensor_type = GGUFQuantizer.quantize_q4_k(data)
                elif self.quantization == "f16":
                    data = data.astype(np.float16)
                    tensor_type = GGMLType.F16

                writer.add_tensor(gguf_name, data, tensor_type)

            writer.write()
            logger.info(f"Exported model to {output_path}")

            return str(output_path)

        def _should_quantize(self, name: str) -> bool:
            """Check if tensor should be quantized."""
            skip_patterns = ["embd", "norm", "bias"]
            return not any(p in name.lower() for p in skip_patterns)

        def _infer_metadata(self, model: Any, metadata: GGUFMetadata) -> GGUFMetadata:
            """Infer metadata from model config.

            Supports both HuggingFace-style configs (hidden_size, num_hidden_layers)
            and ForgeConfig-style configs (dim, n_layers, n_heads, etc.).
            """
            if hasattr(model, 'config'):
                config = model.config

                # Embedding dimension: HF hidden_size or ForgeConfig dim
                if hasattr(config, 'hidden_size'):
                    metadata.embedding_length = config.hidden_size
                elif hasattr(config, 'dim'):
                    metadata.embedding_length = config.dim

                # Layer count: HF num_hidden_layers or ForgeConfig n_layers
                if hasattr(config, 'num_hidden_layers'):
                    metadata.block_count = config.num_hidden_layers
                elif hasattr(config, 'n_layers'):
                    metadata.block_count = config.n_layers

                # Feed-forward dimension: HF intermediate_size or ForgeConfig hidden_dim
                if hasattr(config, 'intermediate_size'):
                    metadata.feed_forward_length = config.intermediate_size
                elif hasattr(config, 'hidden_dim'):
                    metadata.feed_forward_length = config.hidden_dim

                # Attention heads: HF num_attention_heads or ForgeConfig n_heads
                if hasattr(config, 'num_attention_heads'):
                    metadata.attention_head_count = config.num_attention_heads
                elif hasattr(config, 'n_heads'):
                    metadata.attention_head_count = config.n_heads

                # KV heads: HF num_key_value_heads or ForgeConfig n_kv_heads
                if hasattr(config, 'num_key_value_heads'):
                    metadata.attention_head_count_kv = config.num_key_value_heads
                elif hasattr(config, 'n_kv_heads'):
                    metadata.attention_head_count_kv = config.n_kv_heads

                # Vocab size (same attr name in both)
                if hasattr(config, 'vocab_size'):
                    metadata.vocab_size = config.vocab_size

                # Context length: HF max_position_embeddings or ForgeConfig max_seq_len
                if hasattr(config, 'max_position_embeddings'):
                    metadata.context_length = config.max_position_embeddings
                elif hasattr(config, 'max_seq_len'):
                    metadata.context_length = config.max_seq_len

                # RoPE theta (ForgeConfig)
                if hasattr(config, 'rope_theta') and config.rope_theta:
                    metadata.rope_freq_base = config.rope_theta

            # RMSNorm epsilon — read from the actual final-norm module
            # if present, otherwise leave the default. llama.cpp requires
            # this key for the "llama" arch.
            try:
                final_norm = getattr(model, 'norm', None)
                eps = getattr(final_norm, 'eps', None)
                if isinstance(eps, (int, float)) and eps > 0:
                    metadata.attention_layer_norm_rms_epsilon = float(eps)
            except Exception:
                pass

            return metadata

        def _add_tokenizer_metadata(self, writer: GGUFWriter, tokenizer: Any) -> None:
            """Add tokenizer vocabulary to metadata.

            llama.cpp reads:
              tokenizer.ggml.tokens     -> ARRAY[STRING]
              tokenizer.ggml.scores     -> ARRAY[FLOAT32]
              tokenizer.ggml.token_type -> ARRAY[INT32]
            Writing token_type as INT64 (the default for `int` lists)
            causes llama.cpp to reject the file.
            """
            try:
                if hasattr(tokenizer, 'get_vocab'):
                    vocab = tokenizer.get_vocab()
                    tokens = [""] * len(vocab)
                    for token, idx in vocab.items():
                        if idx < len(tokens):
                            tokens[idx] = token

                    writer.add_metadata("tokenizer.ggml.tokens", tokens)
                    writer.add_metadata_array(
                        "tokenizer.ggml.scores",
                        [0.0] * len(tokens),
                        GGUFValueType.FLOAT32,
                    )
                    writer.add_metadata_array(
                        "tokenizer.ggml.token_type",
                        [1] * len(tokens),  # 1 = LLAMA_TOKEN_TYPE_NORMAL
                        GGUFValueType.INT32,
                    )
            except Exception as e:
                logger.warning(f"Could not add tokenizer metadata: {e}")


    def export_to_gguf(
        model: Any,
        output_path: Union[str, Path],
        quant_type: str = "F16",
        tokenizer: Any = None,
        metadata: GGUFMetadata = None,
        model_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Export a model to GGUF format.

        Args:
            model: PyTorch model or state dict
            output_path: Output GGUF file path
            quant_type: Quantization type (F16, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0)
            tokenizer: Tokenizer for vocabulary
            metadata: Model metadata
            model_name: Model name for metadata
            description: Model description

        Returns:
            Path to exported file
        """
        # Map string quant type to lowercase for exporter
        quant_lower = quant_type.lower().replace('_', '')
        if 'q4k' in quant_lower:
            quant_lower = 'q4_k'
        elif 'q8' in quant_lower:
            quant_lower = 'q8_0'
        elif 'q4' in quant_lower and 'k' not in quant_lower:
            quant_lower = 'q4_0'
        elif 'f16' in quant_lower:
            quant_lower = 'f16'
        else:
            quant_lower = 'f16'

        # Create metadata if custom values provided
        if metadata is None and (model_name or description):
            metadata = GGUFMetadata()
            if model_name:
                metadata.general_name = model_name
            if description:
                metadata.general_description = description

        exporter = GGUFExporter(quantization=quant_lower)
        return exporter.export(model, str(output_path), metadata, tokenizer)


else:
    # Stub implementations when NumPy not available

    class GGUFQuantizer:
        """Stub when NumPy not available."""
        @classmethod
        def quantize_q4_0(cls, *args, **kwargs):
            raise ImportError("NumPy required for quantization")

        @classmethod
        def quantize_q8_0(cls, *args, **kwargs):
            raise ImportError("NumPy required for quantization")

    class GGUFWriter:
        """Stub when NumPy not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("NumPy required for GGUF export")

    class GGUFExporter:
        """Stub when NumPy not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("NumPy required for GGUF export")

    def export_to_gguf(*args, **kwargs):
        """Stub when NumPy not available."""
        raise ImportError("NumPy required for GGUF export")


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    'GGML_BLOCK_SIZES',
    'GGUF_MAGIC',
    'GGUF_VERSION',
    'QUANT_TYPES',
    'GGMLType',
    'GGUFExporter',
    'GGUFMetadata',
    'GGUFMetadataType',
    'GGUFQuantizer',
    'GGUFTensor',
    'GGUFValueType',
    'GGUFWriter',
    'convert_tensor_name',
    'export_to_gguf',
    'parse_gguf_header',
    'parse_gguf_metadata',
    'read_gguf_value',
]
