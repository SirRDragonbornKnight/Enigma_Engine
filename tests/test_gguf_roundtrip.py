"""
ARCH-V1 — GGUF round-trip verification.

Goal: prove (or refute) that the existing GGUFExporter produces files
llama-cpp-python can load and generate from.

History:
- May 6, 2026 — ARCH-V1 SHIPPED. Established baseline: tensor-name
  mapping was HF-style, Enigma's state_dict was Llama-style, naive
  str.replace also substring-collided (`norm -> output_norm` rewrote
  `attention_norm`). End-to-end load aborts at the C-level.
- May 6, 2026 — ARCH-V1b SHIPPED. Rewrote `convert_tensor_name` to a
  regex pipeline; the 3 structural xfails came off and were extended
  to 16 mapping tests covering all Llama-style + HF-style entries.
- May 6, 2026 — ARCH-V1c SHIPPED. Fixed metadata value-types in
  `GGUFWriter._write_value`: per-key UINT32/FLOAT32 lookup table,
  typed-array path for `tokenizer.ggml.token_type` (INT32) and
  `tokenizer.ggml.scores` (FLOAT32), added missing
  `{arch}.vocab_size` and `{arch}.attention.layer_norm_rms_epsilon`
  keys. File now LOADS cleanly in llama.cpp; abort gone.
- ARCH-V1d (open) — Enigma defaults to `use_qk_norm=True` (Qwen3-style).
  llama.cpp's `llama` arch silently rejects the 4 QK-norm tensors per
  layer, producing `done_getting_tensors: wrong number of tensors`.
  Fix: detect QK-norm and emit `general.architecture=qwen3` plus the
  Qwen3-specific metadata schema.
- ARCH-V1e (open) — byte-vocab tokenizer with `tokenizer.ggml.model=llama`
  has no SentencePiece BPE merges so llama.cpp's tokenizer chokes at
  generation time. Fix: emit a real BPE merges array OR set tokenizer
  model to a no-vocab variant for byte-level models.

Skip behavior:
- llama-cpp-python may not be loadable in this env (Windows: needs
  torch's CUDA DLL dir on PATH). Round-trip tests skip cleanly.
- When llama.cpp IS available, round-trip tests run in a SUBPROCESS so
  llama.cpp's `abort()` cannot kill pytest itself.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# llama-cpp-python detection -- file-existence only, no module import here
# (importing llama_cpp at module load polluted torch DLL state for unrelated
# tests that import torch-dependent extensions like timm). Real import is
# deferred to the subprocess driver below.
# ---------------------------------------------------------------------------

def _llama_cpp_dll_present() -> bool:
    try:
        import importlib.util
        spec = importlib.util.find_spec("llama_cpp")
        if spec is None or spec.origin is None:
            return False
        pkg_dir = Path(spec.origin).parent
        return (pkg_dir / "lib" / "llama.dll").exists() or \
               (pkg_dir / "lib" / "libllama.so").exists() or \
               (pkg_dir / "lib" / "libllama.dylib").exists()
    except Exception:
        return False


_HAS_LLAMA = _llama_cpp_dll_present()
skip_no_llama = pytest.mark.skipif(
    not _HAS_LLAMA,
    reason="llama-cpp-python not installed (or shared lib missing).",
)


# ---------------------------------------------------------------------------
# Test fixtures.
# ---------------------------------------------------------------------------

def _build_tiny_model():
    import torch

    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import ForgeConfig

    cfg = ForgeConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=128,
        max_seq_len=64,
        dropout=0.0,
        use_moe=False,
        use_differential_attn=False,
        n_predict_heads=0,
        neftune_alpha=0.0,
    )
    torch.manual_seed(0)
    return Enigma(cfg).eval(), cfg


def _build_tiny_tokenizer():
    """Byte-level vocab: chr(0)..chr(255)."""
    class _ByteVocab:
        def get_vocab(self):
            return {chr(i): i for i in range(256)}

    return _ByteVocab()


def _export_tiny(quantization: str, out_path: Path) -> str:
    from enigma_engine.core.gguf import GGUFExporter, GGUFMetadata

    model, cfg = _build_tiny_model()
    meta = GGUFMetadata(
        general_name="enigma-test",
        context_length=cfg.max_seq_len,
        embedding_length=cfg.dim,
        block_count=cfg.n_layers,
        feed_forward_length=cfg.hidden_dim,
        attention_head_count=cfg.n_heads,
        attention_head_count_kv=cfg.n_kv_heads,
        rope_dimension_count=cfg.dim // cfg.n_heads,
        rope_freq_base=cfg.rope_theta,
        vocab_size=cfg.vocab_size,
    )
    return GGUFExporter(quantization=quantization).export(
        model, str(out_path), meta, _build_tiny_tokenizer()
    )


# ---------------------------------------------------------------------------
# 1. Sanity baseline — exporter writes a non-empty GGUF-magic file.
# ---------------------------------------------------------------------------

class TestGgufExportWritesFile:
    def test_export_f16_produces_gguf_magic(self, tmp_path):
        path = _export_tiny("f16", tmp_path / "tiny.gguf")
        p = Path(path)
        assert p.exists() and p.stat().st_size > 0
        with open(p, "rb") as f:
            assert f.read(4) == b"GGUF"


# ---------------------------------------------------------------------------
# 1b. ARCH-V1c metadata-type audit. llama.cpp aborts with
#     `GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT32)` when an
#     architecture key like `llama.context_length` is written as UINT64
#     (the legacy default for any positive Python int). These tests
#     parse the GGUF file at the byte level and assert each spec key
#     was emitted with the correct GGUFValueType tag — independent of
#     whether llama.cpp is installed.
# ---------------------------------------------------------------------------

import struct


def _parse_gguf_metadata(path) -> dict:
    """Tiny stdlib GGUF parser. Returns {key: (type_int, value_or_None)}.

    Only parses the header + metadata section; tensors are ignored. Used
    to verify scalar / array type tags without depending on gguf-py or
    llama-cpp-python.
    """
    out: dict = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"bad magic: {magic!r}"
        version = struct.unpack("<I", f.read(4))[0]
        assert version >= 3, f"GGUF version {version} not supported by parser"
        _ = struct.unpack("<Q", f.read(8))[0]  # tensor count
        kv_count = struct.unpack("<Q", f.read(8))[0]

        # GGUFValueType ids (from enigma_engine/core/gguf.py).
        UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = (
            4, 5, 6, 7, 8, 9, 10, 11, 12
        )

        def _read_str() -> str:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8")

        def _read_scalar(t: int):
            if t == UINT32:
                return struct.unpack("<I", f.read(4))[0]
            if t == INT32:
                return struct.unpack("<i", f.read(4))[0]
            if t == UINT64:
                return struct.unpack("<Q", f.read(8))[0]
            if t == INT64:
                return struct.unpack("<q", f.read(8))[0]
            if t == FLOAT32:
                return struct.unpack("<f", f.read(4))[0]
            if t == FLOAT64:
                return struct.unpack("<d", f.read(8))[0]
            if t == BOOL:
                return struct.unpack("<?", f.read(1))[0]
            if t == STRING:
                return _read_str()
            raise ValueError(f"unsupported scalar type {t}")

        for _ in range(kv_count):
            key = _read_str()
            value_type = struct.unpack("<I", f.read(4))[0]
            if value_type == ARRAY:
                elem_type = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                # Skip array contents — type tag is what we audit.
                for _ in range(n):
                    _read_scalar(elem_type)
                out[key] = (value_type, elem_type)
            else:
                out[key] = (value_type, _read_scalar(value_type))
    return out


class TestGgufMetadataTypes:
    """ARCH-V1c: every GGUF spec key must carry the correct value-type
    tag. Without these, llama.cpp aborts at C-level with the famous
    `GGML_ASSERT` against `GGUF_TYPE_UINT32`."""

    UINT32, INT32, FLOAT32, ARRAY = 4, 5, 6, 9

    def test_arch_keys_are_uint32(self, tmp_path):
        meta = _parse_gguf_metadata(_export_tiny("f16", tmp_path / "x.gguf"))
        for key in (
            "llama.context_length",
            "llama.embedding_length",
            "llama.block_count",
            "llama.feed_forward_length",
            "llama.attention.head_count",
            "llama.attention.head_count_kv",
            "llama.rope.dimension_count",
            "llama.vocab_size",
        ):
            assert key in meta, f"missing required arch key: {key}"
            assert meta[key][0] == self.UINT32, \
                f"{key} type tag {meta[key][0]} != UINT32 ({self.UINT32})"

    def test_general_file_type_is_uint32(self, tmp_path):
        meta = _parse_gguf_metadata(_export_tiny("f16", tmp_path / "x.gguf"))
        assert meta["general.file_type"][0] == self.UINT32

    def test_special_token_ids_are_uint32(self, tmp_path):
        meta = _parse_gguf_metadata(_export_tiny("f16", tmp_path / "x.gguf"))
        for key in (
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.padding_token_id",
        ):
            assert key in meta, f"missing required tokenizer key: {key}"
            assert meta[key][0] == self.UINT32, \
                f"{key} type tag {meta[key][0]} != UINT32"

    def test_rope_freq_base_is_float32(self, tmp_path):
        meta = _parse_gguf_metadata(_export_tiny("f16", tmp_path / "x.gguf"))
        assert meta["llama.rope.freq_base"][0] == self.FLOAT32

    def test_rms_norm_eps_present_and_float32(self, tmp_path):
        """llama.cpp REQUIRES `llama.attention.layer_norm_rms_epsilon` for
        the llama arch; loading without it errors with `key not found in
        model: llama.attention.layer_norm_rms_epsilon`."""
        meta = _parse_gguf_metadata(_export_tiny("f16", tmp_path / "x.gguf"))
        key = "llama.attention.layer_norm_rms_epsilon"
        assert key in meta, f"missing required hyperparam: {key}"
        assert meta[key][0] == self.FLOAT32

    def test_token_type_array_is_int32(self, tmp_path):
        """llama.cpp reads `tokenizer.ggml.token_type` as ARRAY[INT32].
        The legacy writer emitted bare-int lists as ARRAY[INT64], which
        causes load to fail."""
        meta = _parse_gguf_metadata(_export_tiny("f16", tmp_path / "x.gguf"))
        kind, elem = meta["tokenizer.ggml.token_type"]
        assert kind == self.ARRAY
        assert elem == self.INT32, f"token_type elem type {elem} != INT32"

    def test_scores_array_is_float32(self, tmp_path):
        meta = _parse_gguf_metadata(_export_tiny("f16", tmp_path / "x.gguf"))
        kind, elem = meta["tokenizer.ggml.scores"]
        assert kind == self.ARRAY
        assert elem == self.FLOAT32

    def test_arch_vocab_size_present(self, tmp_path):
        """The legacy exporter emitted only `tokenizer.ggml.vocab_size`,
        but llama.cpp reads the architecture-prefixed `{arch}.vocab_size`
        (here `llama.vocab_size`). Without it, llama.cpp falls back to
        counting tokenizer.ggml.tokens — fine when the tokenizer is
        present, but the spec key is the canonical source."""
        meta = _parse_gguf_metadata(_export_tiny("f16", tmp_path / "x.gguf"))
        assert "llama.vocab_size" in meta


# ---------------------------------------------------------------------------
# 2. Tensor-name audit — these fail TODAY and document the bug location.
# ARCH-V1b will fix `WEIGHT_NAME_MAP` + `convert_tensor_name` and the
# `xfail` marks below come off. `strict=True` means a future xpass without
# removing the marker errors loudly.
# ---------------------------------------------------------------------------

class TestTensorNameMappingIsLlamaStyle:
    """Enigma's state_dict uses Llama-style names. ARCH-V1b (May 6, 2026)
    rewrote `convert_tensor_name` to a regex pipeline that handles them
    correctly. These tests pin the LLAMA-CPP target names and prevent
    the substring-collision bug from coming back."""

    def test_attn_q_name_is_blk_N_attn_q_weight(self):
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("layers.0.attention.wq.weight") == \
            "blk.0.attn_q.weight"

    def test_attn_k_name_is_blk_N_attn_k_weight(self):
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("layers.3.attention.wk.weight") == \
            "blk.3.attn_k.weight"

    def test_attn_v_name_is_blk_N_attn_v_weight(self):
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("layers.7.attention.wv.weight") == \
            "blk.7.attn_v.weight"

    def test_attn_output_name_is_blk_N_attn_output_weight(self):
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("layers.0.attention.wo.weight") == \
            "blk.0.attn_output.weight"

    def test_qk_norm_names(self):
        """QK-norm tensors must map to attn_q_norm / attn_k_norm so
        llama.cpp can detect QK-norm models (Qwen3 family)."""
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("layers.0.attention.q_norm.weight") == \
            "blk.0.attn_q_norm.weight"
        assert convert_tensor_name("layers.0.attention.k_norm.weight") == \
            "blk.0.attn_k_norm.weight"

    def test_ffn_gate_name_is_blk_N_ffn_gate_weight(self):
        """Llama convention: w1 = ffn_gate, w2 = ffn_down, w3 = ffn_up."""
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("layers.0.feed_forward.w1.weight") == \
            "blk.0.ffn_gate.weight"

    def test_ffn_down_name(self):
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("layers.0.feed_forward.w2.weight") == \
            "blk.0.ffn_down.weight"

    def test_ffn_up_name(self):
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("layers.0.feed_forward.w3.weight") == \
            "blk.0.ffn_up.weight"

    def test_attn_norm_is_not_double_substituted(self):
        """Regression gate: previous str.replace pipeline produced
        `attn_output_norm` because `norm -> output_norm` rewrote the
        inner `norm` of `attention_norm`. The new regex pipeline runs
        a single full-string match per rule, so this can't happen
        again."""
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("layers.0.attention_norm.weight") == \
            "blk.0.attn_norm.weight"

    def test_ffn_norm_unchanged_block_form(self):
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("layers.0.ffn_norm.weight") == \
            "blk.0.ffn_norm.weight"

    def test_token_embd(self):
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("tok_embeddings.weight") == \
            "token_embd.weight"

    def test_final_norm_is_output_norm(self):
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("norm.weight") == "output_norm.weight"

    def test_output_unchanged(self):
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("output.weight") == "output.weight"

    def test_unknown_name_passes_through_unchanged(self):
        """Unknown names must NOT be silently mangled — propagate so
        llama.cpp fails loudly on load instead of swallowing."""
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name("some.unknown.tensor.weight") == \
            "some.unknown.tensor.weight"

    def test_hf_style_fallback_still_works(self):
        """HF-style state_dicts also map correctly (q_proj, gate_proj
        etc.). Kept so future HF model imports don't regress."""
        from enigma_engine.core.gguf import convert_tensor_name
        assert convert_tensor_name(
            "model.layers.0.self_attn.q_proj.weight"
        ) == "blk.0.attn_q.weight"
        assert convert_tensor_name(
            "model.layers.0.mlp.gate_proj.weight"
        ) == "blk.0.ffn_gate.weight"
        assert convert_tensor_name(
            "model.layers.0.input_layernorm.weight"
        ) == "blk.0.attn_norm.weight"


# ---------------------------------------------------------------------------
# 3. Round-trip through llama.cpp -- runs in a subprocess so the C-level
# abort() triggered by malformed GGUF can't take pytest down with it.
# Currently expected to FAIL (subprocess returns non-zero) -- strict=True
# xfail marks come off once ARCH-V1b lands and the round-trip works.
# ---------------------------------------------------------------------------

_ROUND_TRIP_DRIVER = textwrap.dedent('''
    import os, sys
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(torch_lib):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    quant = sys.argv[1]
    out_path = sys.argv[2]
    project_root = sys.argv[3]
    sys.path.insert(0, project_root)
    import torch
    import llama_cpp
    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import ForgeConfig
    from enigma_engine.core.gguf import GGUFExporter, GGUFMetadata

    class _ByteVocab:
        def get_vocab(self):
            return {chr(i): i for i in range(256)}

    cfg = ForgeConfig(
        vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
        hidden_dim=128, max_seq_len=64, dropout=0.0, use_moe=False,
        use_differential_attn=False, n_predict_heads=0, neftune_alpha=0.0,
    )
    torch.manual_seed(0)
    model = Enigma(cfg).eval()
    meta = GGUFMetadata(
        general_name="enigma-test",
        context_length=cfg.max_seq_len, embedding_length=cfg.dim,
        block_count=cfg.n_layers, feed_forward_length=cfg.hidden_dim,
        attention_head_count=cfg.n_heads,
        attention_head_count_kv=cfg.n_kv_heads,
        rope_dimension_count=cfg.dim // cfg.n_heads,
        rope_freq_base=cfg.rope_theta, vocab_size=cfg.vocab_size,
    )
    GGUFExporter(quantization=quant).export(model, out_path, meta, _ByteVocab())
    llama = llama_cpp.Llama(model_path=out_path, n_ctx=cfg.max_seq_len,
                            n_gpu_layers=0, verbose=False)
    res = llama.create_completion(prompt="a", max_tokens=1, temperature=0.0)
    assert res and "choices" in res and len(res["choices"]) == 1
    print("OK")
''')


def _run_round_trip_subprocess(quant: str, out_path: Path) -> tuple[int, str, str]:
    env = dict(os.environ)
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(torch_lib):
        env["PATH"] = torch_lib + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(
        [
            sys.executable, "-c", _ROUND_TRIP_DRIVER,
            quant, str(out_path), str(PROJECT_ROOT),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode, proc.stdout, proc.stderr


@skip_no_llama
class TestGgufRoundTrip:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "ARCH-V1d/V1e: file LOADS cleanly post-V1c (UINT32 metadata + "
            "rms_norm_eps shipped). Round-trip still fails on (a) QK-norm "
            "tensors not supported by llama.cpp's `llama` arch — needs "
            "`general.architecture=qwen3` switch (V1d), and (b) byte-vocab "
            "tokenizer model='llama' has no SentencePiece BPE merges so "
            "generation crashes (V1e)."
        ),
    )
    def test_f16_round_trips(self, tmp_path):
        rc, stdout, _ = _run_round_trip_subprocess(
            "f16", tmp_path / "tiny_f16.gguf"
        )
        assert rc == 0 and "OK" in stdout

    @pytest.mark.xfail(
        strict=True,
        reason="ARCH-V1d/V1e: see test_f16_round_trips reason.",
    )
    def test_q8_0_round_trips(self, tmp_path):
        rc, stdout, _ = _run_round_trip_subprocess(
            "q8_0", tmp_path / "tiny_q8.gguf"
        )
        assert rc == 0 and "OK" in stdout

    @pytest.mark.xfail(
        strict=True,
        reason="ARCH-V1d/V1e: see test_f16_round_trips reason.",
    )
    def test_q4_k_round_trips(self, tmp_path):
        rc, stdout, _ = _run_round_trip_subprocess(
            "q4_k", tmp_path / "tiny_q4k.gguf"
        )
        assert rc == 0 and "OK" in stdout
