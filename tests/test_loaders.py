"""TC-12: Tests for loaders — GGUF, HuggingFace, Ollama, ONNX, GPTQ/AWQ.

Tests pure functions, enums, dataclasses, and deferred import guards
without requiring actual model files or optional libraries.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# GGUF Loader — _find_free_port, deferred imports
# ================================================================


class TestGGUFLoaderPureFunctions:
    """Test pure functions in gguf_loader.py."""

    def test_find_free_port_returns_int(self):
        from enigma_engine.core.gguf_loader import _find_free_port
        port = _find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_find_free_port_unique(self):
        """Two calls should return different ports (almost always)."""
        from enigma_engine.core.gguf_loader import _find_free_port
        ports = {_find_free_port() for _ in range(5)}
        # At least 2 unique ports out of 5 calls
        assert len(ports) >= 2

    def test_llama_server_exe_path_defined(self):
        from enigma_engine.core.gguf_loader import LLAMA_SERVER_EXE
        assert isinstance(LLAMA_SERVER_EXE, Path)
        assert LLAMA_SERVER_EXE.name == "llama-server.exe"

    def test_deferred_import_flags_are_bool(self):
        from enigma_engine.core.gguf_loader import HAVE_LLAMA_CPP, HAVE_TORCH
        assert isinstance(HAVE_LLAMA_CPP, bool)
        assert isinstance(HAVE_TORCH, bool)

    def test_ensure_gguf_imports_idempotent(self):
        """Calling _ensure_gguf_imports twice doesn't crash."""
        from enigma_engine.core.gguf_loader import _ensure_gguf_imports
        _ensure_gguf_imports()
        _ensure_gguf_imports()


# ================================================================
# HuggingFace Loader — format_param_count, _LazyFlag
# ================================================================


class TestHuggingFaceLoaderPureFunctions:
    """Test pure functions in huggingface_loader.py."""

    @pytest.mark.parametrize("params, expected", [
        (500, "500"),
        (1_500, "2K"),
        (42_000, "42K"),
        (124_000_000, "124M"),
        (7_000_000_000, "7.0B"),
        (13_500_000_000, "13.5B"),
    ])
    def test_format_param_count(self, params, expected):
        from enigma_engine.core.huggingface_loader import format_param_count
        assert format_param_count(params) == expected

    def test_lazy_flag_is_bool_compatible(self):
        from enigma_engine.core.huggingface_loader import HAVE_TRANSFORMERS
        # Should be evaluable as bool (True or False)
        result = bool(HAVE_TRANSFORMERS)
        assert isinstance(result, bool)

    def test_lazy_flag_repr(self):
        from enigma_engine.core.huggingface_loader import HAVE_TRANSFORMERS
        r = repr(HAVE_TRANSFORMERS)
        assert "HAVE_TRANSFORMERS" in r

    def test_ensure_imports_returns_bool(self):
        from enigma_engine.core.huggingface_loader import _ensure_imports
        result = _ensure_imports()
        assert isinstance(result, bool)

    def test_ensure_imports_idempotent(self):
        from enigma_engine.core.huggingface_loader import _ensure_imports
        r1 = _ensure_imports()
        r2 = _ensure_imports()
        assert r1 == r2

    def test_convert_hf_config_gpt2_arch_flags(self):
        """S795: GPT-2 must get use_rope=False, use_rms_norm=False,
        use_swiglu=False, use_bias=True — not ForgeConfig defaults."""
        from enigma_engine.core.huggingface_loader import convert_hf_config_to_forge

        class _FakeCfg:
            model_type = "gpt2"
            vocab_size = 50257
            n_embd = 768
            n_layer = 12
            n_head = 12
            n_positions = 1024

        cfg = convert_hf_config_to_forge(_FakeCfg())
        # GPT-2 uses absolute pos embeds, LayerNorm, GELU, bias=True
        assert cfg.get("use_rope") is False, "GPT-2 should not use RoPE"
        assert cfg.get("use_rms_norm") is False, "GPT-2 uses LayerNorm, not RMSNorm"
        assert cfg.get("use_swiglu") is False, "GPT-2 uses GELU, not SwiGLU"
        assert cfg.get("use_bias") is True, "GPT-2 uses bias"

    def test_convert_hf_config_llama_arch_flags(self):
        """S795: LLaMA should get modern defaults (RoPE, RMSNorm, SwiGLU, no bias)."""
        from enigma_engine.core.huggingface_loader import convert_hf_config_to_forge

        class _FakeCfg:
            model_type = "llama"
            vocab_size = 32000
            hidden_size = 4096
            num_hidden_layers = 32
            num_attention_heads = 32
            num_key_value_heads = 8
            max_position_embeddings = 4096
            intermediate_size = 11008
            rope_theta = 10000.0

        cfg = convert_hf_config_to_forge(_FakeCfg())
        # LLaMA uses RoPE, RMSNorm, SwiGLU, no bias — ForgeConfig defaults are correct
        # Either explicitly set True or not set (so ForgeConfig defaults apply)
        assert cfg.get("use_rope", True) is True
        assert cfg.get("use_rms_norm", True) is True
        assert cfg.get("use_swiglu", True) is True
        assert cfg.get("use_bias", False) is False

    def test_chat_with_tools_no_universal_router(self):
        """S794: HuggingFaceEngine.chat_with_tools imports non-existent module."""
        import inspect
        from enigma_engine.core.huggingface_loader import HuggingFaceEngine
        source = inspect.getsource(HuggingFaceEngine.chat_with_tools)
        # Must NOT have a bare 'from .universal_router import' without try/except
        has_try = "try:" in source
        has_import = "universal_router" in source
        if has_import:
            assert has_try, (
                "S794: chat_with_tools imports universal_router without "
                "try/except — always crashes with ImportError"
            )


# ================================================================
# Ollama Loader — OllamaQuantType, OllamaModelInfo
# ================================================================


class TestOllamaLoaderTypes:
    """Test Ollama loader enums and dataclasses."""

    def test_quant_type_values(self):
        from enigma_engine.core.ollama_loader import OllamaQuantType
        assert OllamaQuantType.F32.value == "f32"
        assert OllamaQuantType.F16.value == "f16"
        assert OllamaQuantType.Q4_0.value == "q4_0"
        assert OllamaQuantType.Q8_0.value == "q8_0"

    def test_quant_type_count(self):
        from enigma_engine.core.ollama_loader import OllamaQuantType
        assert len(OllamaQuantType) == 12

    def test_model_info_defaults(self):
        from enigma_engine.core.ollama_loader import OllamaModelInfo
        info = OllamaModelInfo(
            name="test", size=1000, digest="abc",
            modified_at="2026-01-01", quantization="q4_0",
            parameter_size="7B", family="llama",
        )
        assert info.name == "test"
        assert info.context_length == 4096
        assert info.template == ""

    def test_model_info_custom(self):
        from enigma_engine.core.ollama_loader import OllamaModelInfo
        info = OllamaModelInfo(
            name="custom", size=5000, digest="def",
            modified_at="2026-01-01", quantization="q8_0",
            parameter_size="13B", family="mistral",
            context_length=8192, template="{{prompt}}",
        )
        assert info.context_length == 8192
        assert info.family == "mistral"

    def test_ollama_deferred_import_flags(self):
        from enigma_engine.core.ollama_loader import _ensure_ollama_imports
        _ensure_ollama_imports()
        from enigma_engine.core.ollama_loader import HAS_TORCH, HAS_NUMPY
        assert isinstance(HAS_TORCH, bool)
        assert isinstance(HAS_NUMPY, bool)


# ================================================================
# ONNX Loader — deferred imports
# ================================================================


class TestONNXLoader:
    """Test ONNX loader deferred import guard."""

    def test_ensure_onnx_imports_callable(self):
        from enigma_engine.core.onnx_loader import _ensure_onnx_imports
        _ensure_onnx_imports()  # Should not raise

    def test_onnx_flags_are_bool(self):
        from enigma_engine.core.onnx_loader import _ensure_onnx_imports
        _ensure_onnx_imports()
        from enigma_engine.core.onnx_loader import HAVE_ONNX, HAVE_TORCH
        assert isinstance(HAVE_ONNX, bool)
        assert isinstance(HAVE_TORCH, bool)


# ================================================================
# GPTQ/AWQ Loader — deferred imports
# ================================================================


class TestGPTQAWQLoader:
    """Test GPTQ/AWQ loader deferred import guard."""

    def test_ensure_imports_callable(self):
        from enigma_engine.core.gptq_awq_loader import _ensure_imports
        _ensure_imports()  # Should not raise

    def test_flags_are_bool(self):
        from enigma_engine.core.gptq_awq_loader import _ensure_imports
        _ensure_imports()
        from enigma_engine.core.gptq_awq_loader import (
            HAVE_TORCH, HAVE_TRANSFORMERS,
        )
        assert isinstance(HAVE_TORCH, bool)
        assert isinstance(HAVE_TRANSFORMERS, bool)
