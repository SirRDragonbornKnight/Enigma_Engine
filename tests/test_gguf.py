"""Tests for GGUF loader, server backend, and model loading."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# Polish audit — fixes verified 2026-02-26
# ================================================================

class TestGpuSupport:
    """Verify GPU support is properly configured."""

    @pytest.mark.skipif(
        not __import__('torch').cuda.is_available(),
        reason="CUDA not available")
    def test_pytorch_cuda_available(self):
        """PyTorch should detect CUDA GPU."""
        import torch
        assert torch.cuda.is_available(), "CUDA not available in PyTorch"

    @pytest.mark.skipif(
        not __import__('torch').cuda.is_available(),
        reason="CUDA not available")
    def test_pytorch_cuda_version_not_cpu(self):
        """PyTorch should be CUDA build, not CPU-only."""
        import torch
        assert torch.version.cuda is not None, (
            f"PyTorch {torch.__version__} is CPU-only build — needs CUDA")
        assert "+cpu" not in torch.__version__, (
            f"PyTorch {torch.__version__} is CPU-only build — needs CUDA")

    def test_gpu_device_name(self):
        """GPU device name should be available."""
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            assert len(name) > 0, "GPU name is empty"

    def test_hardware_detection_reports_gpu(self):
        """Hardware detection should report GPU."""
        from enigma_engine.core.hardware_detection import detect_hardware
        import torch
        hw = detect_hardware()
        if torch.cuda.is_available():
            assert hw.gpu_available, "GPU available but not detected"
            assert hw.device == "cuda", f"Expected 'cuda', got '{hw.device}'"
            assert hw.gpu_vram_gb > 0, "VRAM should be > 0"

    def test_llama_cpp_gpu_offload(self):
        """llama-cpp-python should support GPU offload."""
        try:
            import os, sys
            original_path = os.environ.get('PATH', '')
            torch_lib = os.path.join(
                sys.prefix, 'Lib', 'site-packages', 'torch', 'lib'
            )
            try:
                if os.path.isdir(torch_lib):
                    os.environ['PATH'] = (
                        torch_lib + os.pathsep + original_path
                    )
                import llama_cpp.llama_cpp as ll
                if hasattr(ll, 'llama_supports_gpu_offload'):
                    assert ll.llama_supports_gpu_offload(), (
                        "llama-cpp-python installed without GPU offload support")
            finally:
                os.environ['PATH'] = original_path
        except ImportError:
            pytest.skip("llama-cpp-python not installed")

    def test_gpu_compute_works(self):
        """GPU compute should produce correct results."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        x = torch.ones(100, device='cuda')
        y = x + x
        assert y.sum().item() == 200.0, "GPU compute produced wrong result"


class TestLlamaServerBackend:
    """Tests for the llama-server subprocess backend in gguf_loader.py."""

    def test_needs_server_backend_returns_false_for_cpu(self):
        """_needs_server_backend should return False when n_gpu_layers=0."""
        from enigma_engine.core.gguf_loader import _needs_server_backend
        assert _needs_server_backend(0) is False

    def test_find_free_port_returns_int(self):
        """_find_free_port should return a valid TCP port."""
        from enigma_engine.core.gguf_loader import _find_free_port
        port = _find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535


# ================================================================
# Persistent Memory system
# ================================================================

class TestGGUFNoSilentFallback:
    """Verify GGUF chat errors propagate instead of silent fallback."""

    def test_gguf_chat_raises_on_error(self):
        """When GGUF chat fails, the error must propagate."""
        from unittest.mock import MagicMock
        from enigma_engine.core.engine_chat import _ChatMixin

        obj = object.__new__(_ChatMixin)
        obj._is_gguf = True
        mock_model = MagicMock()
        mock_model.chat = MagicMock(
            side_effect=RuntimeError("GGUF model crashed"))
        obj.model = mock_model
        import threading
        obj._generation_lock = threading.Lock()
        obj.get_max_context_length = MagicMock(return_value=4096)
        obj.count_tokens = MagicMock(return_value=10)

        with pytest.raises(RuntimeError, match="GGUF model crashed"):
            obj.chat("hello")


# ================================================================
# Suggestion 18: Tokenizer – no silent SimpleTokenizer fallback
# ================================================================

@pytest.mark.structural
class TestHuggingfaceLoaderNoDeadProperty:
    """#49: huggingface_loader must not have dead @property at module level."""

    def test_no_module_level_property(self):
        """Module should not have a @property decorated function at top level."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "core" / "huggingface_loader.py"
        )
        source = source_path.read_text(encoding="utf-8")
        # @property followed by def at module level (not inside a class) is dead code
        # Check there's no @property\ndef pattern outside of class bodies
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == "@property" and i + 1 < len(lines):
                next_line = lines[i + 1]
                # Module-level function would not be indented more than @property
                if next_line.startswith("def "):
                    pytest.fail(
                        f"Dead @property at module level (line {i+1}): "
                        "only works inside classes")


# ================================================================
# Imagegen scheduler choice — Quick Win #4
# ================================================================


@pytest.mark.structural
class TestGGUFDequantErrorPaths:
    """S693: GGUF error paths must not return partial tensor dicts."""

    def test_corruption_paths_return_empty_dict(self):
        """Verify error paths return {} not a partially populated tensors dict."""
        import inspect
        from enigma_engine.core.gguf_dequant import parse_gguf_tensors
        source = inspect.getsource(parse_gguf_tensors)
        # Count occurrences of 'return tensors' — should be exactly 1
        # (the final successful return). Error paths must use 'return {}'.
        return_tensors = source.count("return tensors")
        return_empty = source.count("return {}")
        # The final return is 'return tensors' (success path)
        assert return_tensors == 1, (
            f"Expected exactly 1 'return tensors' (success path), "
            f"found {return_tensors}")
        # Error paths should all return {}
        assert return_empty >= 2, (
            f"Expected at least 2 'return {{}}' (error paths), "
            f"found {return_empty}")

