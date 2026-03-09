"""
Tests for newly added features:
- Multi-step reasoning (CoT-D)
- Reasoning-weighted training loss (CoT-E)
- Data quality scoring & curation (DQ-C / DQ-D)
- Multi-GPU wrappers (MG-B / MG-C)
- Chat export (HTML / PDF)

Run with: python -m pytest tests/test_new_features.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ════════════════════════════════════════════════════════════════════
# CoT-D  – Multi-step reasoning
# ════════════════════════════════════════════════════════════════════

class TestMultiStepReasoning:
    """Test multi-step reasoning helpers in reasoning.py."""

    def test_extract_all_reasoning(self):
        from enigma_engine.core.reasoning import extract_all_reasoning
        text = (
            "<think>Step 1: Identify the problem.</think>"
            "First, let's note the problem.\n"
            "<think>Step 2: Solve it.</think>"
            "The answer is 42."
        )
        blocks = extract_all_reasoning(text)
        assert len(blocks) == 2
        assert "Step 1" in blocks[0][0]
        assert "Step 2" in blocks[1][0]

    def test_count_reasoning_steps(self):
        from enigma_engine.core.reasoning import count_reasoning_steps
        text = "<think>A</think>B<think>C</think>D<think>E</think>F"
        assert count_reasoning_steps(text) == 3

    def test_count_zero_steps(self):
        from enigma_engine.core.reasoning import count_reasoning_steps
        assert count_reasoning_steps("No thinking here.") == 0

    def test_build_multistep_instruction(self):
        from enigma_engine.core.reasoning import build_multistep_reasoning_instruction
        inst = build_multistep_reasoning_instruction()
        assert "<think>" in inst
        assert isinstance(inst, str)

    def test_format_multistep_example(self):
        from enigma_engine.core.reasoning import format_multistep_example
        example = format_multistep_example(
            "What is 2+2?",
            [("Consider basic addition", "We need to add two numbers"),
             ("2 + 2 = 4", "The answer is 4")],
        )
        assert "<think>" in example
        assert "2+2" in example or "2 + 2" in example


# ════════════════════════════════════════════════════════════════════
# MG-B / MG-C  – Multi-GPU
# ════════════════════════════════════════════════════════════════════

class TestMultiGPU:
    """Test multi-GPU utilities from multi_gpu.py."""

    def test_import(self):
        from enigma_engine.core.multi_gpu import (
            get_gpu_count,
            get_gpu_info,
            is_multi_gpu,
            wrap_data_parallel,
            unwrap_data_parallel,
            DistributedConfig,
            DistributedTrainer,
        )
        assert callable(get_gpu_count)
        assert callable(is_multi_gpu)
        assert DistributedConfig is not None

    def test_gpu_count(self):
        from enigma_engine.core.multi_gpu import get_gpu_count
        count = get_gpu_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_gpu_info(self):
        from enigma_engine.core.multi_gpu import get_gpu_info
        info = get_gpu_info()
        assert isinstance(info, list)

    def test_distributed_config(self):
        from enigma_engine.core.multi_gpu import DistributedConfig
        cfg = DistributedConfig()
        assert cfg.backend in ("nccl", "gloo")


# ════════════════════════════════════════════════════════════════════
# Chat Export  – HTML / PDF
# ════════════════════════════════════════════════════════════════════

class TestChatExport:
    """Test chat export utilities from chat_export.py."""

    def test_import(self):
        from enigma_engine.core.chat_export import (
            export_html,
            export_pdf,
            history_to_html,
        )
        assert callable(export_html)
        assert callable(export_pdf)
        assert callable(history_to_html)

    def test_history_to_html_basic(self):
        from enigma_engine.core.chat_export import history_to_html
        history = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
        ]
        html = history_to_html(history, title="Test Chat")
        assert "<!DOCTYPE html>" in html
        assert "Test Chat" in html
        assert "Hello!" in html
        assert "Hi there!" in html

    def test_html_escaping(self):
        from enigma_engine.core.chat_export import history_to_html
        history = [
            {"role": "user", "content": '<script>alert("xss")</script>'},
        ]
        html = history_to_html(history)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_export_html_file(self):
        from enigma_engine.core.chat_export import export_html
        history = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        with tempfile.TemporaryDirectory() as td:
            out = export_html(history, Path(td) / "chat.html")
            assert out.exists()
            content = out.read_text(encoding="utf-8")
            assert "What is 2+2?" in content
            assert "<!DOCTYPE html>" in content

    def test_export_html_ai_name(self):
        from enigma_engine.core.chat_export import history_to_html
        history = [
            {"role": "assistant", "content": "I am Enigma."},
        ]
        html = history_to_html(history, ai_name="Enigma")
        assert "Enigma" in html

    def test_export_pdf_fallback(self):
        """If fpdf2 is not installed, export_pdf should raise ImportError
        and produce an HTML fallback file."""
        from enigma_engine.core.chat_export import export_pdf
        history = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]
        with tempfile.TemporaryDirectory() as td:
            pdf_path = Path(td) / "chat.pdf"
            try:
                export_pdf(history, pdf_path)
                # If fpdf2 is installed, PDF should exist
                assert pdf_path.exists()
            except ImportError as exc:
                # Fallback HTML should have been created
                html_path = pdf_path.with_suffix(".html")
                assert html_path.exists()
                assert "fpdf2" in str(exc)


# ════════════════════════════════════════════════════════════════════
# Lazy-loading in __init__.py
# ════════════════════════════════════════════════════════════════════

class TestLazyLoading:
    """Verify that new modules are accessible via core.__init__ lazy loading."""

    def test_reasoning_eval_lazy(self):
        """reasoning_eval was removed — ReasoningScore should not be in core."""
        with pytest.raises(ImportError):
            from enigma_engine.core import ReasoningScore  # noqa: F401

    def test_data_quality_lazy(self):
        """data_quality was removed — QualityScore should not be in core."""
        with pytest.raises(ImportError):
            from enigma_engine.core import QualityScore  # noqa: F401

    def test_benchmark_lazy(self):
        """benchmark was removed — BenchmarkSuite should not be in core."""
        with pytest.raises(ImportError):
            from enigma_engine.core import BenchmarkSuite  # noqa: F401

    def test_multi_gpu_lazy(self):
        from enigma_engine.core import get_gpu_count, DistributedConfig
        assert callable(get_gpu_count)
        assert DistributedConfig is not None

    def test_chat_export_lazy(self):
        from enigma_engine.core import export_html, history_to_html
        assert callable(export_html)
        assert callable(history_to_html)

    def test_multistep_reasoning_import(self):
        from enigma_engine.core import (
            extract_all_reasoning,
            count_reasoning_steps,
        )
        assert callable(extract_all_reasoning)
        assert callable(count_reasoning_steps)
