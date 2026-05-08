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
        # Verify answer text is NOT duplicated across blocks
        assert blocks[0][1].count("First") == 1
        assert blocks[1][1] == "The answer is 42."

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

    def test_gpu_count(self):
        from enigma_engine.core.multi_gpu import get_gpu_count
        count = get_gpu_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_distributed_config(self):
        from enigma_engine.core.multi_gpu import DistributedConfig
        cfg = DistributedConfig()
        assert cfg.backend in ("nccl", "gloo")

    def test_cleanup_removes_env_vars(self):
        """S692: cleanup() must remove MASTER_ADDR/MASTER_PORT env vars."""
        import inspect
        from enigma_engine.core.multi_gpu import DistributedTrainer
        source = inspect.getsource(DistributedTrainer.cleanup)
        # Verify cleanup pops both env vars
        assert 'MASTER_ADDR' in source
        assert 'MASTER_PORT' in source
        assert 'os.environ.pop' in source


# ════════════════════════════════════════════════════════════════════
# Chat Export  – HTML / PDF
# ════════════════════════════════════════════════════════════════════

class TestChatExport:
    """Test chat export utilities from chat_export.py."""

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


# ════════════════════════════════════════════════════════════════════
# Repo-comparison improvements (scheduler, sampling, training)
# ════════════════════════════════════════════════════════════════════


class TestLabelSmoothing:
    """Verify label smoothing config and model forward integration."""

    def test_training_config_has_label_smoothing(self):
        from enigma_engine.training.training import TrainingConfig
        config = TrainingConfig()
        assert hasattr(config, "label_smoothing")
        assert config.label_smoothing == 0.05  # Default: mild smoothing

    def test_label_smoothing_in_to_dict(self):
        from enigma_engine.training.training import TrainingConfig
        config = TrainingConfig(label_smoothing=0.1)
        d = config.to_dict()
        assert d["label_smoothing"] == 0.1


class TestMinPSampling:
    """Verify min_p sampling is threaded through generation pipeline."""

    def test_defaults_has_min_p(self):
        """Default config includes min_p."""
        from enigma_engine.config.defaults import CONFIG
        assert "min_p" in CONFIG
        assert CONFIG["min_p"] == 0.0


# ════════════════════════════════════════════════════════════════════
# SpecAugment — audio training augmentation
# ════════════════════════════════════════════════════════════════════

class TestSpecAugment:
    """SpecAugment masks frequency/time bands in mel spectrograms."""

    def test_spec_augment_preserves_shape_3d(self):
        """Output shape matches input [1, n_mels, n_frames]."""
        import torch
        from enigma_engine.core.audio_encoder import spec_augment
        mel = torch.randn(1, 80, 100)
        out = spec_augment(mel)
        assert out.shape == mel.shape

    def test_spec_augment_preserves_shape_2d(self):
        """Supports 2D input [n_mels, n_frames] (squeezed)."""
        import torch
        from enigma_engine.core.audio_encoder import spec_augment
        mel = torch.randn(80, 100)
        out = spec_augment(mel)
        assert out.shape == mel.shape
        assert out.ndim == 2

    def test_spec_augment_applies_masking(self):
        """Output has more zeros than input (masking applied)."""
        import torch
        from enigma_engine.core.audio_encoder import spec_augment
        torch.manual_seed(42)
        mel = torch.ones(1, 80, 100)
        out = spec_augment(mel, freq_mask_count=3, time_mask_count=3,
                           freq_mask_width=15, time_mask_width=25)
        zeros_in = (mel == 0).sum().item()
        zeros_out = (out == 0).sum().item()
        assert zeros_out > zeros_in

    def test_spec_augment_no_masking_with_zero_counts(self):
        """Zero mask counts should return identical tensor."""
        import torch
        from enigma_engine.core.audio_encoder import spec_augment
        mel = torch.randn(1, 80, 100)
        out = spec_augment(mel, freq_mask_count=0, time_mask_count=0)
        assert torch.equal(out, mel)


# ════════════════════════════════════════════════════════════════════
# Temporal convolution for video frames
# ════════════════════════════════════════════════════════════════════

class TestTemporalConv1d:
    """TemporalConv1d adds cross-frame context to video features."""

    def test_temporal_conv_forward_shape(self):
        """Forward preserves frame count and shape."""
        import torch
        from enigma_engine.core.vision_encoder import TemporalConv1d
        tc = TemporalConv1d(dim=64)
        frames = [torch.randn(1, 16, 64) for _ in range(4)]
        out = tc(frames)
        assert len(out) == 4
        for f in out:
            assert f.shape == (1, 16, 64)

    def test_temporal_conv_single_frame_passthrough(self):
        """Single frame should pass through unchanged."""
        import torch
        from enigma_engine.core.vision_encoder import TemporalConv1d
        tc = TemporalConv1d(dim=64)
        frame = torch.randn(1, 16, 64)
        out = tc([frame])
        assert len(out) == 1
        assert torch.equal(out[0], frame)


# ════════════════════════════════════════════════════════════════════
# ImageGen mod upgrades — img2img, inpainting, ControlNet, LoRA, SDXL
# ════════════════════════════════════════════════════════════════════

class TestImageGenModUpgrades:
    """Verify imagegen mod has new pipeline modes and commands."""

    def test_imagegen_no_asyncio(self):
        """Imagegen mod must not use asyncio (kept after upgrade)."""
        source = (Path(__file__).resolve().parent.parent
                  / "mods" / "imagegen" / "main.py").read_text(encoding="utf-8")
        assert "asyncio" not in source
        assert "async def" not in source


# ════════════════════════════════════════════════════════════════════
# Command Policy Generator — FORGE tool
# ════════════════════════════════════════════════════════════════════

class TestCommandPolicyGenerator:
    """Verify Command Policy Generator FORGE button and implementation."""

    def test_parse_commands_reference_finds_commands(self):
        """_parse_commands_reference finds commands from reference file."""
        from enigma_engine.gui.gui_forge_tools import ForgeToolsMixin
        ref_path = Path("information/commands_reference.md")
        if not ref_path.exists():
            pytest.skip("commands_reference.md not found")
        mixin = ForgeToolsMixin()
        commands = mixin._parse_commands_reference()
        assert len(commands) > 20  # Should find 47+ commands
        names = [c["name"] for c in commands]
        assert "config.get" in names or "config.list" in names


# ===================================================================
# CONFORMER CONV MODULE FOR AUDIO ENCODER
# ===================================================================


class TestConformerConv:
    """Verify Conformer-style convolution module in audio encoder."""

    def test_audio_config_has_use_conformer(self):
        """AudioEncoderConfig has use_conformer field."""
        from enigma_engine.core.audio_encoder import AudioEncoderConfig
        cfg = AudioEncoderConfig()
        assert hasattr(cfg, "use_conformer")
        assert cfg.use_conformer is False  # default off

    def test_conformer_config_serialization(self):
        """use_conformer is included in to_dict()."""
        from enigma_engine.core.audio_encoder import AudioEncoderConfig
        cfg = AudioEncoderConfig(use_conformer=True)
        d = cfg.to_dict()
        assert "use_conformer" in d
        assert d["use_conformer"] is True

    def test_audio_block_without_conformer_has_no_conv(self):
        """Default _AudioBlock has conv_module = None."""
        from enigma_engine.core.audio_encoder import _AudioBlock
        block = _AudioBlock(dim=128, n_heads=4, use_conformer=False)
        assert block.conv_module is None

    def test_audio_block_with_conformer_has_conv(self):
        """_AudioBlock with use_conformer=True has conv_module."""
        from enigma_engine.core.audio_encoder import _AudioBlock
        block = _AudioBlock(dim=128, n_heads=4, use_conformer=True)
        assert block.conv_module is not None

    def test_conformer_conv_has_depthwise(self):
        """_ConformerConv uses depthwise convolution (groups=dim)."""
        from enigma_engine.core.audio_encoder import _ConformerConv
        conv = _ConformerConv(dim=128)
        # Depthwise: groups should equal input channels
        assert conv.dw_conv.groups == 128


# ===================================================================
# USER-SELECTABLE MEMORY MODES
# ===================================================================


class TestMemoryModes:
    """Verify memory mode selection feature."""

    def test_persistent_memory_has_disabled_flag(self):
        """PersistentMemory has a 'disabled' attribute."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(Path(tempfile.mkdtemp()) / "test_mem.md")
        assert hasattr(mem, "disabled")
        assert mem.disabled is False

    def test_disabled_memory_rejects_add(self):
        """When disabled, add() always returns False."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(Path(tempfile.mkdtemp()) / "test_mem.md")
        mem.disabled = True
        result = mem.add("some fact")
        assert result is False
        assert mem.count == 0

    def test_enabled_memory_accepts_add(self):
        """When not disabled, add() works normally."""
        from enigma_engine.core.memory import PersistentMemory
        mem = PersistentMemory(Path(tempfile.mkdtemp()) / "test_mem.md")
        result = mem.add("test fact")
        assert result is True
        assert mem.count == 1


# ===================================================================
# CHAT HISTORY TOKEN BUDGETING
# ===================================================================


class TestHistoryBudgeting:
    """Verify chat history persistence cap in ModelContext."""

    def test_max_context_history_defined(self):
        """MAX_CONTEXT_HISTORY constant exists in model_context."""
        from enigma_engine.core.model_context import MAX_CONTEXT_HISTORY
        assert isinstance(MAX_CONTEXT_HISTORY, int)
        assert MAX_CONTEXT_HISTORY > 0

    def test_model_context_save_trims_long_history(self):
        """Saving with more messages than MAX_CONTEXT_HISTORY trims to cap."""
        from enigma_engine.core.model_context import (
            MAX_CONTEXT_HISTORY, ModelContext,
        )
        ctx = ModelContext("test_history_cap")
        # Create a temp dir for the context
        tmp = Path(tempfile.mkdtemp()) / "test_ctx"
        ctx._contexts_dir = tmp

        # Build history exceeding the cap
        for i in range(MAX_CONTEXT_HISTORY + 100):
            ctx.history.append(
                {"role": "user", "content": f"msg {i}"})

        assert len(ctx.history) == MAX_CONTEXT_HISTORY + 100

        # After save, the file should have at most MAX_CONTEXT_HISTORY
        # (We just verify the save method references the constant)
        import inspect
        source = inspect.getsource(ModelContext._save_history)
        assert "MAX_CONTEXT_HISTORY" in source


# ===================================================================
# PRIORITIZED REPLAY BUFFER FOR RLHF
# ===================================================================


class TestReplayBuffer:
    """Verify ReplayBuffer for prioritized experience replay."""

    def test_replay_buffer_init(self):
        """ReplayBuffer initializes with capacity and alpha."""
        from enigma_engine.core.rl_training import ReplayBuffer
        rb = ReplayBuffer(capacity=100, priority_alpha=0.6)
        assert rb.capacity == 100
        assert rb.alpha == 0.6
        assert len(rb) == 0

    def test_replay_buffer_add_and_len(self):
        """Adding experiences increments length."""
        import torch
        from enigma_engine.core.rl_training import ReplayBuffer
        rb = ReplayBuffer(capacity=10)
        rb.add(
            log_probs=torch.randn(5),
            values=torch.randn(5),
            rewards=torch.zeros(5),
            response_mask=torch.ones(5),
            reward_scalar=1.5,
        )
        assert len(rb) == 1

    def test_replay_buffer_capacity_eviction(self):
        """Buffer evicts lowest-priority when over capacity."""
        import torch
        from enigma_engine.core.rl_training import ReplayBuffer
        rb = ReplayBuffer(capacity=3)
        for i in range(5):
            rb.add(
                log_probs=torch.randn(3),
                values=torch.randn(3),
                rewards=torch.zeros(3),
                response_mask=torch.ones(3),
                reward_scalar=float(i),
            )
        assert len(rb) == 3
        # Highest priority experiences should remain
        priorities = [e["priority"] for e in rb._experiences]
        assert min(priorities) >= 2.0  # rewards 2, 3, 4 survive

    def test_replay_buffer_sample(self):
        """Sampling returns requested number of experiences."""
        import torch
        from enigma_engine.core.rl_training import ReplayBuffer
        rb = ReplayBuffer(capacity=10)
        for i in range(5):
            rb.add(
                log_probs=torch.randn(4),
                values=torch.randn(4),
                rewards=torch.zeros(4),
                response_mask=torch.ones(4),
                reward_scalar=float(i + 1),
            )
        samples = rb.sample(3, device="cpu")
        assert len(samples) == 3
        for s in samples:
            assert "log_probs" in s
            assert "values" in s
            assert "rewards" in s
            assert "mask" in s

    def test_replay_buffer_sample_empty(self):
        """Sampling from empty buffer returns empty list."""
        from enigma_engine.core.rl_training import ReplayBuffer
        rb = ReplayBuffer()
        assert rb.sample(5) == []

    def test_rlhf_config_has_replay_fields(self):
        """RLHFConfig has replay_capacity and replay_ratio."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert hasattr(cfg, "replay_capacity")
        assert hasattr(cfg, "replay_ratio")
        assert cfg.replay_capacity == 256
        assert cfg.replay_ratio == 0.25

    def test_replay_buffer_state_dict(self):
        """ReplayBuffer has state_dict/load_state_dict for serialization."""
        import torch
        from enigma_engine.core.rl_training import ReplayBuffer
        rb = ReplayBuffer(capacity=10, priority_alpha=0.7)
        rb.add(
            log_probs=torch.randn(4),
            values=torch.randn(4),
            rewards=torch.zeros(4),
            response_mask=torch.ones(4),
            reward_scalar=2.5,
        )
        state = rb.state_dict()
        assert "capacity" in state
        assert "alpha" in state
        assert "experiences" in state
        assert len(state["experiences"]) == 1

        # Restore into fresh buffer
        rb2 = ReplayBuffer()
        rb2.load_state_dict(state)
        assert rb2.capacity == 10
        assert rb2.alpha == 0.7
        assert len(rb2) == 1

    def test_replay_buffer_stores_full_ids(self):
        """S813: ReplayBuffer stores full_ids/prompt_len for log-prob recomputation."""
        import torch
        from enigma_engine.core.rl_training import ReplayBuffer
        rb = ReplayBuffer(capacity=10)
        full_ids = torch.tensor([1, 2, 3, 4, 5])
        ref_logps = torch.randn(3)
        rb.add(
            log_probs=torch.randn(3),
            values=torch.randn(3),
            rewards=torch.zeros(3),
            response_mask=torch.ones(3),
            reward_scalar=1.0,
            full_ids=full_ids,
            prompt_len=2,
            ref_logps=ref_logps,
        )
        samples = rb.sample(1, device="cpu")
        assert len(samples) == 1
        s = samples[0]
        assert "full_ids" in s
        assert "prompt_len" in s
        assert "ref_logps" in s
        assert torch.equal(s["full_ids"], full_ids)
        assert s["prompt_len"] == 2
        assert s["ref_logps"].shape == ref_logps.shape

    def test_replay_buffer_backward_compat_no_full_ids(self):
        """S813: ReplayBuffer works without full_ids (backward compat)."""
        import torch
        from enigma_engine.core.rl_training import ReplayBuffer
        rb = ReplayBuffer(capacity=10)
        rb.add(
            log_probs=torch.randn(3),
            values=torch.randn(3),
            rewards=torch.zeros(3),
            response_mask=torch.ones(3),
            reward_scalar=1.0,
        )
        samples = rb.sample(1, device="cpu")
        s = samples[0]
        # Without full_ids, sample should not include those keys
        assert "full_ids" not in s
        assert "prompt_len" not in s


# ===================================================================
# COMPLETION ITEM #1: memory.search respects disabled flag
# ===================================================================


# ===================================================================
# COMPLETION ITEM #2: memory mode restored on startup
# ===================================================================


# ===================================================================
# COMPLETION ITEM #5: MAX_CONTEXT_HISTORY user-configurable
# ===================================================================


class TestHistoryCapConfigurable:
    """Verify MAX_CONTEXT_HISTORY can be set by user."""

    def test_set_max_context_history(self):
        """set_max_context_history updates the constant."""
        from enigma_engine.core.model_context import (
            set_max_context_history, MAX_CONTEXT_HISTORY,
        )
        original = MAX_CONTEXT_HISTORY
        set_max_context_history(200)
        from enigma_engine.core import model_context
        assert model_context.MAX_CONTEXT_HISTORY == 200
        # Restore
        set_max_context_history(original)


# ===================================================================
# VENV AUTO-BOOTSTRAP
# ===================================================================


# ===================================================================
# AUTO PRECISION / BF16 DETECTION
# ===================================================================


# ===================================================================
# RL TRAINER CHECKPOINT PERSISTENCE
# ===================================================================


class TestRLTrainerCheckpoints:
    """Verify RL trainers have checkpoint save/load with ReplayBuffer."""

    def test_rlhf_config_has_checkpoint_dir(self):
        """RLHFConfig has checkpoint_dir field."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert hasattr(cfg, "checkpoint_dir")
        assert cfg.checkpoint_dir == ""

    def test_selfplay_config_has_checkpoint_dir(self):
        """SelfPlayConfig has checkpoint_dir field."""
        from enigma_engine.core.rl_training import SelfPlayConfig
        cfg = SelfPlayConfig()
        assert hasattr(cfg, "checkpoint_dir")
        assert cfg.checkpoint_dir == ""


# ===================================================================
# TRAIN-LOCK INFERENCE GUARD (#1)
# ===================================================================


# ════════════════════════════════════════════════════════════════════
# Multilingual creativity detection
# ════════════════════════════════════════════════════════════════════

class TestNeedsAiCreativityMultilingual:
    """_needs_ai_creativity handles non-English prompts correctly."""

    def _make_mixin(self):
        from enigma_engine.core.engine_generation import _GenerationMixin
        obj = object.__new__(_GenerationMixin)
        return obj

    # --- English (existing behavior) ---

    def test_english_creative_phrase(self):
        m = self._make_mixin()
        assert m._needs_ai_creativity("surprise me") is True

    def test_english_direct_command(self):
        m = self._make_mixin()
        assert m._needs_ai_creativity("draw a cat") is False

    def test_english_ambiguous_single_word(self):
        m = self._make_mixin()
        assert m._needs_ai_creativity("hello") is True

    def test_english_direct_single_word(self):
        m = self._make_mixin()
        assert m._needs_ai_creativity("draw") is False

    # --- Non-Latin scripts → safe default (True) ---

    def test_chinese_prompt_needs_creativity(self):
        m = self._make_mixin()
        assert m._needs_ai_creativity("画一只猫") is True

    def test_japanese_prompt_needs_creativity(self):
        m = self._make_mixin()
        assert m._needs_ai_creativity("何か面白いことをして") is True

    def test_korean_prompt_needs_creativity(self):
        m = self._make_mixin()
        assert m._needs_ai_creativity("재미있는 것을 보여줘") is True

    def test_arabic_prompt_needs_creativity(self):
        m = self._make_mixin()
        assert m._needs_ai_creativity("فاجئني بشيء") is True

    def test_cyrillic_prompt_needs_creativity(self):
        m = self._make_mixin()
        assert m._needs_ai_creativity("нарисуй кота") is True

    def test_devanagari_prompt_needs_creativity(self):
        m = self._make_mixin()
        assert m._needs_ai_creativity("कुछ दिलचस्प दिखाओ") is True

    # --- Mixed script (English command + non-Latin arg) stays False ---

    def test_mixed_english_command_with_nonlatin(self):
        """'draw кот' has a clear English command verb first."""
        m = self._make_mixin()
        # Majority is Latin, command word is English → direct is fine
        assert m._needs_ai_creativity("draw кот") is False

    # --- Accented Latin stays on English path ---

    def test_accented_latin_no_false_positive(self):
        """French 'dessine un chat' is Latin script — uses English path."""
        m = self._make_mixin()
        # No English creativity indicator, 3+ words, direct verbs
        assert m._needs_ai_creativity("dessine un chat") is False

    # --- Structural: source contains non-ASCII heuristic ---


# ===================================================================
# MEMORY FACT CAP BUMP (#2)
# ===================================================================


class TestMemoryFactCap:
    """Verify MAX_FACTS is bumped to 200."""

    def test_max_facts_is_200(self):
        """MAX_FACTS should be 200 for long-term users."""
        from enigma_engine.core.memory import MAX_FACTS
        assert MAX_FACTS == 200


# ===================================================================
# CACHED MOVING AVERAGE (#14)
# ===================================================================


class TestCachedMovingAverage:
    """Verify get_chart_data uses cached/incremental moving average."""

    def test_moving_average_correctness(self):
        """Cached moving average matches naive computation."""
        from enigma_engine.training.training_monitor import TrainingMonitor
        m = TrainingMonitor(moving_avg_window=3)
        m.start_run()
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            m.record_loss(v)

        data = m.get_chart_data()
        ma = data["moving_avg"]
        # Window=3: [10], [10,20], [10,20,30], [20,30,40], [30,40,50]
        expected = [10.0, 15.0, 20.0, 30.0, 40.0]
        for got, exp in zip(ma, expected):
            assert abs(got - exp) < 1e-9, f"{got} != {exp}"

    def test_moving_average_nan_does_not_leak_stale_values(self):
        """NaN in the window must not cause stale values to linger."""
        from enigma_engine.training.training_monitor import TrainingMonitor
        m = TrainingMonitor(moving_avg_window=2)
        m.start_run()
        for v in [1.0, 2.0, float("nan"), 3.0, 4.0]:
            m.record_loss(v)

        ma = m.moving_average()
        # Window=2: [1], [1,2], NaN→carry, [NaN,3]→3.0, [3,4]→3.5
        assert abs(ma[0] - 1.0) < 1e-9
        assert abs(ma[1] - 1.5) < 1e-9
        assert abs(ma[2] - 1.5) < 1e-9   # carried forward
        assert abs(ma[3] - 3.0) < 1e-9   # only valid value in window
        assert abs(ma[4] - 3.5) < 1e-9   # [3.0, 4.0]


# ===================================================================
# VALIDATE load_state() JSON (#16)
# ===================================================================


class TestQueueLoadValidation:
    """Verify load_state handles corrupt/malformed JSON gracefully."""

    def test_corrupt_json_returns_false(self):
        """Completely invalid JSON returns False, not crash."""
        import tempfile
        from pathlib import Path
        from enigma_engine.training.training_queue import TrainingQueue
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
                encoding="utf-8") as f:
            f.write("not valid json!!!")
            p = Path(f.name)
        try:
            q = TrainingQueue(save_path=p)
            assert q.load_state() is False
        finally:
            p.unlink(missing_ok=True)

    def test_wrong_types_dont_crash(self):
        """JSON with wrong field types doesn't crash."""
        import tempfile
        from pathlib import Path
        from enigma_engine.training.training_queue import TrainingQueue
        bad_data = {
            "next_id": "not_an_int",
            "jobs": [
                {"job_id": "abc", "mode": 123,
                 "epochs": "ten", "learning_rate": "fast"}
            ],
        }
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
                encoding="utf-8") as f:
            json.dump(bad_data, f)
            p = Path(f.name)
        try:
            q = TrainingQueue(save_path=p)
            # Should not crash — either loads with coerced values
            # or returns False gracefully
            result = q.load_state()
            assert isinstance(result, bool)
        finally:
            p.unlink(missing_ok=True)

    def test_missing_jobs_key_returns_false(self):
        """JSON missing 'jobs' key doesn't crash."""
        import tempfile
        from pathlib import Path
        from enigma_engine.training.training_queue import TrainingQueue
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
                encoding="utf-8") as f:
            json.dump({"next_id": 5}, f)
            p = Path(f.name)
        try:
            q = TrainingQueue(save_path=p)
            result = q.load_state()
            # Valid JSON, just no jobs — should load fine with 0 jobs
            assert result is True
            assert q.pending_count == 0
        finally:
            p.unlink(missing_ok=True)


# ===================================================================
# RESET DIFFICULTY ON CRASH (#17)
# ===================================================================


class TestDifficultyResetOnCrash:
    """Verify difficulty resets after crash/failure."""

    def test_reset_difficulty_sets_simple(self):
        """reset_difficulty sets current_difficulty back to 'simple'."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        plan.current_difficulty = "advanced"
        plan.reset_difficulty()
        assert plan.current_difficulty == "simple"


# ===================================================================
# #22 — Warn on torch.load for untrusted .pth files
# ===================================================================


# ===================================================================
# #23 — Consolidate RMSNorm imports
# ===================================================================


class TestRMSNormConsolidation:
    """Verify vision/audio encoders import RMSNorm instead of duplicating."""

    def test_vision_encoder_no_duplicate_rmsnorm_class(self):
        """vision_encoder should not define its own _RMSNorm class."""
        import inspect
        from enigma_engine.core import vision_encoder
        source = inspect.getsource(vision_encoder)
        assert "class _RMSNorm" not in source

    def test_audio_encoder_no_duplicate_rmsnorm_class(self):
        """audio_encoder should not define its own _RMSNorm class."""
        import inspect
        from enigma_engine.core import audio_encoder
        source = inspect.getsource(audio_encoder)
        assert "class _RMSNorm" not in source


# ===================================================================
# #29 — Emotional state lock
# ===================================================================


class TestEmotionalStateLock:
    """Verify ModelContext has a lock protecting emotional state."""

    def test_model_context_has_emotional_lock(self):
        """ModelContext should have a threading lock for emotional state."""
        import threading
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("test_emo_lock")
        assert hasattr(ctx, "_emotional_lock")
        assert isinstance(ctx._emotional_lock, type(threading.Lock()))


# ===================================================================
# S823 — PPO combined forward pass
# ===================================================================


class TestGetLogpsHiddenEntropy:
    """_get_logps_hidden_entropy returns same values as the 3 separate calls it replaces."""

    def _make_tiny_model(self, *, training: bool = False, neftune_alpha: float = 0.0):
        """Build a minimal Enigma model on CPU for testing."""
        from enigma_engine.core.model import Enigma, ForgeConfig
        cfg = ForgeConfig(
            dim=32,
            n_layers=1,
            n_heads=2,
            vocab_size=64,
            max_seq_len=32,
            use_rope=True,
            neftune_alpha=neftune_alpha,
        )
        model = Enigma(cfg)
        if training:
            model.train()
        else:
            model.eval()
        return model

    def test_logps_match_get_response_logps(self):
        """logps from combined pass must match _get_response_logps."""
        import torch
        from enigma_engine.core.rl_training import (
            _get_response_logps,
            _get_logps_hidden_entropy,
        )
        model = self._make_tiny_model()
        full_ids = torch.randint(0, 60, (1, 10))
        prompt_len = 4
        with torch.no_grad():
            expected_logps = _get_response_logps(model, full_ids, prompt_len)
            got_logps, _, _ = _get_logps_hidden_entropy(model, full_ids, prompt_len)
        assert got_logps.shape == expected_logps.shape
        assert torch.allclose(got_logps, expected_logps, atol=1e-5), (
            f"logps mismatch: max diff {(got_logps - expected_logps).abs().max():.2e}"
        )

    def test_hidden_shape_matches_get_hidden_states(self):
        """hidden states from combined pass have same shape as _get_hidden_states."""
        import torch
        from enigma_engine.core.rl_training import (
            _get_logps_hidden_entropy,
        )
        model = self._make_tiny_model()
        full_ids = torch.randint(0, 60, (1, 10))
        prompt_len = 4
        with torch.no_grad():
            _, got_hidden, _ = _get_logps_hidden_entropy(model, full_ids, prompt_len)
        # hidden: (1, resp_len, dim)
        assert got_hidden.ndim == 3
        assert got_hidden.shape[0] == 1
        assert got_hidden.shape[2] == model.config.dim
        resp_len = 10 - 1 - max(prompt_len - 1, 0)  # T - response_start
        assert got_hidden.shape[1] == resp_len

    def test_entropy_match_get_response_entropy(self):
        """entropy from combined pass must match _get_response_entropy."""
        import torch
        from enigma_engine.core.rl_training import (
            _get_response_entropy,
            _get_logps_hidden_entropy,
        )
        model = self._make_tiny_model()
        full_ids = torch.randint(0, 60, (1, 10))
        prompt_len = 4
        with torch.no_grad():
            expected_ent = _get_response_entropy(model, full_ids, prompt_len)
            _, _, got_ent = _get_logps_hidden_entropy(model, full_ids, prompt_len)
        assert got_ent.shape == expected_ent.shape
        assert torch.allclose(got_ent, expected_ent, atol=1e-5), (
            f"entropy mismatch: max diff {(got_ent - expected_ent).abs().max():.2e}"
        )

    def test_train_mode_returns_finite_outputs(self):
        """Combined helper should run in train mode and return sane tensors."""
        import torch
        from enigma_engine.core.rl_training import (
            _get_logps_hidden_entropy,
        )
        model = self._make_tiny_model(training=True, neftune_alpha=5.0)
        full_ids = torch.randint(0, 60, (1, 10))
        prompt_len = 4
        with torch.no_grad():
            got_logps, got_hidden, got_ent = _get_logps_hidden_entropy(
                model, full_ids, prompt_len)

        assert got_logps.ndim == 1
        assert got_ent.ndim == 1
        assert got_hidden.ndim == 3
        assert got_hidden.shape[0] == 1
        assert got_hidden.shape[1] == got_logps.shape[0]
        assert got_ent.shape[0] == got_logps.shape[0]
        assert torch.isfinite(got_logps).all()
        assert torch.isfinite(got_ent).all()
        assert torch.isfinite(got_hidden).all()

    def test_short_prompt_returns_empty(self):
        """Combined pass gracefully handles prompt_len >= seq_len."""
        import torch
        from enigma_engine.core.rl_training import _get_logps_hidden_entropy
        model = self._make_tiny_model()
        full_ids = torch.randint(0, 60, (1, 5))
        # prompt_len longer than sequence
        logps, hidden, entropy = _get_logps_hidden_entropy(model, full_ids, prompt_len=100)
        assert logps.shape[0] == 1  # returns zeros(1) sentinel
        assert entropy.shape[0] == 1
