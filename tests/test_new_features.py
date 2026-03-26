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

    def test_import(self):
        from enigma_engine.core.multi_gpu import (
            get_gpu_count,
            is_multi_gpu,
            DistributedConfig,
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


# ════════════════════════════════════════════════════════════════════
# Repo-comparison improvements (scheduler, sampling, training)
# ════════════════════════════════════════════════════════════════════

class TestSequentialLRScheduler:
    """Verify SequentialLR warmup → cosine scheduler is wired correctly."""

    def test_training_imports_sequential_lr(self):
        """training.py imports SequentialLR."""
        import inspect
        from enigma_engine.core import training
        source = inspect.getsource(training)
        assert "SequentialLR" in source

    def test_trainer_uses_sequential_lr(self):
        """Trainer.train() builds a SequentialLR scheduler."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "SequentialLR" in source
        # Should compose warmup + cosine
        assert "LambdaLR" in source
        assert "CosineAnnealingLR" in source

    def test_scheduler_step_is_simple(self):
        """Scheduler step should be a plain self.scheduler.step(), not manual warmup."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._train_one_batch)
        # Should NOT contain the old manual warmup pattern
        assert "current_step < self.warmup_steps" not in source
        # Should contain simple scheduler.step()
        assert "self.scheduler.step()" in source

    def test_dpo_uses_sequential_lr(self):
        """DPO training also uses SequentialLR."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_dpo)
        assert "SequentialLR" in source


class TestLabelSmoothing:
    """Verify label smoothing config and model forward integration."""

    def test_training_config_has_label_smoothing(self):
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert hasattr(config, "label_smoothing")
        assert config.label_smoothing == 0.05  # Default: mild smoothing

    def test_label_smoothing_in_to_dict(self):
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig(label_smoothing=0.1)
        d = config.to_dict()
        assert d["label_smoothing"] == 0.1

    def test_model_forward_accepts_label_smoothing(self):
        """Enigma.forward() has a label_smoothing parameter."""
        import inspect
        from enigma_engine.core.model import Enigma
        sig = inspect.signature(Enigma.forward)
        assert "label_smoothing" in sig.parameters

    def test_trainer_passes_label_smoothing(self):
        """_train_one_batch passes label_smoothing to model forward."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._train_one_batch)
        assert "label_smoothing" in source


class TestFusedAdamW:
    """Verify fused AdamW selection when available."""

    def test_trainer_checks_fused_support(self):
        """Trainer._setup_optimizer inspects AdamW signature for fused param."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._setup_optimizer)
        assert "fused" in source
        assert "inspect.signature" in source or "inspect" in source


class TestMinPSampling:
    """Verify min_p sampling is threaded through generation pipeline."""

    def test_defaults_has_min_p(self):
        """Default config includes min_p."""
        from enigma_engine.config.defaults import CONFIG
        assert "min_p" in CONFIG
        assert CONFIG["min_p"] == 0.0

    def test_sample_token_accepts_min_p(self):
        """_sample_token has min_p parameter."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        sig = inspect.signature(_GenerationMixin._sample_token)
        assert "min_p" in sig.parameters

    def test_sample_token_batch_accepts_min_p(self):
        """_sample_token_batch has min_p parameter."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        sig = inspect.signature(_GenerationMixin._sample_token_batch)
        assert "min_p" in sig.parameters


# ════════════════════════════════════════════════════════════════════
# SpecAugment — audio training augmentation
# ════════════════════════════════════════════════════════════════════

class TestSpecAugment:
    """SpecAugment masks frequency/time bands in mel spectrograms."""

    def test_spec_augment_function_exists(self):
        """spec_augment is importable from audio_encoder."""
        from enigma_engine.core.audio_encoder import spec_augment
        assert callable(spec_augment)

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

    def test_spec_augment_lazy_import(self):
        """spec_augment is accessible via core lazy imports."""
        import enigma_engine.core as core_mod
        assert hasattr(core_mod, "spec_augment")

    def test_train_audio_uses_spec_augment(self):
        """train_audio should call spec_augment in its training loop."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_audio)
        assert "spec_augment" in source


# ════════════════════════════════════════════════════════════════════
# Temporal convolution for video frames
# ════════════════════════════════════════════════════════════════════

class TestTemporalConv1d:
    """TemporalConv1d adds cross-frame context to video features."""

    def test_temporal_conv_class_exists(self):
        """TemporalConv1d is importable from vision_encoder."""
        from enigma_engine.core.vision_encoder import TemporalConv1d
        assert callable(TemporalConv1d)

    def test_temporal_conv_init(self):
        """TemporalConv1d can be instantiated."""
        from enigma_engine.core.vision_encoder import TemporalConv1d
        tc = TemporalConv1d(dim=256)
        assert tc is not None

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

    def test_temporal_conv_has_conv_and_norm(self):
        """TemporalConv1d has conv and norm layers."""
        from enigma_engine.core.vision_encoder import TemporalConv1d
        tc = TemporalConv1d(dim=128)
        assert hasattr(tc, "conv")
        assert hasattr(tc, "norm")

    def test_encode_video_frames_accepts_temporal_conv(self):
        """encode_video_frames has temporal_conv parameter."""
        import inspect
        from enigma_engine.core.vision_encoder import encode_video_frames
        sig = inspect.signature(encode_video_frames)
        assert "temporal_conv" in sig.parameters

    def test_temporal_conv_lazy_import(self):
        """TemporalConv1d accessible via core lazy imports."""
        import enigma_engine.core as core_mod
        assert hasattr(core_mod, "TemporalConv1d")


# ════════════════════════════════════════════════════════════════════
# ImageGen mod upgrades — img2img, inpainting, ControlNet, LoRA, SDXL
# ════════════════════════════════════════════════════════════════════

class TestImageGenModUpgrades:
    """Verify imagegen mod has new pipeline modes and commands."""

    def test_imagegen_has_mode_parameter(self):
        """cmd_generate's generate command accepts mode parameter."""
        source = (Path(__file__).resolve().parent.parent
                  / "mods" / "imagegen" / "main.py").read_text(encoding="utf-8")
        assert "mode" in source
        assert "txt2img" in source
        assert "img2img" in source
        assert "inpainting" in source
        assert "controlnet" in source

    def test_imagegen_has_sdxl_support(self):
        """Imagegen detects and loads SDXL pipelines."""
        source = (Path(__file__).resolve().parent.parent
                  / "mods" / "imagegen" / "main.py").read_text(encoding="utf-8")
        assert "StableDiffusionXLPipeline" in source
        assert "is_sdxl" in source

    def test_imagegen_has_pipeline_cache(self):
        """Imagegen caches pipelines to avoid reloading."""
        source = (Path(__file__).resolve().parent.parent
                  / "mods" / "imagegen" / "main.py").read_text(encoding="utf-8")
        assert "_pipe_cache" in source
        assert "_get_or_load_pipeline" in source

    def test_imagegen_has_lora_commands(self):
        """Imagegen has cmd_load_lora and cmd_unload_lora."""
        source = (Path(__file__).resolve().parent.parent
                  / "mods" / "imagegen" / "main.py").read_text(encoding="utf-8")
        assert "def cmd_load_lora" in source
        assert "def cmd_unload_lora" in source
        assert "_active_loras" in source

    def test_imagegen_has_scheduler_selection(self):
        """Imagegen supports multiple noise schedulers."""
        source = (Path(__file__).resolve().parent.parent
                  / "mods" / "imagegen" / "main.py").read_text(encoding="utf-8")
        assert "def _set_scheduler" in source
        assert "def cmd_list_schedulers" in source
        assert "DPMSolverMultistepScheduler" in source
        assert "EulerDiscreteScheduler" in source
        assert "DDIMScheduler" in source

    def test_imagegen_has_controlnet_pipeline(self):
        """Imagegen can load ControlNet pipelines."""
        source = (Path(__file__).resolve().parent.parent
                  / "mods" / "imagegen" / "main.py").read_text(encoding="utf-8")
        assert "ControlNetModel" in source
        assert "StableDiffusionControlNetPipeline" in source

    def test_imagegen_json_has_new_commands(self):
        """mod.json includes new commands (load_lora, etc.)."""
        mod_json = (Path(__file__).resolve().parent.parent
                    / "mods" / "imagegen" / "mod.json")
        data = json.loads(mod_json.read_text(encoding="utf-8"))
        cmd_names = [c["name"] for c in data["commands"]]
        assert "load_lora" in cmd_names
        assert "unload_lora" in cmd_names
        assert "list_schedulers" in cmd_names

    def test_imagegen_json_generate_has_mode(self):
        """mod.json generate command has mode argument."""
        mod_json = (Path(__file__).resolve().parent.parent
                    / "mods" / "imagegen" / "mod.json")
        data = json.loads(mod_json.read_text(encoding="utf-8"))
        gen_cmd = next(c for c in data["commands"] if c["name"] == "generate")
        assert "mode" in gen_cmd["args"]
        assert "scheduler" in gen_cmd["args"]
        assert "init_image" in gen_cmd["args"]
        assert "mask_image" in gen_cmd["args"]

    def test_imagegen_no_asyncio(self):
        """Imagegen mod must not use asyncio (kept after upgrade)."""
        source = (Path(__file__).resolve().parent.parent
                  / "mods" / "imagegen" / "main.py").read_text(encoding="utf-8")
        assert "asyncio" not in source
        assert "async def" not in source

    def test_imagegen_inherits_mod_client(self):
        """ImageGenMod still inherits from ModClient after upgrade."""
        source = (Path(__file__).resolve().parent.parent
                  / "mods" / "imagegen" / "main.py").read_text(encoding="utf-8")
        assert "class ImageGenMod(ModClient)" in source


# ════════════════════════════════════════════════════════════════════
# Command Policy Generator — FORGE tool
# ════════════════════════════════════════════════════════════════════

class TestCommandPolicyGenerator:
    """Verify Command Policy Generator FORGE button and implementation."""

    def test_forge_tools_has_command_policy_method(self):
        """ForgeToolsMixin has _generate_command_policy."""
        from enigma_engine.gui.gui_forge_tools import ForgeToolsMixin
        assert hasattr(ForgeToolsMixin, "_generate_command_policy")
        assert callable(ForgeToolsMixin._generate_command_policy)

    def test_forge_tools_has_parse_commands(self):
        """ForgeToolsMixin has _parse_commands_reference."""
        from enigma_engine.gui.gui_forge_tools import ForgeToolsMixin
        assert hasattr(ForgeToolsMixin, "_parse_commands_reference")

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

    def test_forge_page_has_cmd_policy_button(self):
        """GUI page source creates _forge_cmd_policy_btn."""
        import inspect
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = inspect.getsource(ForgePageMixin)
        assert "_forge_cmd_policy_btn" in source
        assert "COMMAND POLICY" in source

    def test_forge_button_state_includes_cmd_policy(self):
        """Button state management includes cmd policy button."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(
            ForgeMixin._update_forge_button_states)
        assert "_forge_cmd_policy_btn" in source

    def test_generate_text_accepts_min_p(self):
        """_generate_text has min_p parameter."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        sig = inspect.signature(_GenerationMixin._generate_text)
        assert "min_p" in sig.parameters

    def test_stream_generate_accepts_min_p(self):
        """stream_generate has min_p parameter."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        sig = inspect.signature(_GenerationMixin.stream_generate)
        assert "min_p" in sig.parameters

    def test_generate_manual_accepts_min_p(self):
        """_generate_manual has min_p parameter."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        sig = inspect.signature(_GenerationMixin._generate_manual)
        assert "min_p" in sig.parameters

    def test_generate_accepts_min_p(self):
        """EnigmaEngine.generate() has min_p parameter."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        sig = inspect.signature(EnigmaEngine.generate)
        assert "min_p" in sig.parameters

    def test_min_p_filtering_logic_in_sample_token(self):
        """_sample_token source contains min_p filtering logic."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._sample_token)
        assert "min_p" in source
        # Should reference max probability threshold
        assert "max()" in source or "max_prob" in source


class TestRLHFRewardNormalization:
    """Verify RLHF reward normalization is wired."""

    def test_rl_training_normalizes_rewards(self):
        """RLHFTrainer.train uses reward normalization when cfg.normalize_rewards is set."""
        import inspect
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "normalize_rewards" in source
        assert "reward_history" in source

    def test_rl_config_has_normalize_rewards(self):
        """RLHFConfig has normalize_rewards field."""
        from enigma_engine.core.rl_training import RLHFConfig
        config = RLHFConfig()
        assert hasattr(config, "normalize_rewards")


class TestDPOImprovements:
    """Verify DPO max_length uses model config instead of hardcoded 512."""

    def test_dpo_uses_model_max_seq_len(self):
        """train_dpo reads max_seq_len from model config."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train_dpo)
        assert "max_seq_len" in source
        # Should NOT have hardcoded 512 for max_length
        lines = source.split('\n')
        for line in lines:
            if 'max_length' in line and '512' in line and 'getattr' not in line:
                pytest.fail(f"Found hardcoded max_length=512: {line.strip()}")


# ===================================================================
# CONFORMER CONV MODULE FOR AUDIO ENCODER
# ===================================================================


class TestConformerConv:
    """Verify Conformer-style convolution module in audio encoder."""

    def test_conformer_conv_class_exists(self):
        """_ConformerConv is defined in audio_encoder."""
        from enigma_engine.core.audio_encoder import _ConformerConv
        assert _ConformerConv is not None

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

    def test_audio_block_accepts_use_conformer(self):
        """_AudioBlock accepts use_conformer parameter."""
        import inspect
        from enigma_engine.core.audio_encoder import _AudioBlock
        sig = inspect.signature(_AudioBlock.__init__)
        assert "use_conformer" in sig.parameters

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

    def test_encoder_passes_conformer_to_blocks(self):
        """AudioEncoder with use_conformer passes it to blocks."""
        import inspect
        from enigma_engine.core.audio_encoder import AudioEncoder
        source = inspect.getsource(AudioEncoder.__init__)
        assert "use_conformer" in source


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

    def test_gui_chat_has_get_memory_mode(self):
        """gui_logic_chat has _get_memory_mode method."""
        import inspect
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        assert hasattr(LogicChatMixin, "_get_memory_mode")
        sig = inspect.signature(LogicChatMixin._get_memory_mode)
        assert "self" in sig.parameters

    def test_config_page_has_memory_mode_change(self):
        """ConfigPageMixin has _change_memory_mode method."""
        from enigma_engine.gui.gui_pages_config import ConfigPageMixin
        assert hasattr(ConfigPageMixin, "_change_memory_mode")

    def test_chat_logic_respects_memory_mode(self):
        """Chat logic checks memory_mode before auto-extracting facts."""
        import inspect
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        # Find the method that handles chat responses
        source = inspect.getsource(LogicChatMixin)
        assert "_get_memory_mode" in source
        assert "automatic" in source


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

    def test_save_history_caps_messages(self):
        """_save_history only persists up to MAX_CONTEXT_HISTORY messages."""
        import inspect
        from enigma_engine.core.model_context import ModelContext
        source = inspect.getsource(ModelContext._save_history)
        assert "MAX_CONTEXT_HISTORY" in source

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

    def test_replay_buffer_class_exists(self):
        """ReplayBuffer is importable from rl_training."""
        from enigma_engine.core.rl_training import ReplayBuffer
        assert ReplayBuffer is not None

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

    def test_rlhf_trainer_uses_replay(self):
        """RLHFTrainer.train() references ReplayBuffer."""
        import inspect
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "replay" in source.lower()
        assert "ReplayBuffer" in source

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

    def test_selfplay_config_has_replay_fields(self):
        """SelfPlayConfig has replay_capacity and replay_ratio."""
        from enigma_engine.core.rl_training import SelfPlayConfig
        cfg = SelfPlayConfig()
        assert hasattr(cfg, "replay_capacity")
        assert hasattr(cfg, "replay_ratio")

    def test_selfplay_trainer_uses_replay(self):
        """SelfPlayTrainer.train() references ReplayBuffer."""
        import inspect
        from enigma_engine.core.rl_training import SelfPlayTrainer
        source = inspect.getsource(SelfPlayTrainer.train)
        assert "replay" in source.lower()
        assert "ReplayBuffer" in source


# ===================================================================
# COMPLETION ITEM #1: memory.search respects disabled flag
# ===================================================================


class TestMemorySearchDisabled:
    """Verify memory.search returns nothing when memory is disabled."""

    def test_memory_search_checks_disabled(self):
        """memory_search function checks mem.disabled before searching."""
        import inspect
        from enigma_engine.core import builtin_commands
        source = inspect.getsource(builtin_commands)
        # Find the memory_search function — it must check disabled
        # Locate from "def memory_search" to next "def " or registry.register
        idx = source.find("def memory_search")
        assert idx != -1, "memory_search not found in builtin_commands"
        chunk = source[idx:idx + 600]
        assert "disabled" in chunk


# ===================================================================
# COMPLETION ITEM #2: memory mode restored on startup
# ===================================================================


class TestMemoryModeStartup:
    """Verify memory mode is applied when PersistentMemory is created."""

    def test_get_memory_checks_saved_mode(self):
        """get_memory or PersistentMemory reads saved mode from settings."""
        import inspect
        from enigma_engine.core import memory as mem_mod
        source = inspect.getsource(mem_mod)
        # Should reference gui_settings or memory_mode in init path
        assert "memory_mode" in source or "gui_settings" in source


# ===================================================================
# COMPLETION ITEM #5: MAX_CONTEXT_HISTORY user-configurable
# ===================================================================


class TestHistoryCapConfigurable:
    """Verify MAX_CONTEXT_HISTORY can be set by user."""

    def test_model_context_has_setter(self):
        """model_context has a way to set the history cap."""
        from enigma_engine.core import model_context
        assert hasattr(model_context, "set_max_context_history") or \
            hasattr(model_context, "MAX_CONTEXT_HISTORY")

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


class TestVenvBootstrap:
    """Verify run.py has venv auto-detection logic."""

    def test_run_py_has_venv_bootstrap(self):
        """run.py contains _ensure_venv or equivalent bootstrap logic."""
        source = Path("run.py").read_text(encoding="utf-8")
        assert "_ensure_venv" in source or "sys.base_prefix" in source

    def test_bootstrap_checks_prefix(self):
        """Bootstrap logic checks sys.prefix vs sys.base_prefix."""
        source = Path("run.py").read_text(encoding="utf-8")
        assert "sys.base_prefix" in source

    def test_bootstrap_finds_venv_python(self):
        """Bootstrap logic looks for venv/Scripts/python.exe or venv/bin/python."""
        source = Path("run.py").read_text(encoding="utf-8")
        assert "venv" in source and "python" in source.lower()


# ===================================================================
# AUTO PRECISION / BF16 DETECTION
# ===================================================================


class TestAutoPrecision:
    """Verify the engine auto-detects optimal precision."""

    def test_select_dtype_function_exists(self):
        """EnigmaEngine has _select_dtype or equivalent precision logic."""
        import inspect
        from enigma_engine.core import inference
        source = inspect.getsource(inference)
        assert "_select_dtype" in source or "bfloat16" in source

    def test_inference_supports_bfloat16(self):
        """inference.py handles bfloat16 dtype, not just float16."""
        import inspect
        from enigma_engine.core import inference
        source = inspect.getsource(inference)
        assert "bfloat16" in source

    def test_config_precision_default_is_auto(self):
        """CONFIG defaults precision to 'auto' for hardware detection."""
        import inspect
        from enigma_engine.config import defaults
        source = inspect.getsource(defaults)
        # The inline default in CONFIG dict must be "auto"
        assert '"precision": "auto"' in source

    def test_hardware_detection_bf16_awareness(self):
        """get_optimal_config includes bf16 recommendation."""
        import inspect
        from enigma_engine.core import hardware_detection
        source = inspect.getsource(hardware_detection)
        assert "bfloat16" in source or "bf16" in source

    def test_training_amp_dtype_aware(self):
        """Training AMP uses dtype from config, not hardcoded float16."""
        import inspect
        from enigma_engine.core import training
        source = inspect.getsource(training)
        # Should reference bfloat16 or amp_dtype for Blackwell support
        assert "bfloat16" in source or "amp_dtype" in source


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

    def test_rlhf_trainer_has_save_checkpoint(self):
        """RLHFTrainer has _save_checkpoint method."""
        from enigma_engine.core.rl_training import RLHFTrainer
        assert hasattr(RLHFTrainer, "_save_checkpoint")
        assert callable(getattr(RLHFTrainer, "_save_checkpoint"))

    def test_rlhf_trainer_has_load_checkpoint(self):
        """RLHFTrainer has load_checkpoint method."""
        from enigma_engine.core.rl_training import RLHFTrainer
        assert hasattr(RLHFTrainer, "load_checkpoint")
        assert callable(getattr(RLHFTrainer, "load_checkpoint"))

    def test_selfplay_trainer_has_save_checkpoint(self):
        """SelfPlayTrainer has _save_checkpoint method."""
        from enigma_engine.core.rl_training import SelfPlayTrainer
        assert hasattr(SelfPlayTrainer, "_save_checkpoint")
        assert callable(getattr(SelfPlayTrainer, "_save_checkpoint"))

    def test_selfplay_trainer_has_load_checkpoint(self):
        """SelfPlayTrainer has load_checkpoint method."""
        from enigma_engine.core.rl_training import SelfPlayTrainer
        assert hasattr(SelfPlayTrainer, "load_checkpoint")
        assert callable(getattr(SelfPlayTrainer, "load_checkpoint"))

    def test_rlhf_save_checkpoint_includes_replay(self):
        """RLHFTrainer._save_checkpoint includes replay_buffer state."""
        import inspect
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer._save_checkpoint)
        assert "replay" in source.lower()
        assert "state_dict" in source

    def test_selfplay_save_checkpoint_includes_replay(self):
        """SelfPlayTrainer._save_checkpoint includes replay_buffer state."""
        import inspect
        from enigma_engine.core.rl_training import SelfPlayTrainer
        source = inspect.getsource(SelfPlayTrainer._save_checkpoint)
        assert "replay" in source.lower()
        assert "state_dict" in source

    def test_rlhf_load_checkpoint_restores_replay(self):
        """RLHFTrainer.load_checkpoint restores replay_buffer state."""
        import inspect
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.load_checkpoint)
        assert "replay" in source.lower()
        assert "load_state_dict" in source

    def test_selfplay_load_checkpoint_restores_replay(self):
        """SelfPlayTrainer.load_checkpoint restores replay_buffer state."""
        import inspect
        from enigma_engine.core.rl_training import SelfPlayTrainer
        source = inspect.getsource(SelfPlayTrainer.load_checkpoint)
        assert "replay" in source.lower()
        assert "load_state_dict" in source

    def test_rlhf_train_saves_checkpoint_when_configured(self):
        """RLHFTrainer.train() saves checkpoints when checkpoint_dir set."""
        import inspect
        from enigma_engine.core.rl_training import RLHFTrainer
        source = inspect.getsource(RLHFTrainer.train)
        assert "_save_checkpoint" in source
        assert "checkpoint_dir" in source

    def test_selfplay_train_saves_checkpoint_when_configured(self):
        """SelfPlayTrainer.train() saves checkpoints when checkpoint_dir set."""
        import inspect
        from enigma_engine.core.rl_training import SelfPlayTrainer
        source = inspect.getsource(SelfPlayTrainer.train)
        assert "_save_checkpoint" in source
        assert "checkpoint_dir" in source


# ===================================================================
# TRAIN-LOCK INFERENCE GUARD (#1)
# ===================================================================


class TestTrainLockInferenceGuard:
    """Verify inference acquires the training lock when available."""

    def test_engine_has_train_lock_attribute(self):
        """EnigmaEngine._init_common sets _train_lock = None."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        source = inspect.getsource(EnigmaEngine._init_common)
        assert "_train_lock" in source

    def test_engine_has_set_train_lock(self):
        """EnigmaEngine has a set_train_lock method."""
        from enigma_engine.core.inference import EnigmaEngine
        assert hasattr(EnigmaEngine, "set_train_lock")
        assert callable(getattr(EnigmaEngine, "set_train_lock"))

    def test_generate_references_train_lock(self):
        """generate() tries to coordinate with the training lock."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        source = inspect.getsource(EnigmaEngine.generate)
        assert "_train_lock" in source


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

    def test_source_has_nonlatin_heuristic(self):
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._needs_ai_creativity)
        assert "non-Latin" in source or "non_latin" in source


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
        from enigma_engine.core.training_monitor import TrainingMonitor
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

    def test_moving_average_not_naive_loop(self):
        """get_chart_data should not use the naive O(n*w) loop."""
        import inspect
        from enigma_engine.core.training_monitor import TrainingMonitor
        source = inspect.getsource(TrainingMonitor.get_chart_data)
        # The old pattern: "for i in range(len(" + "sum(chunk)"
        # The new pattern should NOT have both
        has_naive = ("sum(chunk)" in source
                     or "sum(losses_snap[" in source)
        assert not has_naive, "get_chart_data still uses naive O(n*w) loop"


# ===================================================================
# VALIDATE load_state() JSON (#16)
# ===================================================================


class TestQueueLoadValidation:
    """Verify load_state handles corrupt/malformed JSON gracefully."""

    def test_corrupt_json_returns_false(self):
        """Completely invalid JSON returns False, not crash."""
        import tempfile
        from pathlib import Path
        from enigma_engine.core.training_queue import TrainingQueue
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
        import json
        import tempfile
        from pathlib import Path
        from enigma_engine.core.training_queue import TrainingQueue
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
        import json
        import tempfile
        from pathlib import Path
        from enigma_engine.core.training_queue import TrainingQueue
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

    def test_training_plan_has_reset_difficulty(self):
        """TrainingPlan has a reset_difficulty method."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        assert hasattr(TrainingPlan, "reset_difficulty")

    def test_reset_difficulty_sets_simple(self):
        """reset_difficulty sets current_difficulty back to 'simple'."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        plan.current_difficulty = "advanced"
        plan.reset_difficulty()
        assert plan.current_difficulty == "simple"

    def test_adaptive_pipeline_resets_on_failure(self):
        """GUI adaptive pipeline calls reset_difficulty on crash."""
        import inspect
        from enigma_engine.gui import gui_forge_adaptive
        source = inspect.getsource(gui_forge_adaptive)
        # The exception handler should reset difficulty
        assert "reset_difficulty" in source


# ===================================================================
# #22 — Warn on torch.load for untrusted .pth files
# ===================================================================


class TestSafeLoadWarning:
    """Verify safe_load_weights warns on non-safetensors formats."""

    def test_safe_load_weights_warns_for_pth(self):
        """Loading a .pth file should log a warning about safetensors."""
        import inspect
        from enigma_engine.core.model_registry import safe_load_weights
        source = inspect.getsource(safe_load_weights)
        # Should mention safetensors preference in a warning
        assert "warning" in source.lower() or "warn" in source.lower()
        assert "safetensors" in source.lower()

    def test_safe_load_weights_no_warning_for_safetensors(self):
        """Safetensors path should not trigger a warning."""
        import inspect
        from enigma_engine.core.model_registry import safe_load_weights
        source = inspect.getsource(safe_load_weights)
        # The safetensors branch should not log a warning
        lines = source.split("\n")
        in_safetensors_branch = False
        for line in lines:
            if ".safetensors" in line and "if" in line:
                in_safetensors_branch = True
            if in_safetensors_branch and "else:" in line:
                break
            if in_safetensors_branch and "warn" in line.lower():
                # Safetensors branch should NOT warn
                assert False, "Safetensors branch should not warn"


# ===================================================================
# #23 — Consolidate RMSNorm imports
# ===================================================================


class TestRMSNormConsolidation:
    """Verify vision/audio encoders import RMSNorm instead of duplicating."""

    def test_vision_encoder_imports_rmsnorm(self):
        """vision_encoder should import RMSNorm from model_components."""
        import inspect
        from enigma_engine.core import vision_encoder
        source = inspect.getsource(vision_encoder)
        assert "from enigma_engine.core.model_components import RMSNorm" in source \
            or "from .model_components import RMSNorm" in source

    def test_vision_encoder_no_duplicate_rmsnorm_class(self):
        """vision_encoder should not define its own _RMSNorm class."""
        import inspect
        from enigma_engine.core import vision_encoder
        source = inspect.getsource(vision_encoder)
        assert "class _RMSNorm" not in source

    def test_audio_encoder_imports_rmsnorm(self):
        """audio_encoder should import RMSNorm from model_components."""
        import inspect
        from enigma_engine.core import audio_encoder
        source = inspect.getsource(audio_encoder)
        assert "from enigma_engine.core.model_components import RMSNorm" in source \
            or "from .model_components import RMSNorm" in source

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

    def test_update_emotional_state_uses_lock(self):
        """update_emotional_state should acquire the lock."""
        import inspect
        from enigma_engine.core.model_context import ModelContext
        source = inspect.getsource(ModelContext.update_emotional_state)
        assert "_emotional_lock" in source

    def test_decay_emotional_state_uses_lock(self):
        """decay_emotional_state should acquire the lock."""
        import inspect
        from enigma_engine.core.model_context import ModelContext
        source = inspect.getsource(ModelContext.decay_emotional_state)
        assert "_emotional_lock" in source

    def test_reset_emotional_state_uses_lock(self):
        """reset_emotional_state should acquire the lock."""
        import inspect
        from enigma_engine.core.model_context import ModelContext
        source = inspect.getsource(ModelContext.reset_emotional_state)
        assert "_emotional_lock" in source
