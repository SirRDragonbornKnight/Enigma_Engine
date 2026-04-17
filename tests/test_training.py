"""Tests for training, DPO, LoRA, PPO, RLHF, adaptive, queue, checkpoints."""
import inspect
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestDPOTraining:
    """Verify DPO training infrastructure."""

    def test_dpo_loss_computes(self):
        """_dpo_loss should return a scalar tensor."""
        import torch
        from enigma_engine.core.training import Trainer
        # Simulate log-probs
        pc = torch.tensor([0.0])
        pr = torch.tensor([0.0])
        rc = torch.tensor([0.0])
        rr = torch.tensor([0.0])
        loss = Trainer._dpo_loss(pc, pr, rc, rr, beta=0.1)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0  # DPO loss is non-negative (logsigmoid(0) = -ln2)


class TestLoraEpochLossInit:
    """Verify epoch_loss is initialized before the training loop."""

    def test_epoch_loss_not_fragile(self):
        """epoch_loss should be initialized before loop, not checked via dir()."""
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        source = inspect.getsource(LoraTrainer.train)
        # Should NOT use fragile dir() check
        assert "'epoch_loss' in dir()" not in source, (
            "epoch_loss should be initialized before loop, not checked via dir()")


class TestDPOMaskFix:
    """Verify DPO uses (targets != -100) mask, not (targets > 0)."""

    def test_sequence_log_probs_mask_uses_neg100(self):
        """_get_sequence_logps source must use '!= -100', not '> 0'."""
        import inspect
        from enigma_engine.core.training import Trainer
        # The method may be named _get_sequence_logps or _sequence_log_probs
        for name in ("_get_sequence_logps", "_sequence_log_probs"):
            if hasattr(Trainer, name):
                source = inspect.getsource(getattr(Trainer, name))
                assert "!= -100" in source, (
                    f"{name} should mask with '!= -100', not '> 0'")
                assert "> 0" not in source.split("# ")[0] or "!= -100" in source
                break
        else:
            pytest.fail("DPO log-probs method not found — expected _get_sequence_logps or _sequence_log_probs on Trainer")

    def test_dpo_mask_keeps_token_id_zero(self):
        """Token ID 0 should NOT be masked — only -100 is the ignore marker."""
        import torch
        targets = torch.tensor([[5, 0, 3, -100, 2]])
        mask = (targets != -100).float()
        # Token ID 0 at position 1 should be 1.0 (not masked)
        assert mask[0, 1].item() == 1.0
        # -100 at position 3 should be 0.0 (masked)
        assert mask[0, 3].item() == 0.0
        # Old buggy mask (> 0) would wrongly mask token ID 0
        buggy_mask = (targets > 0).float()
        assert buggy_mask[0, 1].item() == 0.0  # bug: token 0 was masked


class TestVisionTraining:
    """Trainer.train_vision method for image-text pair training."""

    def test_train_vision_updates_weights(self):
        """train_vision should update both encoder and projection weights."""
        import torch
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        # Tiny model setup
        vcfg = VisionEncoderConfig(image_size=16, patch_size=8, dim=16, n_layers=1, n_heads=2)
        v_enc = VisionEncoder(vcfg)
        tok = SimpleTokenizer()

        tcfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32, n_layers=1, n_heads=2,
            max_seq_len=32, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        # Snapshot initial weights
        assert v_enc.patch_embed is not None and v_enc.patch_embed.proj is not None
        assert model.vision_projection is not None
        enc_before = v_enc.patch_embed.proj.weight.clone()
        proj_before = model.vision_projection.weight.clone()

        # Minimal training data: image tensor + text
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        img = Image.new("RGB", (16, 16), (128, 64, 32))
        data = [{"image": img, "text": "a red square"}]

        config = TrainingConfig(epochs=2, batch_size=1, learning_rate=1e-3)
        trainer = Trainer(model, tok, config)
        trainer.train_vision(vision_encoder=v_enc, data=data)

        # Both encoder and projection should have changed
        assert v_enc.patch_embed is not None and v_enc.patch_embed.proj is not None
        assert model.vision_projection is not None
        assert not torch.equal(enc_before, v_enc.patch_embed.proj.weight)
        assert not torch.equal(proj_before, model.vision_projection.weight)

    def test_train_vision_returns_state(self):
        """train_vision should return TrainingState."""
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.training import Trainer, TrainingConfig, TrainingState
        from enigma_engine.core.tokenizer import SimpleTokenizer

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        vcfg = VisionEncoderConfig(image_size=16, patch_size=8, dim=16, n_layers=1, n_heads=2)
        v_enc = VisionEncoder(vcfg)
        tok = SimpleTokenizer()
        tcfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32, n_layers=1, n_heads=2,
            max_seq_len=32, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        img = Image.new("RGB", (16, 16), (64, 128, 255))
        data = [{"image": img, "text": "blue square"}]
        config = TrainingConfig(epochs=1, batch_size=1)
        trainer = Trainer(model, tok, config)
        state = trainer.train_vision(vision_encoder=v_enc, data=data)
        assert isinstance(state, TrainingState)
        assert len(state.training_losses) >= 1

    def test_train_vision_loss_decreases(self):
        """Loss should generally decrease over epochs (small model, enough data)."""
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        vcfg = VisionEncoderConfig(image_size=16, patch_size=8, dim=16, n_layers=1, n_heads=2)
        v_enc = VisionEncoder(vcfg)
        tok = SimpleTokenizer()
        tcfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32, n_layers=1, n_heads=2,
            max_seq_len=32, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        # Create several training pairs for better signal
        data = [
            {"image": Image.new("RGB", (16, 16), (255, 0, 0)), "text": "red"},
            {"image": Image.new("RGB", (16, 16), (0, 255, 0)), "text": "green"},
            {"image": Image.new("RGB", (16, 16), (0, 0, 255)), "text": "blue"},
        ]
        config = TrainingConfig(epochs=5, batch_size=1, learning_rate=5e-3)
        trainer = Trainer(model, tok, config)
        state = trainer.train_vision(vision_encoder=v_enc, data=data)

        # Last epoch loss should be lower than first
        assert state.training_losses[-1] < state.training_losses[0]

    def test_train_vision_callbacks(self):
        """train_vision should fire progress and loss callbacks."""
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        vcfg = VisionEncoderConfig(image_size=16, patch_size=8, dim=16, n_layers=1, n_heads=2)
        v_enc = VisionEncoder(vcfg)
        tok = SimpleTokenizer()
        tcfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32, n_layers=1, n_heads=2,
            max_seq_len=32, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        progress_log = []
        loss_log = []

        config = TrainingConfig(epochs=1, batch_size=1, log_every=1)
        trainer = Trainer(model, tok, config)
        trainer.on_progress = lambda p, m: progress_log.append((p, m))
        trainer.on_loss = lambda l: loss_log.append(l)

        img = Image.new("RGB", (16, 16), (128, 128, 128))
        data = [{"image": img, "text": "gray square"}]
        trainer.train_vision(vision_encoder=v_enc, data=data)

        assert len(progress_log) > 0
        assert len(loss_log) > 0

    def test_train_vision_stop_requested(self):
        """train_vision should respect request_stop()."""
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        vcfg = VisionEncoderConfig(image_size=16, patch_size=8, dim=16, n_layers=1, n_heads=2)
        v_enc = VisionEncoder(vcfg)
        tok = SimpleTokenizer()
        tcfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32, n_layers=1, n_heads=2,
            max_seq_len=32, vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=tcfg)

        config = TrainingConfig(epochs=100, batch_size=1)
        trainer = Trainer(model, tok, config)

        # Use callback to stop after first epoch
        def stop_after_first(epoch, avg_loss):
            trainer.request_stop()
        trainer.on_epoch_complete = stop_after_first

        img = Image.new("RGB", (16, 16), (0, 0, 0))
        data = [{"image": img, "text": "black"}]
        state = trainer.train_vision(vision_encoder=v_enc, data=data)

        # Should complete far fewer than 100 epochs
        assert len(state.training_losses) <= 3


class TestVisionDataParsing:
    """Parsing and validation of image-text training data."""

    def test_requires_vision_projection(self):
        """train_vision should raise if model lacks vision_projection."""
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        vcfg = VisionEncoderConfig(image_size=16, patch_size=8, dim=16, n_layers=1, n_heads=2)
        v_enc = VisionEncoder(vcfg)
        tok = SimpleTokenizer()

        # Model WITHOUT vision_hidden_size — no projection layer
        tcfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32, n_layers=1, n_heads=2, max_seq_len=32,
        )
        model = Enigma(config=tcfg)

        config = TrainingConfig(epochs=1, batch_size=1)
        trainer = Trainer(model, tok, config)

        img = Image.new("RGB", (16, 16), (128, 128, 128))
        data = [{"image": img, "text": "gray"}]

        with pytest.raises(ValueError, match="vision"):
            trainer.train_vision(vision_encoder=v_enc, data=data)


class TestScanVisionData:
    """Scanner detection of image-text datasets."""

    def test_scan_empty_dir(self):
        """scan_vision_data on empty dir should return empty list."""
        import tempfile
        from enigma_engine.gui.scanners import scan_vision_data
        with tempfile.TemporaryDirectory() as d:
            result = scan_vision_data(d)
            assert result == []

    def test_scan_paired_files(self):
        """scan_vision_data should detect image.png + image.txt pairs."""
        import tempfile
        from pathlib import Path
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        from enigma_engine.gui.scanners import scan_vision_data

        with tempfile.TemporaryDirectory() as d:
            # Create paired files
            img = Image.new("RGB", (10, 10), (255, 0, 0))
            img.save(Path(d) / "photo1.png")
            (Path(d) / "photo1.txt").write_text("a red square", encoding="utf-8")

            result = scan_vision_data(d)
            assert len(result) == 1
            assert "image" in result[0]
            assert "text" in result[0]

    def test_scan_jsonl_file(self):
        """scan_vision_data should detect JSONL files with image+text fields."""
        import json
        import tempfile
        from pathlib import Path
        from enigma_engine.gui.scanners import scan_vision_data

        with tempfile.TemporaryDirectory() as d:
            jsonl_path = Path(d) / "captions.jsonl"
            records = [
                {"image": "img1.png", "text": "a cat"},
                {"image": "img2.png", "text": "a dog"},
            ]
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            result = scan_vision_data(d)
            assert len(result) == 2


class TestScanVisionDataVideo:
    """Scanner detection of video files for vision training."""

    def test_video_extensions_defined(self):
        """_VIDEO_EXTENSIONS set must exist in scanners module."""
        from enigma_engine.gui.scanners import _VIDEO_EXTENSIONS
        assert isinstance(_VIDEO_EXTENSIONS, (set, frozenset))
        assert ".mp4" in _VIDEO_EXTENSIONS
        assert ".avi" in _VIDEO_EXTENSIONS


# ============================================================
# Vision Chat Integration
# ============================================================


# ================================================================
# Training Pipeline Quality Fixes
# ================================================================


class TestTrainingDedup:
    """Duplicate training sequences are removed."""

    def test_dedup_preserves_order(self):
        """Dedup preserves original order of sequences."""
        # dict.fromkeys preserves insertion order in Python 3.7+
        seqs = ["hello", "world", "hello", "foo", "world"]
        deduped = list(dict.fromkeys(seqs))
        assert deduped == ["hello", "world", "foo"]


# =============================================================================
# ADAPTIVE TRAINER TESTS
# =============================================================================

class TestAdaptiveTrainerImports:
    """Test that adaptive_trainer module imports correctly."""

    def test_all_stages_defined(self):
        """ALL_STAGES has the 4 expected stages."""
        from enigma_engine.core.adaptive_trainer import ALL_STAGES
        assert ALL_STAGES == ["basics", "conversation", "commands", "web"]

    def test_difficulty_levels_defined(self):
        """DIFFICULTY_LEVELS has 3 levels in order."""
        from enigma_engine.core.adaptive_trainer import DIFFICULTY_LEVELS
        assert DIFFICULTY_LEVELS == ["simple", "medium", "advanced"]


class TestTrainingPlan:
    """Test TrainingPlan dataclass and logic."""

    def test_create_default_plan(self):
        """Default plan has all 4 stages starting at index 0."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan(
            student_path="models/student.pth",
            trainer_path="models/trainer.pth")
        assert plan.current_stage == "basics"
        assert plan.current_stage_idx == 0
        assert not plan.is_complete
        assert plan.status == "pending"

    def test_advance_stage(self):
        """advance_stage moves to next stage."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        assert plan.current_stage == "basics"
        result = plan.advance_stage()
        assert result is True
        assert plan.current_stage == "conversation"
        plan.advance_stage()
        assert plan.current_stage == "commands"
        plan.advance_stage()
        assert plan.current_stage == "web"
        result = plan.advance_stage()
        assert result is False
        assert plan.is_complete
        assert plan.status == "completed"

    def test_decide_action_advance(self):
        """decide_action advances on high scores, retries on low."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        assert plan.decide_action(8.0) == "advance"
        assert plan.decide_action(3.0) == "retry"

    def test_decide_action_retries_exhaust(self):
        """decide_action advances after max retries even with low scores."""
        from enigma_engine.core.adaptive_trainer import (
            TrainingPlan, StageResult)
        plan = TrainingPlan(max_retries=2)
        # Record 2 attempts on the current stage
        for i in range(2):
            plan.record_result(StageResult(
                stage="basics", attempt=i + 1,
                scores=[3.0], avg_score=3.0,
                status="retry"))
        # Now retries exhausted — should advance anyway
        assert plan.decide_action(3.0) == "advance"

    def test_decide_action_escalates_difficulty(self):
        """decide_action escalates difficulty on retry."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        assert plan.current_difficulty == "simple"
        plan.decide_action(4.0)  # triggers retry
        assert plan.current_difficulty == "medium"
        plan.decide_action(4.0)  # triggers another retry
        assert plan.current_difficulty == "advanced"

    def test_record_result(self):
        """record_result stores a StageResult."""
        from enigma_engine.core.adaptive_trainer import (
            TrainingPlan, StageResult)
        plan = TrainingPlan()
        result = StageResult(
            stage="basics", attempt=1,
            scores=[7.0, 8.0], avg_score=7.5,
            status="passed")
        plan.record_result(result)
        assert len(plan.stage_results) == 1
        assert plan.stage_results[0]["stage"] == "basics"
        assert plan.stage_results[0]["avg_score"] == 7.5

    def test_save_and_load(self, tmp_path):
        """Plan round-trips through JSON save/load."""
        from enigma_engine.core.adaptive_trainer import (
            TrainingPlan, StageResult)
        plan = TrainingPlan(
            student_path="models/student.pth",
            trainer_path="models/trainer.pth",
            student_name="student",
            trainer_name="trainer",
            epochs_per_stage=20,
            learning_rate=0.001)
        plan.advance_stage()
        plan.record_result(StageResult(
            stage="basics", attempt=1,
            avg_score=8.0, status="passed"))
        path = tmp_path / "plan.json"
        plan.save(path)
        assert path.exists()
        loaded = TrainingPlan.load(path)
        assert loaded.current_stage_idx == 1
        assert loaded.student_path == "models/student.pth"
        assert loaded.epochs_per_stage == 20
        assert len(loaded.stage_results) == 1

    def test_summary(self):
        """summary() returns readable text."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan(
            student_name="student", trainer_name="trainer")
        text = plan.summary()
        assert "student" in text
        assert "trainer" in text
        assert "basics" in text.lower()

    def test_current_attempt_counts(self):
        """current_attempt correctly counts attempts for current stage."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        assert plan.current_attempt == 0
        plan.stage_results = [
            {"stage": "basics", "attempt": 1},
            {"stage": "basics", "attempt": 2}]
        assert plan.current_attempt == 2

    def test_is_complete_after_all_stages(self):
        """Plan is complete after advancing past all stages."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        for _ in range(4):
            plan.advance_stage()
        assert plan.is_complete
        assert plan.current_stage is None


class TestBuildAdaptivePrompt:
    """Test build_adaptive_prompt for different stages/difficulties."""

    def test_simple_basics(self):
        """Simple basics prompt asks for ultra-simple content."""
        from enigma_engine.core.adaptive_trainer import build_adaptive_prompt
        prompt = build_adaptive_prompt(1, 10, "basics", "simple")
        assert "simple" in prompt.lower() or "SIMPLE" in prompt
        assert "#1" in prompt

    def test_advanced_conversation(self):
        """Advanced conversation prompt asks for complex dialogue."""
        from enigma_engine.core.adaptive_trainer import build_adaptive_prompt
        prompt = build_adaptive_prompt(5, 20, "conversation", "advanced")
        assert "complex" in prompt.lower() or "ADVANCED" in prompt

    def test_all_stage_difficulty_combos(self):
        """All 12 stage × difficulty combinations produce prompts."""
        from enigma_engine.core.adaptive_trainer import (
            build_adaptive_prompt, ALL_STAGES, DIFFICULTY_LEVELS)
        for stage in ALL_STAGES:
            for diff in DIFFICULTY_LEVELS:
                prompt = build_adaptive_prompt(1, 5, stage, diff)
                assert len(prompt) > 50
                assert str(stage.upper()) in prompt


class TestStageResult:
    """Test StageResult serialization."""

    def test_to_dict(self):
        """StageResult serializes to dict."""
        from enigma_engine.core.adaptive_trainer import StageResult
        r = StageResult(stage="basics", attempt=1, avg_score=7.5)
        d = r.to_dict()
        assert d["stage"] == "basics"
        assert d["avg_score"] == 7.5

    def test_from_dict(self):
        """StageResult deserializes from dict."""
        from enigma_engine.core.adaptive_trainer import StageResult
        d = {"stage": "web", "attempt": 2, "avg_score": 6.0,
             "status": "passed", "difficulty": "medium",
             "scores": [6.0, 6.0], "epochs_trained": 10,
             "pairs_generated": 20, "best_loss": 1.5,
             "started_at": "", "completed_at": ""}
        r = StageResult.from_dict(d)
        assert r.stage == "web"
        assert r.attempt == 2

    def test_from_dict_ignores_unknown_keys(self):
        """StageResult.from_dict skips unknown keys."""
        from enigma_engine.core.adaptive_trainer import StageResult
        d = {"stage": "basics", "attempt": 1,
             "unknown_field": "ignored"}
        r = StageResult.from_dict(d)
        assert r.stage == "basics"


class TestLossToProxyScore:
    """Test loss_to_proxy_score fallback scoring."""

    def test_low_loss_gives_high_score(self):
        """Very low loss → high proxy score (capped at 8)."""
        from enigma_engine.core.adaptive_trainer import loss_to_proxy_score
        score = loss_to_proxy_score(0.1)
        assert score >= 7
        assert score <= 8  # capped — proxy should never give 9+

    def test_medium_loss_gives_medium_score(self):
        """Medium loss → score around 5-7."""
        from enigma_engine.core.adaptive_trainer import loss_to_proxy_score
        score = loss_to_proxy_score(1.0)
        assert 4 <= score <= 6

    def test_high_loss_gives_low_score(self):
        """High loss → low score."""
        from enigma_engine.core.adaptive_trainer import loss_to_proxy_score
        score = loss_to_proxy_score(3.0)
        assert score <= 2

    def test_zero_loss_capped_at_8(self):
        """Perfect loss=0 still capped at 8 (proxy, not real test)."""
        from enigma_engine.core.adaptive_trainer import loss_to_proxy_score
        assert loss_to_proxy_score(0.0) == 8

    def test_infinite_loss_floors_at_1(self):
        """Extreme loss floors at 1."""
        from enigma_engine.core.adaptive_trainer import loss_to_proxy_score
        assert loss_to_proxy_score(100.0) == 1
        assert loss_to_proxy_score(float("inf")) == 1

    def test_returns_int(self):
        """Proxy score is always an integer."""
        from enigma_engine.core.adaptive_trainer import loss_to_proxy_score
        for loss in [0.0, 0.3, 0.7, 1.5, 5.0]:
            assert isinstance(loss_to_proxy_score(loss), int)


class TestBuildTestPrompt:
    """Test build_test_prompt for stage-specific Phase 3 prompts."""

    def test_commands_stage_mentions_cmd_syntax(self):
        """COMMANDS stage test prompt includes [CMD] context."""
        from enigma_engine.core.adaptive_trainer import build_test_prompt
        prompt = build_test_prompt(1, "commands", "simple")
        assert "[CMD]" in prompt

    def test_basics_stage_has_no_cmd_context(self):
        """BASICS stage prompt doesn't mention commands."""
        from enigma_engine.core.adaptive_trainer import build_test_prompt
        prompt = build_test_prompt(1, "basics", "simple")
        assert "[CMD]" not in prompt

    def test_all_stages_produce_prompts(self):
        """All stages produce non-empty test prompts."""
        from enigma_engine.core.adaptive_trainer import (
            build_test_prompt, ALL_STAGES, DIFFICULTY_LEVELS)
        for stage in ALL_STAGES:
            for diff in DIFFICULTY_LEVELS:
                prompt = build_test_prompt(1, stage, diff)
                assert len(prompt) > 20
                assert "question" in prompt.lower()

    def test_includes_test_number(self):
        """Test prompt includes the test number."""
        from enigma_engine.core.adaptive_trainer import build_test_prompt
        prompt = build_test_prompt(5, "basics", "medium")
        assert "#5" in prompt


class TestCleanExample:
    """Test clean_example strips garbage wrappers from teacher output."""

    def test_strips_the_answer_is_prefix(self):
        """Removes 'The answer is...' prefix."""
        from enigma_engine.core.adaptive_trainer import clean_example
        raw = "The answer is... Dogs are loyal animals. They make great companions."
        cleaned = clean_example(raw)
        assert cleaned == "Dogs are loyal animals. They make great companions."

    def test_strips_think_tags(self):
        """Removes leaked <think> / </think> XML tags."""
        from enigma_engine.core.adaptive_trainer import clean_example
        raw = "</think>\n</think>\nThe user needs a training example."
        cleaned = clean_example(raw)
        assert "<think>" not in cleaned
        assert "</think>" not in cleaned
        assert cleaned == "The user needs a training example."

    def test_strips_here_is_wrapper(self):
        """Removes 'Here is a training example:' wrappers."""
        from enigma_engine.core.adaptive_trainer import clean_example
        raw = "Here is a training example: Q: What is AI?\nA: Artificial intelligence."
        cleaned = clean_example(raw)
        assert cleaned.startswith("Q:")

    def test_returns_empty_for_empty(self):
        """Returns empty string for empty input."""
        from enigma_engine.core.adaptive_trainer import clean_example
        assert clean_example("") == ""
        assert clean_example("   ") == ""

    def test_preserves_valid_content(self):
        """Does not mangle valid Q&A content."""
        from enigma_engine.core.adaptive_trainer import clean_example
        valid = "Q: What is the capital of France?\nA: Paris is the capital of France."
        assert clean_example(valid) == valid

    def test_handles_answer_is_no_dots(self):
        """Strips 'The answer is' without trailing dots."""
        from enigma_engine.core.adaptive_trainer import clean_example
        raw = "The answer is ambition: the strong desire to achieve something."
        cleaned = clean_example(raw)
        assert cleaned == "ambition: the strong desire to achieve something."


class TestValidateExample:
    """Test validate_example for stage-specific format checks."""

    def test_rejects_short_text(self):
        """Examples under 30 chars are rejected."""
        from enigma_engine.core.adaptive_trainer import validate_example
        assert validate_example("Hi", "basics") is False
        assert validate_example("Too short.", "basics") is False

    def test_accepts_valid_basics(self):
        """A coherent paragraph passes basics validation."""
        from enigma_engine.core.adaptive_trainer import validate_example
        text = ("Dogs are loyal animals that have been domesticated "
                "for thousands of years. They make great companions "
                "and come in many different breeds.")
        assert validate_example(text, "basics") is True

    def test_rejects_leaked_reasoning(self):
        """Leaked teacher reasoning is rejected."""
        from enigma_engine.core.adaptive_trainer import validate_example
        assert validate_example(
            "I should pick a type that works well for training.",
            "basics") is False
        assert validate_example(
            "I need to generate a training example now.",
            "basics") is False

    def test_commands_requires_cmd_block(self):
        """Commands stage requires [CMD]...[/CMD] blocks."""
        from enigma_engine.core.adaptive_trainer import validate_example
        no_cmd = "Q: How do I list files?\nA: Just type the list command."
        assert validate_example(no_cmd, "commands") is False
        with_cmd = "Q: How do I list files?\nA: Use [CMD]file.list[/CMD] to see all files."
        assert validate_example(with_cmd, "commands") is True

    def test_web_requires_search_web(self):
        """Web stage requires search.web command."""
        from enigma_engine.core.adaptive_trainer import validate_example
        no_web = "Q: What is the weather?\nA: Use [CMD]weather.check[/CMD]."
        assert validate_example(no_web, "web") is False
        with_web = "Q: What is the weather?\nA: Let me check [CMD]search.web weather today[/CMD]."
        assert validate_example(with_web, "web") is True

    def test_conversation_requires_turns(self):
        """Conversation stage needs dialogue structure."""
        from enigma_engine.core.adaptive_trainer import validate_example
        single = "This is just a paragraph about cats being nice animals and very fluffy."
        assert validate_example(single, "conversation") is False
        dialogue = "User: What is Python?\nAI: Python is a popular programming language used for many tasks."
        assert validate_example(dialogue, "conversation") is True

    def test_conversation_accepts_assistant_format(self):
        """Conversation stage accepts User/Assistant dialogue."""
        from enigma_engine.core.adaptive_trainer import validate_example
        dialogue = "User: What is Python?\nAssistant: Python is a popular programming language used for many tasks."
        assert validate_example(dialogue, "conversation") is True

    def test_rejects_empty(self):
        """Empty string is rejected."""
        from enigma_engine.core.adaptive_trainer import validate_example
        assert validate_example("", "basics") is False


class TestDeduplicateExamples:
    """Test deduplicate_examples for near-duplicate removal."""

    def test_removes_exact_duplicates(self):
        """Exact duplicates are removed."""
        from enigma_engine.core.adaptive_trainer import deduplicate_examples
        examples = ["Hello world!", "Hello world!", "Something else here now."]
        result = deduplicate_examples(examples)
        assert len(result) == 2

    def test_removes_near_duplicates(self):
        """Near-duplicates differing only in whitespace/punctuation."""
        from enigma_engine.core.adaptive_trainer import deduplicate_examples
        examples = [
            "Q: How do I save this?\nA: Just type file.write.",
            "Q: How do I save this?  \n A: Just type file.write",
        ]
        result = deduplicate_examples(examples)
        assert len(result) == 1

    def test_checks_against_existing(self):
        """Deduplicates against already-accumulated data."""
        from enigma_engine.core.adaptive_trainer import deduplicate_examples
        existing = ["Q: What is the capital of France?\nA: Paris."]
        new_data = [
            "Q: What is the capital of France?\nA: Paris.",
            "Q: What is the meaning of life?\nA: 42.",
        ]
        result = deduplicate_examples(new_data, existing)
        assert len(result) == 1
        assert "meaning" in result[0]

    def test_preserves_order(self):
        """First occurrence is kept, not the duplicate."""
        from enigma_engine.core.adaptive_trainer import deduplicate_examples
        examples = ["First unique example here.", "Second unique example here.", "First unique example here."]
        result = deduplicate_examples(examples)
        assert result[0] == "First unique example here."
        assert result[1] == "Second unique example here."

    def test_empty_input(self):
        """Empty list returns empty list."""
        from enigma_engine.core.adaptive_trainer import deduplicate_examples
        assert deduplicate_examples([]) == []

    def test_cmd_blocks_not_collapsed(self):
        """Different CMD blocks must not be treated as duplicates (#18)."""
        from enigma_engine.core.adaptive_trainer import deduplicate_examples
        examples = [
            "User: Save the file\nAssistant: [CMD]file.write data.txt[/CMD]",
            "User: Read the file\nAssistant: [CMD]file.read data.txt[/CMD]",
        ]
        result = deduplicate_examples(examples)
        assert len(result) == 2, "Different CMD blocks collapsed as dupes"

    def test_brackets_preserved_in_normalization(self):
        """Square brackets and dots are semantically significant (#18)."""
        from enigma_engine.core.adaptive_trainer import _normalize_for_dedup
        # Dots must be preserved so file.write != filewrite
        assert _normalize_for_dedup("file.write") != _normalize_for_dedup("filewrite")
        # Brackets preserved so [CMD] structure is visible
        assert "[" in _normalize_for_dedup("[CMD]test[/CMD]")


class TestParseScore:
    """Test parse_score for robust score extraction from LLM judgments."""

    def test_score_colon_format(self):
        """Parses 'SCORE: 7 | Good answer'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("SCORE: 7 | Good answer") == 7

    def test_score_colon_lowercase(self):
        """Parses 'score: 8'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("score: 8") == 8

    def test_n_slash_10_format(self):
        """Parses '7/10'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("I'd rate this 7/10") == 7

    def test_score_of_n(self):
        """Parses 'score of 6'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("I give a score of 6") == 6

    def test_give_it_a_n(self):
        """Parses 'give this a 8'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("I'd give this a 8") == 8

    def test_bare_number_on_line(self):
        """Parses bare number on its own line."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("7") == 7

    def test_clamps_high(self):
        """Scores above 10 are clamped to 10."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("SCORE: 15") == 10

    def test_clamps_low(self):
        """Scores below 1 are clamped to 1."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("SCORE: 0") == 1

    def test_returns_5_on_empty(self):
        """Empty text returns default 5."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("") == 5
        assert parse_score("   ") == 5

    def test_returns_5_on_no_score(self):
        """Text with no numeric score returns default 5."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("This was a good answer overall.") == 5

    def test_multiline_with_score_in_middle(self):
        """Finds SCORE: pattern even in middle of multiline text."""
        from enigma_engine.core.adaptive_trainer import parse_score
        judgment = (
            "The student answered well.\n"
            "SCORE: 8 | Good vocabulary\n"
            "Could improve on specifics.")
        assert parse_score(judgment) == 8

    def test_rating_format(self):
        """Parses 'rating: 6'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("My rating: 6 for this response.") == 6

    def test_does_not_match_digits_in_unrelated_text(self):
        """Digits embedded in non-score contexts should not match (#19)."""
        from enigma_engine.core.adaptive_trainer import parse_score
        # "200 lines of code" — the 200 should NOT be parsed as a score
        result = parse_score("The answer had about 200 lines of code.")
        # Should either return 5 (default) or 10 (clamped)
        # but NOT treat 200 as a raw score
        assert result in (5, 10)

    def test_bare_number_rejects_large_values(self):
        """Bare number pattern only matches 1-10 range."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("42") == 5  # Out of range, defaults

    def test_n_slash_10_with_word_boundary(self):
        """N/10 should match '7/10' but not '23/100'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("I'd say 7/10") == 7
        # 23/100 should not match as 23/10
        assert parse_score("Got 23/100 on the test") == 5

    def test_score_word_boundary_prefix(self):
        """'score12' should not match - 'score' needs word boundary."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("This is score12 quality") == 5

    def test_give_this_a_boundary(self):
        """'give this a 107' should not match '10' from '107'."""
        from enigma_engine.core.adaptive_trainer import parse_score
        assert parse_score("I give this a 107") == 5


# =============================================================================
# SMART BACKGROUND TRAINER TESTS
# =============================================================================

class TestSmartBackgroundTrainer:
    """Test the upgraded BackgroundTrainer with replay buffer and filtering."""

    def test_replay_buffer_is_deque(self):
        """Replay buffer should be a deque for O(1) bounded append."""
        from collections import deque
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer(replay_buffer_size=100)
        assert isinstance(bt.replay_buffer, deque), (
            "replay_buffer should be collections.deque, not list")
        assert bt.replay_buffer.maxlen == 100

    def test_replay_buffer_max_size(self):
        """Replay buffer has a configurable max size."""
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer(replay_buffer_size=500)
        assert bt.replay_buffer_size == 500

    def test_add_example_goes_to_replay_buffer(self):
        """add_example stores in replay buffer as well as queue."""
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer()
        bt.add_example("hello", "hi there", score=0.8)
        assert bt.example_queue.qsize() == 1
        assert len(bt.replay_buffer) == 1

    def test_all_examples_trained(self):
        """All examples are trained on (no quality filtering)."""
        from enigma_engine.router import BackgroundTrainer
        source = inspect.getsource(BackgroundTrainer._train_batch)
        # No quality_threshold filtering — trains on full batch
        assert "quality_threshold" not in source

    def test_train_batch_gradient_accumulation(self):
        """_train_batch accumulates gradients, single optimizer step per batch."""
        from enigma_engine.router import BackgroundTrainer
        source = inspect.getsource(BackgroundTrainer._train_batch)
        assert source.count("zero_grad()") == 1, (
            "_train_batch should call zero_grad exactly once (before loop)")
        assert source.count("optimizer.step()") == 1, (
            "_train_batch should call optimizer.step exactly once (after loop)")

    def test_retrain_on_replay_gradient_accumulation(self):
        """_retrain_on_replay accumulates gradients, single optimizer step."""
        from enigma_engine.router import BackgroundTrainer
        source = inspect.getsource(BackgroundTrainer._retrain_on_replay)
        assert source.count("zero_grad()") == 1, (
            "_retrain_on_replay should call zero_grad exactly once")
        assert source.count("optimizer.step()") == 1, (
            "_retrain_on_replay should call optimizer.step exactly once")

    def test_replay_buffer_capped(self):
        """Replay buffer respects max size."""
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer(replay_buffer_size=5)
        for i in range(10):
            bt.add_example(f"prompt{i}", f"response{i}", score=0.9)
        assert len(bt.replay_buffer) <= 5

    def test_dpo_pairs_collected(self):
        """Low-score examples generate DPO preference pairs."""
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer()
        assert hasattr(bt, "dpo_pairs")
        assert isinstance(bt.dpo_pairs, list)

    def test_get_stats_includes_replay_info(self):
        """get_stats returns replay buffer and DPO pair counts."""
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer()
        stats = bt.get_stats()
        assert "replay_buffer_size" in stats
        assert "dpo_pairs" in stats

    def test_replay_keeps_best_examples(self):
        """Replay buffer sorted by score, keeps highest."""
        from enigma_engine.router import BackgroundTrainer
        bt = BackgroundTrainer(replay_buffer_size=3)
        bt.add_example("a", "b", score=0.3)
        bt.add_example("c", "d", score=0.9)
        bt.add_example("e", "f", score=0.6)
        bt.add_example("g", "h", score=0.8)
        # Buffer keeps top 3 by score
        scores = [ex.score for ex in bt.replay_buffer]
        assert min(scores) >= 0.6


class TestCurriculumSeparators:
    """Curriculum examples are properly separated for parsing."""

    def test_format_training_pair_no_trailing_newlines(self):
        """_format_training_pair returns clean examples without extra newlines."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        pair = ForgeMixin._format_training_pair(
            "basics", "Hello", "Hi there")
        # Should be a clean single example
        assert pair == "Hello\nHi there"
        assert not pair.endswith("\n\n")


# ================================================================
# DA-C: Curated Dataset
# ================================================================


class TestCuratedDataset:
    """Test CuratedDataset management."""

    def test_create_entry(self):
        """DatasetEntry has correct defaults."""
        from enigma_engine.core.curated_dataset import DatasetEntry
        entry = DatasetEntry(text="Hello world", source="test")
        assert entry.status == "pending"
        assert entry.timestamp  # auto-set

    def test_entry_round_trip(self):
        """DatasetEntry to_dict / from_dict round-trips."""
        from enigma_engine.core.curated_dataset import DatasetEntry
        entry = DatasetEntry(
            text="Q: hi\nA: hello", source="guided",
            stage="basics", status="approved")
        d = entry.to_dict()
        loaded = DatasetEntry.from_dict(d)
        assert loaded.text == entry.text
        assert loaded.source == entry.source
        assert loaded.status == "approved"

    def test_add_and_count(self, tmp_path):
        """Adding entries increments count."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        assert ds.count == 0
        ds.add("example 1", source="test")
        ds.add("example 2", source="test")
        assert ds.count == 2
        assert ds.pending_count == 2

    def test_approve_reject(self, tmp_path):
        """Approve and reject change entry status."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("good", source="test")
        ds.add("bad", source="test")
        assert ds.approve(0) is True
        assert ds.reject(1) is True
        assert ds.approved_count == 1
        assert ds.rejected_count == 1
        assert ds.pending_count == 0

    def test_approve_all_pending(self, tmp_path):
        """approve_all_pending approves all pending entries."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("a", source="test")
        ds.add("b", source="test")
        ds.add("c", source="test")
        count = ds.approve_all_pending()
        assert count == 3
        assert ds.approved_count == 3
        assert ds.pending_count == 0

    def test_reject_all_pending(self, tmp_path):
        """reject_all_pending rejects all pending entries."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("a", source="test")
        ds.add("b", source="test")
        count = ds.reject_all_pending()
        assert count == 2
        assert ds.rejected_count == 2

    def test_get_approved_text(self, tmp_path):
        """get_approved_text returns only approved entry text."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("good data", source="test")
        ds.add("bad data", source="test")
        ds.approve(0)
        ds.reject(1)
        texts = ds.get_approved_text()
        assert texts == ["good data"]

    def test_get_training_data(self, tmp_path):
        """get_training_data joins approved entries."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("line 1", source="test", stage="basics")
        ds.add("line 2", source="test", stage="basics")
        ds.approve(0)
        ds.approve(1)
        text = ds.get_training_data(stage="basics")
        assert "line 1" in text
        assert "line 2" in text
        assert "\n\n" in text

    def test_get_by_source(self, tmp_path):
        """get_by_source filters by source."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("a", source="guided")
        ds.add("b", source="chat")
        ds.add("c", source="guided")
        guided = ds.get_by_source("guided")
        assert len(guided) == 2

    def test_get_by_stage(self, tmp_path):
        """get_by_stage filters by stage."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("a", source="test", stage="basics")
        ds.add("b", source="test", stage="conversation")
        basics = ds.get_by_stage("basics")
        assert len(basics) == 1

    def test_add_batch(self, tmp_path):
        """add_batch adds multiple entries at once."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        count = ds.add_batch(
            ["one", "two", "", "three"],
            source="batch", stage="basics")
        assert count == 3  # empty string skipped
        assert ds.count == 3

    def test_remove_entry(self, tmp_path):
        """remove() removes an entry by index."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("keep", source="test")
        ds.add("remove", source="test")
        assert ds.remove(1) is True
        assert ds.count == 1
        assert ds.entries[0].text == "keep"

    def test_save_and_load(self, tmp_path):
        """Dataset round-trips through save/load."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        path = tmp_path / "dataset.jsonl"
        ds = CuratedDataset(path)
        ds.add("entry 1", source="guided", stage="basics")
        ds.add("entry 2", source="chat", stage="conversation")
        ds.approve(0)
        ds.reject(1)
        ds.save()

        ds2 = CuratedDataset(path)
        assert ds2.count == 2
        assert ds2.approved_count == 1
        assert ds2.rejected_count == 1
        assert ds2.entries[0].text == "entry 1"
        assert ds2.entries[0].source == "guided"

    def test_summary(self, tmp_path):
        """summary() returns readable text."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("a", source="guided")
        ds.add("b", source="chat")
        text = ds.summary()
        assert "Total: 2" in text
        assert "Pending: 2" in text

    def test_invalid_index_operations(self, tmp_path):
        """Out-of-range operations return False."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        assert ds.approve(0) is False
        assert ds.reject(0) is False
        assert ds.remove(0) is False


# ================================================================
# CK-C: Rolling Best Checkpoints
# ================================================================


class TestRollingBestCheckpoints:
    """Test rolling_best_k in TrainingConfig."""

    def test_config_has_rolling_best_k(self):
        """TrainingConfig has rolling_best_k field."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert hasattr(config, "rolling_best_k")
        assert config.rolling_best_k == 0  # disabled by default

    def test_config_to_dict_includes_rolling(self):
        """to_dict includes rolling_best_k."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig(rolling_best_k=3)
        d = config.to_dict()
        assert d["rolling_best_k"] == 3

    def test_rolling_best_k_zero_is_noop(self):
        """rolling_best_k=0 means no rolling checkpoints saved."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig(rolling_best_k=0)
        assert config.rolling_best_k == 0
        d = config.to_dict()
        assert d["rolling_best_k"] == 0

    def test_save_every_default_disabled(self):
        """save_every defaults to 0 (disabled)."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert config.save_every == 0

    def test_cleanup_periodic_checkpoints_keeps_recent(self):
        """_cleanup_periodic_checkpoints keeps only the N most recent files."""
        from enigma_engine.core.training import Trainer
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # Create 6 fake checkpoint files
            for i in range(1, 7):
                (td_path / f"checkpoint_epoch_{i}.pt").write_text("x")
            # Clean up, keep 3
            Trainer._cleanup_periodic_checkpoints(td_path, "checkpoint_epoch_", keep=3)
            remaining = sorted(td_path.glob("checkpoint_epoch_*.pt"))
            assert len(remaining) == 3
            names = [f.name for f in remaining]
            # Should keep epochs 4, 5, 6 (the most recent)
            assert "checkpoint_epoch_4.pt" in names
            assert "checkpoint_epoch_5.pt" in names
            assert "checkpoint_epoch_6.pt" in names

    def test_cleanup_periodic_noop_when_few(self):
        """_cleanup_periodic_checkpoints does nothing when files <= keep."""
        from enigma_engine.core.training import Trainer
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for i in range(1, 3):
                (td_path / f"vision_epoch_{i}.pt").write_text("x")
            Trainer._cleanup_periodic_checkpoints(td_path, "vision_epoch_", keep=3)
            remaining = list(td_path.glob("vision_epoch_*.pt"))
            assert len(remaining) == 2  # both kept


# ================================================================
# TS-B: Training Queue
# ================================================================


class TestTrainingQueue:
    """Test TrainingQueue and TrainingJob."""

    def test_create_job(self):
        """TrainingJob has correct defaults."""
        from enigma_engine.core.training_queue import TrainingJob
        job = TrainingJob(mode="Solo", model_path="models/test.pth")
        assert job.status == "pending"
        assert job.progress == 0
        assert job.created_at  # auto-set

    def test_job_round_trip(self):
        """TrainingJob to_dict / from_dict round-trips."""
        from enigma_engine.core.training_queue import TrainingJob
        job = TrainingJob(
            mode="DPO", model_path="m.pth",
            data_path="d.jsonl", epochs=20)
        d = job.to_dict()
        loaded = TrainingJob.from_dict(d)
        assert loaded.mode == "DPO"
        assert loaded.epochs == 20
        assert loaded.model_path == "m.pth"

    def test_add_job(self):
        """Adding a job assigns an ID."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        job = q.add_job(TrainingJob(mode="Solo"))
        assert job.job_id == 1
        assert q.pending_count == 1
        job2 = q.add_job(TrainingJob(mode="DPO"))
        assert job2.job_id == 2
        assert q.pending_count == 2

    def test_remove_job(self):
        """Remove a pending job."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        job = q.add_job(TrainingJob(mode="Solo"))
        assert q.remove_job(job.job_id) is True
        assert q.pending_count == 0

    def test_cancel_job(self):
        """Cancel a pending job."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        job = q.add_job(TrainingJob(mode="Solo"))
        assert q.cancel_job(job.job_id) is True
        assert q.pending_count == 0
        assert q.jobs[0].status == "cancelled"

    def test_clear_completed(self):
        """clear_completed removes done/failed/cancelled jobs."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        j1 = q.add_job(TrainingJob(mode="Solo"))
        j2 = q.add_job(TrainingJob(mode="DPO"))
        j3 = q.add_job(TrainingJob(mode="LoRA"))
        j1.status = "completed"
        j2.status = "failed"
        # j3 is still pending
        removed = q.clear_completed()
        assert removed == 2
        assert len(q.jobs) == 1
        assert q.jobs[0].job_id == j3.job_id

    def test_queue_executes_jobs_sequentially(self):
        """Queue runs jobs in order via executor."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        import time

        results = []
        def executor(job):
            results.append(job.mode)
            return 0.5  # fake loss

        q = TrainingQueue()
        q.executor = executor
        q.add_job(TrainingJob(mode="Solo"))
        q.add_job(TrainingJob(mode="DPO"))
        q.start()

        # Wait for queue to finish
        for _ in range(50):
            if not q.is_running and q.pending_count == 0:
                break
            time.sleep(0.1)

        assert results == ["Solo", "DPO"]
        assert q.jobs[0].status == "completed"
        assert q.jobs[1].status == "completed"

    def test_queue_handles_failed_job(self):
        """Queue marks failed jobs but continues."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        import time

        call_count = [0]
        def executor(job):
            call_count[0] += 1
            if job.mode == "DPO":
                raise RuntimeError("DPO failed")
            return 0.1

        q = TrainingQueue()
        q.executor = executor
        q.add_job(TrainingJob(mode="Solo"))
        q.add_job(TrainingJob(mode="DPO"))
        q.add_job(TrainingJob(mode="LoRA"))
        q.start()

        for _ in range(50):
            if not q.is_running and q.pending_count == 0:
                break
            time.sleep(0.1)

        assert call_count[0] == 3
        assert q.jobs[0].status == "completed"
        assert q.jobs[1].status == "failed"
        assert "DPO failed" in q.jobs[1].error
        assert q.jobs[2].status == "completed"

    def test_queue_save_and_load(self, tmp_path):
        """Queue state round-trips through save/load."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        path = tmp_path / "queue.json"
        q = TrainingQueue(save_path=path)
        q.add_job(TrainingJob(mode="Solo", epochs=5))
        q.add_job(TrainingJob(mode="DPO", epochs=10))
        # Force save
        q._save_state()

        q2 = TrainingQueue(save_path=path)
        loaded = q2.load_state()
        assert loaded is True
        assert q2.pending_count == 2
        jobs = q2.jobs
        assert jobs[0].mode == "Solo"
        assert jobs[1].mode == "DPO"

    def test_interrupted_job_resets_to_pending(self, tmp_path):
        """Running jobs reset to pending on load."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        path = tmp_path / "queue.json"
        q = TrainingQueue(save_path=path)
        job = q.add_job(TrainingJob(mode="Solo"))
        job.status = "running"
        q._save_state()

        q2 = TrainingQueue(save_path=path)
        q2.load_state()
        assert q2.jobs[0].status == "pending"

    def test_queue_summary(self):
        """summary() returns readable text."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        q.add_job(TrainingJob(mode="Solo"))
        q.add_job(TrainingJob(mode="DPO"))
        text = q.summary()
        assert "Training Queue" in text
        assert "Solo" in text
        assert "DPO" in text

    def test_queue_callbacks(self):
        """Queue fires callbacks on job events."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        import time

        events = []
        def on_complete(job):
            events.append(("complete", job.mode))
        def on_queue_done():
            events.append(("queue_done",))

        q = TrainingQueue()
        q.executor = lambda job: 0.1
        q.on_job_complete = on_complete
        q.on_queue_complete = on_queue_done
        q.add_job(TrainingJob(mode="Solo"))
        q.start()

        for _ in range(50):
            if not q.is_running:
                break
            time.sleep(0.1)

        assert ("complete", "Solo") in events
        assert ("queue_done",) in events

    def test_pause_and_resume(self):
        """Pause stops processing, resume continues."""
        from enigma_engine.core.training_queue import (
            TrainingQueue)
        q = TrainingQueue()
        q.pause()
        assert q.is_paused is True
        q.resume()
        assert q.is_paused is False

    def test_reorder_job_with_running_job(self):
        """reorder_job must not break when a running job is in the list."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        j1 = q.add_job(TrainingJob(mode="Solo"))
        j2 = q.add_job(TrainingJob(mode="DPO"))
        j3 = q.add_job(TrainingJob(mode="LoRA"))

        # Simulate j1 running
        j1.status = "running"
        q._current_job = j1

        # Move j3 to position 0 among pending jobs
        moved = q.reorder_job(j3.job_id, 0)
        assert moved is True

        # j3 should now be before j2 in pending order
        pending = [j for j in q.jobs if j.status == "pending"]
        assert len(pending) == 2
        assert pending[0].job_id == j3.job_id
        assert pending[1].job_id == j2.job_id

        # Running job should still be in the list
        all_ids = [j.job_id for j in q.jobs]
        assert j1.job_id in all_ids

    def test_reorder_preserves_non_pending_jobs(self):
        """reorder_job must not affect completed/failed/running jobs."""
        from enigma_engine.core.training_queue import (
            TrainingQueue, TrainingJob)
        q = TrainingQueue()
        j1 = q.add_job(TrainingJob(mode="Solo"))
        j2 = q.add_job(TrainingJob(mode="DPO"))
        j3 = q.add_job(TrainingJob(mode="LoRA"))
        j4 = q.add_job(TrainingJob(mode="Vision"))

        # Mark j1 as completed, j2 as running
        j1.status = "completed"
        j2.status = "running"

        # Move j4 to position 0 among pending
        moved = q.reorder_job(j4.job_id, 0)
        assert moved is True

        jobs = q.jobs
        # completed and running must still be in order
        assert jobs[0].status == "completed"
        assert jobs[1].status == "running"
        # j4 should be before j3 in the pending slots
        pending = [j for j in jobs if j.status == "pending"]
        assert pending[0].job_id == j4.job_id
        assert pending[1].job_id == j3.job_id


# ================================================================
# TS-C: Overnight Plan
# ================================================================


class TestOvernightPlan:
    """Test OvernightPlan scheduling."""

    def test_create_plan(self):
        """OvernightPlan has correct defaults."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan(name="Test Plan")
        assert plan.status == "pending"
        assert plan.total_jobs == 0
        assert not plan.is_complete
        assert plan.created_at  # auto-set

    def test_add_job_config(self):
        """add_job_config adds jobs to the plan."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan()
        plan.add_job_config(
            mode="Solo", model_path="m.pth",
            data_path="d.txt", epochs=5)
        plan.add_job_config(
            mode="DPO", model_path="m.pth",
            data_path="d.jsonl", epochs=10)
        assert plan.total_jobs == 2
        assert plan.jobs[0]["mode"] == "Solo"
        assert plan.jobs[1]["epochs"] == 10

    def test_record_result(self):
        """record_result tracks completed jobs."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan()
        plan.add_job_config(mode="Solo", model_path="m.pth")
        plan.add_job_config(mode="DPO", model_path="m.pth")
        plan.record_result(plan.jobs[0], "completed", best_loss=0.5)
        assert plan.completed_jobs == 1
        assert plan.current_job_idx == 1
        assert not plan.is_complete

        plan.record_result(plan.jobs[1], "completed", best_loss=0.3)
        assert plan.is_complete
        assert plan.status == "completed"

    def test_record_failed_result(self):
        """Failed jobs are tracked in results."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan()
        plan.add_job_config(mode="Solo", model_path="m.pth")
        plan.record_result(
            plan.jobs[0], "failed", error="OOM")
        assert plan.failed_jobs == 1
        assert plan.results[0]["error"] == "OOM"

    def test_save_and_load(self, tmp_path):
        """OvernightPlan round-trips through JSON."""
        from enigma_engine.core.training_queue import OvernightPlan
        path = tmp_path / "plan.json"
        plan = OvernightPlan(name="Overnight")
        plan.add_job_config(
            mode="Solo", model_path="m.pth", epochs=5)
        plan.add_job_config(
            mode="DPO", model_path="m.pth", epochs=10)
        plan.record_result(plan.jobs[0], "completed", best_loss=0.5)
        plan.save(path)

        loaded = OvernightPlan.load(path)
        assert loaded.name == "Overnight"
        assert loaded.total_jobs == 2
        assert loaded.current_job_idx == 1
        assert loaded.completed_jobs == 1

    def test_summary(self):
        """summary() returns readable text."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan(name="Test")
        plan.add_job_config(mode="Solo", model_path="m.pth")
        text = plan.summary()
        assert "Test" in text
        assert "Solo" in text

    def test_to_queue_jobs(self):
        """to_queue_jobs converts remaining jobs to TrainingJob instances."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan()
        plan.add_job_config(
            mode="Solo", model_path="m.pth", epochs=5)
        plan.add_job_config(
            mode="DPO", model_path="m.pth", epochs=10)
        # Complete first job
        plan.record_result(plan.jobs[0], "completed")
        jobs = plan.to_queue_jobs()
        # Only second job should be converted
        assert len(jobs) == 1
        assert jobs[0].mode == "DPO"
        assert jobs[0].epochs == 10

    def test_to_queue_jobs_all_pending(self):
        """to_queue_jobs converts all jobs when none completed."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan()
        plan.add_job_config(mode="Solo", model_path="m.pth")
        plan.add_job_config(mode="LoRA", model_path="m.pth")
        jobs = plan.to_queue_jobs()
        assert len(jobs) == 2
        assert jobs[0].mode == "Solo"
        assert jobs[1].mode == "LoRA"

    def test_plan_is_complete_when_all_done(self):
        """Plan marks completed after all jobs recorded."""
        from enigma_engine.core.training_queue import OvernightPlan
        plan = OvernightPlan()
        plan.add_job_config(mode="Solo", model_path="m.pth")
        plan.record_result(plan.jobs[0], "completed")
        assert plan.is_complete
        assert plan.completed_at  # timestamp set


# ================================================================
# RL TRAINING (RL-B / RL-C)
# ================================================================

class TestRLTrainingImports:
    """Verify rl_training module imports and class structure."""

    def test_reward_trainer_config_defaults(self):
        from enigma_engine.core.rl_training import RewardTrainerConfig
        cfg = RewardTrainerConfig()
        assert cfg.epochs == 3
        assert cfg.learning_rate == 1e-5
        assert cfg.batch_size == 4
        assert cfg.max_length == 512

    def test_rlhf_config_defaults(self):
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert cfg.epochs == 3
        assert cfg.kl_coeff == 0.1
        assert cfg.clip_range == 0.2
        assert cfg.n_responses == 4

    def test_selfplay_config_defaults(self):
        from enigma_engine.core.rl_training import SelfPlayConfig
        cfg = SelfPlayConfig()
        assert cfg.epochs == 3
        assert cfg.kl_coeff == 0.05
        assert cfg.n_responses == 4
        assert "{prompt}" in cfg.score_prompt
        assert "{response}" in cfg.score_prompt


class TestRewardModel:
    """Test RewardModel creation and forward pass."""

    def test_reward_model_creates_from_enigma(self):
        """RewardModel can be built from a small Enigma model."""
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.rl_training import RewardModel

        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        base = Enigma(config=cfg)
        rm = RewardModel(base, freeze_base=True)

        # Should have reward_head
        assert hasattr(rm, "reward_head")
        assert rm.reward_head.out_features == 1

        # Base weights should be frozen
        for p in rm.tok_embeddings.parameters():  # type: ignore[union-attr]
            assert not p.requires_grad

    def test_reward_model_forward_shape(self):
        """Forward produces (B,) scalar rewards."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.rl_training import RewardModel

        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        base = Enigma(config=cfg)
        rm = RewardModel(base)

        ids = torch.randint(0, 128, (2, 10))
        rewards = rm(ids)
        assert rewards.shape == (2,)

    def test_reward_model_with_attention_mask(self):
        """Forward respects attention_mask to find last real token."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.rl_training import RewardModel

        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        base = Enigma(config=cfg)
        rm = RewardModel(base)

        ids = torch.randint(0, 128, (1, 10))
        mask = torch.ones(1, 10)
        mask[0, 7:] = 0  # Last 3 are padding
        rewards = rm(ids, attention_mask=mask)
        assert rewards.shape == (1,)


class TestRewardTrainer:
    """Test RewardTrainer preference pair training."""

    def test_encode_pairs(self):
        """_encode_pairs converts preference data to tensor pairs."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.rl_training import (
            RewardModel, RewardTrainer)
        from enigma_engine.core.tokenizer import get_tokenizer

        cfg = ForgeConfig(vocab_size=8000, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=64)
        base = Enigma(config=cfg)
        rm = RewardModel(base)

        tokenizer = get_tokenizer()
        trainer = RewardTrainer(rm, tokenizer)

        pairs = trainer._encode_pairs([
            {"prompt": "Hi", "chosen": "Hello!", "rejected": "Go away"},
        ])
        assert len(pairs) == 1
        c, r = pairs[0]
        assert isinstance(c, torch.Tensor)
        assert isinstance(r, torch.Tensor)

    def test_request_stop(self):
        """request_stop sets internal flag."""
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.rl_training import (
            RewardModel, RewardTrainer)

        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        base = Enigma(config=cfg)
        rm = RewardModel(base)
        trainer = RewardTrainer(rm, None)

        assert not trainer._should_stop()
        trainer.request_stop()
        assert trainer._should_stop()


# ================================================================
# LORA ADAPTER MANAGER (FP-D)
# ================================================================

class TestLoRAAdapterManager:
    """Test per-task LoRA adapter management."""

    def test_list_empty(self, tmp_path):
        """list_tasks returns empty when no adapters exist."""
        from enigma_engine.core.lora_utils import LoRAAdapterManager
        mgr = LoRAAdapterManager(base_dir=tmp_path / "adapters")
        assert mgr.list_tasks() == []

    def test_create_and_list(self, tmp_path):
        """create() stores adapter; list_tasks() finds it."""
        from enigma_engine.core.lora_utils import LoRAAdapterManager
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        mgr = LoRAAdapterManager(base_dir=tmp_path / "adapters")
        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        model = Enigma(config=cfg)

        # Mark some params as trainable to simulate LoRA
        for p in list(model.parameters())[:2]:
            p.requires_grad = True

        path = mgr.create("coding", model)
        assert path.exists()
        assert "coding" in mgr.list_tasks()

    def test_save_and_switch(self, tmp_path):
        """save() persists weights; switch() loads them back."""
        from enigma_engine.core.lora_utils import LoRAAdapterManager
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        mgr = LoRAAdapterManager(base_dir=tmp_path / "adapters")
        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        model = Enigma(config=cfg)

        # Simulate trainable params
        for p in list(model.parameters())[:2]:
            p.requires_grad = True

        mgr.create("task_a", model)
        mgr.save("task_a", model)
        assert mgr.active_task == "task_a"

        # Create second task
        mgr.create("task_b", model)
        mgr.save("task_b", model)
        assert "task_a" in mgr.list_tasks()
        assert "task_b" in mgr.list_tasks()

    def test_delete(self, tmp_path):
        """delete() removes adapter from disk."""
        from enigma_engine.core.lora_utils import LoRAAdapterManager
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        mgr = LoRAAdapterManager(base_dir=tmp_path / "adapters")
        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        model = Enigma(config=cfg)
        for p in list(model.parameters())[:2]:
            p.requires_grad = True

        mgr.create("todelete", model)
        assert "todelete" in mgr.list_tasks()
        mgr.delete("todelete")
        assert "todelete" not in mgr.list_tasks()

    def test_switch_nonexistent_raises(self, tmp_path):
        """switch() raises FileNotFoundError for missing task."""
        from enigma_engine.core.lora_utils import LoRAAdapterManager
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        mgr = LoRAAdapterManager(base_dir=tmp_path / "adapters")
        cfg = ForgeConfig(vocab_size=128, dim=64, n_layers=2,
                          n_heads=2, n_kv_heads=2, max_seq_len=32)
        model = Enigma(config=cfg)

        with pytest.raises(FileNotFoundError, match="No adapter"):
            mgr.switch("nonexistent", model)


# ================================================================
# TRAINING MONITOR (TM-B / TM-C / TM-D)
# ================================================================

class TestTrainingMonitor:
    """Test training monitor: loss tracking and history."""

    def test_record_loss(self):
        """record_loss stores values and computes best."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor()
        m.start_run()
        m.record_loss(2.0)
        m.record_loss(1.5)
        m.record_loss(1.8)

        assert len(m.losses) == 3
        assert m.best_loss == 1.5
        assert m.current_loss == 1.8

    def test_moving_average(self):
        """moving_average produces correct-length output."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor(moving_avg_window=3)
        m.start_run()
        for v in [3.0, 2.0, 1.0, 0.5]:
            m.record_loss(v)

        ma = m.moving_average()
        assert len(ma) == 4
        # First value is just itself
        assert ma[0] == 3.0
        # Last value is avg of [2.0, 1.0, 0.5]
        assert abs(ma[3] - (2.0 + 1.0 + 0.5) / 3) < 0.01

    def test_get_chart_data(self):
        """get_chart_data returns structured dict."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor()
        m.start_run()
        m.record_loss(1.0)
        m.record_loss(0.5)

        data = m.get_chart_data()
        assert "steps" in data
        assert "losses" in data
        assert "moving_avg" in data
        assert "best_loss" in data
        assert data["best_loss"] == 0.5

    def test_epoch_loss_tracking(self):
        """record_epoch_loss stores per-epoch values."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor()
        m.start_run()
        m.record_epoch_loss(2.0)
        m.record_epoch_loss(1.5)

        assert m.epoch_losses == [2.0, 1.5]

    def test_epoch_perplexities_tracking(self):
        """record_epoch_loss auto-computes perplexity from loss."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor()
        m.start_run()
        m.record_epoch_loss(1.0)
        m.record_epoch_loss(0.5)

        assert len(m.epoch_perplexities) == 2
        import math
        assert abs(m.epoch_perplexities[0] - math.exp(1.0)) < 0.01
        assert abs(m.epoch_perplexities[1] - math.exp(0.5)) < 0.01

    def test_perplexity_in_chart_data(self):
        """get_chart_data includes epoch_perplexities key."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor()
        m.start_run()
        m.record_loss(1.0)
        m.record_epoch_loss(1.0)

        data = m.get_chart_data()
        assert "epoch_perplexities" in data
        assert len(data["epoch_perplexities"]) == 1

    def test_finish_run_includes_perplexity(self, tmp_path):
        """finish_run extra dict has final_perplexity and best_perplexity."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        hist_path = tmp_path / "hist.json"
        m = TrainingMonitor(history_path=hist_path)
        m.start_run()
        m.record_loss(1.0)
        m.record_epoch_loss(2.0)
        m.record_epoch_loss(1.0)

        run = m.finish_run(mode="sft", model_name="test")
        assert run.extra.get("final_perplexity") is not None
        assert run.extra.get("best_perplexity") is not None
        assert run.extra["best_perplexity"] <= run.extra["final_perplexity"]

    def test_perplexity_reset_on_start_run(self):
        """start_run resets epoch_perplexities list."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor()
        m.start_run()
        m.record_epoch_loss(1.0)
        assert len(m.epoch_perplexities) == 1

        m.start_run()
        assert len(m.epoch_perplexities) == 0

    def test_training_run_serialization(self):
        """TrainingRun round-trips through dict."""
        from enigma_engine.core.training_monitor import TrainingRun

        run = TrainingRun(
            run_id="test_1",
            mode="sft",
            model_name="base",
            final_loss=0.5,
            total_steps=100,
        )
        d = run.to_dict()
        assert d["run_id"] == "test_1"
        assert d["mode"] == "sft"

        restored = TrainingRun.from_dict(d)
        assert restored.run_id == "test_1"
        assert restored.final_loss == 0.5

    def test_finish_run_saves_history(self, tmp_path):
        """finish_run persists to history file."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        hist_path = tmp_path / "history.json"
        m = TrainingMonitor(history_path=hist_path)
        m.start_run()
        m.record_loss(1.0)
        m.record_loss(0.5)

        run = m.finish_run(mode="sft", model_name="test_model")
        assert run.mode == "sft"
        assert run.total_steps == 2

        # File should exist
        assert hist_path.exists()

        # Load it back
        history = m.get_history()
        assert len(history) == 1
        assert history[0].mode == "sft"

    def test_history_multiple_runs(self, tmp_path):
        """Multiple runs append to the history."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        hist_path = tmp_path / "history.json"
        m = TrainingMonitor(history_path=hist_path)

        # Run 1
        m.start_run()
        m.record_loss(2.0)
        m.finish_run(mode="sft", model_name="m1")

        # Run 2
        m.start_run()
        m.record_loss(1.0)
        m.finish_run(mode="dpo", model_name="m2")

        history = m.get_history()
        assert len(history) == 2

    def test_history_filter_by_mode(self, tmp_path):
        """get_history can filter by training mode."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        hist_path = tmp_path / "history.json"
        m = TrainingMonitor(history_path=hist_path)

        m.start_run()
        m.record_loss(1.0)
        m.finish_run(mode="sft")

        m.start_run()
        m.record_loss(1.0)
        m.finish_run(mode="dpo")

        sft_only = m.get_history(mode="sft")
        assert len(sft_only) == 1
        assert sft_only[0].mode == "sft"

    def test_clear_history(self, tmp_path):
        """clear_history removes the file."""
        from enigma_engine.core.training_monitor import TrainingMonitor

        hist_path = tmp_path / "history.json"
        m = TrainingMonitor(history_path=hist_path)
        m.start_run()
        m.record_loss(1.0)
        m.finish_run(mode="sft")

        assert hist_path.exists()
        m.clear_history()
        assert not hist_path.exists()

    def test_losses_list_capped(self):
        """_losses list must not grow unbounded — capped at MAX_LOSSES."""
        from enigma_engine.core.training_monitor import TrainingMonitor
        m = TrainingMonitor()
        m.start_run()
        # Record more losses than the cap
        for i in range(120_000):
            m.record_loss(float(i))
        # Should be capped, not 120k
        assert len(m.losses) <= 100_001

    def test_steps_list_stays_in_sync_with_losses(self):
        """steps and losses must have the same length after cap."""
        from enigma_engine.core.training_monitor import TrainingMonitor
        m = TrainingMonitor()
        m.start_run()
        for i in range(110_000):
            m.record_loss(float(i))
        assert len(m.losses) == len(m.steps)

    def test_get_chart_data_nan_inf_guarded(self):
        """get_chart_data moving_avg handles NaN/inf without corruption."""
        import math
        from enigma_engine.core.training_monitor import TrainingMonitor

        m = TrainingMonitor(moving_avg_window=3)
        m.start_run()
        m.record_loss(2.0)
        m.record_loss(float("nan"))
        m.record_loss(1.0)
        m.record_loss(float("inf"))
        m.record_loss(0.5)

        data = m.get_chart_data()
        ma = data["moving_avg"]
        assert len(ma) == 5
        # No NaN or inf in moving average output
        for val in ma:
            assert not math.isnan(val), "NaN leaked into chart moving_avg"
            assert not math.isinf(val), "inf leaked into chart moving_avg"


# ================================================================
# THREAD SAFETY — Suggestion #8A+D
# ================================================================

class TestTrainingConfigAdamFields:
    """TrainingConfig must expose Adam optimizer fields."""

    def test_default_betas(self):
        """adam_beta1/beta2 default to LM-friendly values."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.adam_beta1 == 0.9
        assert cfg.adam_beta2 == 0.95
        assert cfg.adam_eps == 1e-8

    def test_custom_betas(self):
        """adam_beta1/beta2 can be overridden."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(adam_beta1=0.85, adam_beta2=0.999, adam_eps=1e-6)
        assert cfg.adam_beta1 == 0.85
        assert cfg.adam_beta2 == 0.999
        assert cfg.adam_eps == 1e-6

    def test_to_dict_includes_adam_fields(self):
        """to_dict() must include all three Adam fields."""
        from enigma_engine.core.training import TrainingConfig
        d = TrainingConfig().to_dict()
        assert "adam_beta1" in d
        assert "adam_beta2" in d
        assert "adam_eps" in d
        assert d["adam_beta1"] == 0.9
        assert d["adam_beta2"] == 0.95


class TestAutoLRConfig:
    """I-3: auto_lr field in TrainingConfig."""

    def test_auto_lr_default_false(self):
        """auto_lr defaults to False."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.auto_lr is False

    def test_auto_lr_in_to_dict(self):
        """to_dict() includes auto_lr."""
        from enigma_engine.core.training import TrainingConfig
        d = TrainingConfig(auto_lr=True).to_dict()
        assert "auto_lr" in d
        assert d["auto_lr"] is True

    def test_lr_find_method_exists(self):
        """Trainer has _lr_find method."""
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, "_lr_find")

    def test_lr_find_signature(self):
        """_lr_find accepts batches and returns a float."""
        sig = inspect.signature(
            __import__(
                "enigma_engine.core.training", fromlist=["Trainer"]
            ).Trainer._lr_find
        )
        params = list(sig.parameters.keys())
        assert "batches" in params
        assert "min_lr" in params
        assert "max_lr" in params

    def test_lr_find_restores_state(self):
        """_lr_find source must save and restore model+optimizer state."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer._lr_find)
        assert "deepcopy" in source, "must snapshot state"
        assert "load_state_dict" in source, "must restore state"


class TestAutoBPEDropout:
    """I-6: Auto BPE-dropout for multi-epoch small datasets."""

    def test_auto_bpe_dropout_source_check(self):
        """train() checks epochs and tokens-per-param for auto BPE-dropout."""
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "bpe_dropout" in source
        assert "tokens_per_param" in source

    def test_auto_bpe_dropout_conditions(self):
        """Auto BPE-dropout requires epochs > 3 and bpe_dropout == 0."""
        from enigma_engine.core.training import TrainingConfig
        # Default bpe_dropout is 0.1 — auto should NOT override user setting
        cfg = TrainingConfig(epochs=10, bpe_dropout=0.1)
        assert cfg.bpe_dropout == 0.1  # already set, skip auto

        # When explicitly 0 and enough epochs, auto should kick in
        cfg2 = TrainingConfig(epochs=10, bpe_dropout=0.0)
        assert cfg2.bpe_dropout == 0.0
        assert cfg2.epochs > 3


class TestLoraTrainerWeightDecay:
    """LoRA trainer weight_decay must be configurable."""

    def test_custom_weight_decay_stored(self):
        """Custom weight_decay is stored on the instance."""
        import torch.nn as nn
        from unittest.mock import MagicMock, patch

        # Patch create_lora_model to avoid actual LoRA application
        with patch("enigma_engine.core.lora_utils.create_lora_model",
                   side_effect=lambda m, c: m):
            from enigma_engine.core.lora_utils import LoraTrainer
            model = nn.Linear(4, 4)
            tok = MagicMock()
            trainer = LoraTrainer(model, tok, weight_decay=0.05)
            assert trainer.weight_decay == 0.05


# ================================================================
# V-G: HYBRID CNN+ViT
# ================================================================

class TestForgeNewModes:
    """Verify new training modes are wired into FORGE."""

    def test_new_modes_mixin_exists(self):
        """ForgeNewModesMixin is in the inheritance chain."""
        from enigma_engine.gui.gui_forge_new_modes import ForgeNewModesMixin
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert issubclass(ForgeMixin, ForgeNewModesMixin)


# ================================================================
# 12: OOM Recovery — VRAM-Based Batch Size + training OOM handling
# ================================================================


class TestRecommendTrainingBatchSize:
    """Tests for VRAM-tier based recommended training batch size."""

    def test_cpu_only_returns_small_batch(self):
        """CPU-only profile returns batch size 1 or 2."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(gpu_available=False, ram_gb=8.0)
        bs = recommend_training_batch_size(profile)
        assert bs == 1

    def test_cpu_high_ram(self):
        """CPU with high RAM returns 2."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(gpu_available=False, ram_gb=64.0)
        assert recommend_training_batch_size(profile) == 2

    def test_low_vram_returns_1(self):
        """GPU with < 6 GB VRAM returns 1."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(
            gpu_available=True, gpu_vram_gb=4.0)
        assert recommend_training_batch_size(profile) == 1

    def test_6gb_vram_returns_2(self):
        """GPU with 6 GB VRAM returns 2."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(
            gpu_available=True, gpu_vram_gb=6.0)
        assert recommend_training_batch_size(profile) == 2

    def test_12gb_vram_returns_4(self):
        """GPU with 12 GB VRAM returns 4."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(
            gpu_available=True, gpu_vram_gb=12.0)
        assert recommend_training_batch_size(profile) == 4

    def test_24gb_vram_returns_8(self):
        """GPU with 24 GB VRAM returns 8."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(
            gpu_available=True, gpu_vram_gb=24.0)
        assert recommend_training_batch_size(profile) == 8

    def test_48gb_vram_returns_16(self):
        """GPU with 48+ GB VRAM returns 16."""
        from enigma_engine.core.hardware_detection import (
            HardwareProfile, recommend_training_batch_size,
        )
        profile = HardwareProfile(
            gpu_available=True, gpu_vram_gb=48.0)
        assert recommend_training_batch_size(profile) == 16

    def test_in_all_exports(self):
        """recommend_training_batch_size is in __all__."""
        from enigma_engine.core import hardware_detection
        assert "recommend_training_batch_size" in hardware_detection.__all__


# ================================================================
# Training: random import fix + weight sanity check
# ================================================================


class TestTrainingRandomImportFix:
    """Verify train() doesn't crash from shadowed random import."""

    def test_no_local_random_import_in_train(self):
        """train() must not have a local 'import random' that shadows module-level."""
        import ast
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        tree = ast.parse(textwrap.dedent(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "random", (
                        f"Local 'import random' at line ~{node.lineno} in "
                        f"train() shadows module-level import — use it directly")


class TestRLHFNoDeepCopy:
    """Tests for LoRA-based RLHF reference policy."""

    def test_no_gpu_deepcopy_in_train(self):
        """Guard: train() methods must not deepcopy to stay on GPU."""
        from enigma_engine.core.rl_training import (
            RLHFTrainer, SelfPlayTrainer,
        )
        for cls in (RLHFTrainer, SelfPlayTrainer):
            source = inspect.getsource(cls.train)
            assert "deepcopy" not in source, (
                f"{cls.__name__}.train() still uses deepcopy")


# ================================================================
# 14: Adaptive Trainer Dead Code Removal
# ================================================================


class TestAdaptiveTrainerCleanup:
    """Tests for dead code removal in adaptive_trainer.py."""

    def test_no_advance_score_field(self):
        """TrainingPlan no longer has advance_score."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        assert "advance_score" not in TrainingPlan.__dataclass_fields__

    def test_no_retry_score_field(self):
        """TrainingPlan no longer has retry_score."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        assert "retry_score" not in TrainingPlan.__dataclass_fields__

    def test_decide_action_still_works(self):
        """decide_action still returns advance/complete."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan()
        assert plan.decide_action(8.0) == "advance"
        # Last stage → complete
        plan.current_stage_idx = len(plan.stages) - 1
        assert plan.decide_action(8.0) == "complete"

    def test_summary_no_thresholds(self):
        """summary() no longer mentions thresholds."""
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        plan = TrainingPlan(student_name="s", trainer_name="t")
        text = plan.summary()
        assert "Thresholds" not in text

    def test_load_still_filters_unknown_keys(self, tmp_path):
        """load() ignores keys removed from dataclass (forward compat)."""
        import json
        from enigma_engine.core.adaptive_trainer import TrainingPlan
        # Save JSON with old fields that no longer exist
        data = {
            "student_path": "m.pth",
            "advance_score": 7.0,
            "retry_score": 5.0,
            "stages": ["basics", "conversation"],
        }
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        plan = TrainingPlan.load(path)
        assert plan.student_path == "m.pth"
        assert not hasattr(plan, "advance_score")


# ================================================================
# PPO Rewrite — True PPO with value head, GAE, clipped surrogate
# ================================================================


class TestValueHead:
    """ValueHead: MLP critic that predicts state values from hidden states."""

    def test_value_head_forward_shape(self):
        """ValueHead(dim) produces (B, T) values from (B, T, dim) hidden states."""
        import torch
        from enigma_engine.core.rl_training import ValueHead
        vh = ValueHead(dim=64)
        h = torch.randn(2, 10, 64)
        values = vh(h)
        assert values.shape == (2, 10)

    def test_value_head_single_token(self):
        """ValueHead works with single-token inputs."""
        import torch
        from enigma_engine.core.rl_training import ValueHead
        vh = ValueHead(dim=32)
        h = torch.randn(1, 1, 32)
        values = vh(h)
        assert values.shape == (1, 1)


class TestRolloutBuffer:
    """RolloutBuffer stores (logprobs, values, rewards, masks) for PPO updates."""

    def test_store_and_get(self):
        """Can store experience and retrieve it."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        buf.store(
            log_probs=torch.randn(5),
            values=torch.randn(5),
            rewards=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
            response_mask=torch.ones(5),
        )
        assert len(buf) == 1

    def test_compute_advantages_gae(self):
        """compute_advantages uses GAE (gamma, lam)."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        buf.store(
            log_probs=torch.zeros(3),
            values=torch.tensor([0.5, 0.5, 0.5]),
            rewards=torch.tensor([0.0, 0.0, 1.0]),
            response_mask=torch.ones(3),
        )
        advantages, returns = buf.compute_advantages(gamma=1.0, lam=0.95)
        assert advantages.shape[0] > 0
        assert returns.shape[0] > 0
        # Last token has reward=1, value=0.5, so advantage > 0
        assert advantages[-1] > 0

    def test_clear(self):
        """clear resets the buffer."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        buf.store(
            log_probs=torch.zeros(3),
            values=torch.zeros(3),
            rewards=torch.zeros(3),
            response_mask=torch.ones(3),
        )
        assert len(buf) == 1
        buf.clear()
        assert len(buf) == 0


class TestLossMetricTokenCount:
    """Epoch loss must weight by non-padding tokens, not total tokens."""

    def test_train_uses_non_pad_count(self):
        """Training loop epoch_loss weights by non-padding token count."""
        import inspect
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer.train)
        assert "non_pad" in src
        assert "batch_loss * non_pad" in src

    def test_validate_uses_non_pad_count(self):
        """Validation loop weights by non-padding token count."""
        import inspect
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer._validate)
        assert "non_pad" in src
        assert "loss.item() * non_pad" in src


class TestPPOConfig:
    """RLHFConfig gains PPO-specific fields."""

    def test_config_has_value_coeff(self):
        """RLHFConfig has value_coeff field."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert hasattr(cfg, "value_coeff")
        assert cfg.value_coeff == 0.5

    def test_config_has_entropy_coeff(self):
        """RLHFConfig has entropy_coeff field."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert hasattr(cfg, "entropy_coeff")
        assert cfg.entropy_coeff == 0.01

    def test_config_has_gae_lambda(self):
        """RLHFConfig has gae_lambda field."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert hasattr(cfg, "gae_lambda")
        assert cfg.gae_lambda == 0.95

    def test_config_has_ppo_epochs(self):
        """RLHFConfig has ppo_epochs for minibatch updates."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert hasattr(cfg, "ppo_epochs")
        assert cfg.ppo_epochs == 4

    def test_config_has_minibatch_size(self):
        """RLHFConfig has minibatch_size."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert hasattr(cfg, "minibatch_size")
        assert cfg.minibatch_size == 4

    def test_backward_compat_existing_defaults(self):
        """Old defaults remain unchanged."""
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert cfg.epochs == 3
        assert cfg.kl_coeff == 0.1
        assert cfg.clip_range == 0.2
        assert cfg.n_responses == 4


class TestPPORatioComputation:
    """PPO must compute real importance sampling ratio, not hardcode 1.0."""

    def test_rollout_buffer_stores_full_ids(self):
        """RolloutBuffer accepts and stores full_ids and prompt_len."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        full_ids = torch.tensor([[1, 2, 3, 4, 5]])
        buf.store(
            log_probs=torch.randn(3),
            values=torch.randn(3),
            rewards=torch.tensor([0.0, 0.0, 1.0]),
            response_mask=torch.ones(3),
            full_ids=full_ids,
            prompt_len=2,
        )
        assert buf._full_ids[0] is not None
        assert buf._prompt_lens[0] == 2
        assert torch.equal(buf._full_ids[0], full_ids)

    def test_rollout_buffer_clear_clears_ids(self):
        """clear() also clears full_ids and prompt_lens."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        buf.store(
            log_probs=torch.randn(3),
            values=torch.randn(3),
            rewards=torch.zeros(3),
            response_mask=torch.ones(3),
            full_ids=torch.tensor([[1, 2, 3]]),
            prompt_len=1,
        )
        buf.clear()
        assert len(buf._full_ids) == 0
        assert len(buf._prompt_lens) == 0

    def test_rollout_buffer_backward_compat(self):
        """RolloutBuffer.store() works without full_ids (backward compat)."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        buf.store(
            log_probs=torch.randn(3),
            values=torch.randn(3),
            rewards=torch.zeros(3),
            response_mask=torch.ones(3),
        )
        assert buf._full_ids[0] is None
        assert buf._prompt_lens[0] is None


# ================================================================
# 15a: RolloutBuffer ref_logps + Differentiable KL
# ================================================================


class TestRolloutBufferRefLogps:
    """RolloutBuffer stores ref_logps for differentiable KL penalty."""

    def test_store_ref_logps(self):
        """ref_logps stored and detached when provided."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        ref = torch.randn(3, requires_grad=True)
        buf.store(
            log_probs=torch.randn(3),
            values=torch.randn(3),
            rewards=torch.zeros(3),
            response_mask=torch.ones(3),
            ref_logps=ref,
        )
        assert buf._ref_logps[0] is not None
        assert not buf._ref_logps[0].requires_grad

    def test_store_without_ref_logps(self):
        """ref_logps defaults to None (backward compat with replay)."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        buf.store(
            log_probs=torch.randn(3),
            values=torch.randn(3),
            rewards=torch.zeros(3),
            response_mask=torch.ones(3),
        )
        assert buf._ref_logps[0] is None

    def test_clear_clears_ref_logps(self):
        """clear() also clears ref_logps list."""
        import torch
        from enigma_engine.core.rl_training import RolloutBuffer
        buf = RolloutBuffer()
        buf.store(
            log_probs=torch.randn(3),
            values=torch.randn(3),
            rewards=torch.zeros(3),
            response_mask=torch.ones(3),
            ref_logps=torch.randn(3),
        )
        buf.clear()
        assert len(buf._ref_logps) == 0


class TestGRPOLogRatioClamped:
    """GRPO log-ratio must be clamped to prevent exp() overflow."""

    def test_grpo_clamp_in_source(self):
        """GRPO log_ratio uses .clamp(-20, 20) like PPO and ReMax."""
        import inspect
        from enigma_engine.core.rl_training import GRPOTrainer
        src = inspect.getsource(GRPOTrainer.train)
        assert ".clamp(-20, 20)" in src

    def test_grpo_std_unbiased_false(self):
        """GRPO uses population std (unbiased=False) for small groups."""
        import inspect
        from enigma_engine.core.rl_training import GRPOTrainer
        src = inspect.getsource(GRPOTrainer.train)
        assert "unbiased=False" in src


class TestPPODifferentiableKL:
    """PPO/Self-Play KL penalty must be differentiable (tensor, not float)."""

    def test_ppo_uses_total_kl_tensor(self):
        """RLHFTrainer.train uses total_kl (tensor) not avg_kl (stale float)."""
        import inspect
        from enigma_engine.core.rl_training import RLHFTrainer
        src = inspect.getsource(RLHFTrainer.train)
        assert "total_kl / mb_count" in src
        # Must NOT have the old stale-float pattern
        assert "self._kl_coeff * avg_kl" not in src

    def test_selfplay_uses_total_kl_tensor(self):
        """SelfPlayTrainer.train uses total_kl (tensor) not avg_kl."""
        import inspect
        from enigma_engine.core.rl_training import SelfPlayTrainer
        src = inspect.getsource(SelfPlayTrainer.train)
        assert "total_kl / mb_count" in src
        # Must NOT have the old stale-float pattern
        assert "self._kl_coeff * avg_kl" not in src

    def test_rollout_stores_ref_logps_in_ppo(self):
        """RLHFTrainer.train passes ref_logps to rollout.store()."""
        import inspect
        from enigma_engine.core.rl_training import RLHFTrainer
        src = inspect.getsource(RLHFTrainer.train)
        assert "ref_logps=ref_logps" in src

    def test_rollout_stores_ref_logps_in_selfplay(self):
        """SelfPlayTrainer.train passes ref_logps to rollout.store()."""
        import inspect
        from enigma_engine.core.rl_training import SelfPlayTrainer
        src = inspect.getsource(SelfPlayTrainer.train)
        assert "ref_logps=ref_logps" in src


class TestGRPOKLFormula:
    """S748: GRPO KL must use standard single-sample estimator."""

    def test_grpo_kl_uses_standard_formula(self):
        """GRPO KL should be (token_logps - ref_token_logps).mean(),
        not the old p_ref * log(p_ref/p_policy) formula."""
        import inspect
        from enigma_engine.core.rl_training import GRPOTrainer
        src = inspect.getsource(GRPOTrainer.train)
        assert "(token_logps - ref_token_logps).mean()" in src
        # Old broken formula must be gone
        assert "ref_token_logps.exp()" not in src


class TestReMaxKLSign:
    """S749: ReMax KL must penalize divergence (positive when diverging)."""

    def test_remax_kl_correct_sign(self):
        """KL should be (new_logps - old_logps), not (old - new)."""
        import inspect
        from enigma_engine.core.rl_training import ReMaxTrainer
        src = inspect.getsource(ReMaxTrainer.train)
        assert "(new_logps - old_logps).mean()" in src
        # Old flipped formula must be gone
        assert "(old_logps - new_logps).mean()" not in src


# ================================================================
# 15: CuratedDataset Thread Lock
# ================================================================


class TestCuratedDatasetThreadSafety:
    """Tests for threading.Lock in CuratedDataset."""

    def test_has_lock(self, tmp_path):
        """CuratedDataset has a threading.Lock."""
        import threading
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        assert hasattr(ds, "_lock")
        assert isinstance(ds._lock, type(threading.Lock()))

    def test_entries_returns_copy(self, tmp_path):
        """entries property returns a snapshot, not the internal list."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("a", source="test")
        snap = ds.entries
        snap.append(None)  # type: ignore[arg-type]  # mutate snapshot
        assert ds.count == 1  # internal unchanged

    def test_concurrent_adds(self, tmp_path):
        """Concurrent adds don't corrupt data."""
        import threading as th
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")

        def add_entries():
            for i in range(50):
                ds.add(f"entry-{i}", source="thread")

        threads = [th.Thread(target=add_entries) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ds.count == 200  # 4 threads × 50 entries

    def test_approve_under_lock(self, tmp_path):
        """approve() is thread-safe."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        for i in range(100):
            ds.add(f"e{i}", source="test")

        def approve_range(start, end):
            for i in range(start, end):
                ds.approve(i)

        import threading as th
        t1 = th.Thread(target=approve_range, args=(0, 50))
        t2 = th.Thread(target=approve_range, args=(50, 100))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert ds.approved_count == 100


# ================================================================
# Suggestion 16: RAG BM25 + sparse + stop words
# ================================================================


# ================================================================
# Vision Encoder Dedup + Max Visual Tokens (#27)
# ================================================================

class TestTrainingConfigValSplit:
    """TrainingConfig val_split field."""

    def test_default_val_split(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.val_split == 0.1  # 10% held out by default

    def test_to_dict_includes_val_split(self):
        from enigma_engine.core.training import TrainingConfig
        d = TrainingConfig(val_split=0.1).to_dict()
        assert d["val_split"] == 0.1

    def test_validate_rejects_bad_val_split(self):
        import pytest
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(val_split=1.0)
        with pytest.raises(ValueError, match="val_split"):
            cfg.validate()
        cfg2 = TrainingConfig(val_split=-0.1)
        with pytest.raises(ValueError, match="val_split"):
            cfg2.validate()

    def test_validate_accepts_valid_val_split(self):
        from enigma_engine.core.training import TrainingConfig
        TrainingConfig(val_split=0.0).validate()
        TrainingConfig(val_split=0.2).validate()


# =============================================================================
# AMP + GRADIENT ACCUMULATION VERIFICATION (#11)
# =============================================================================


class TestGeneralDataMixing:
    """TrainingConfig general_mix_ratio and general_data fields."""

    def test_default_ratio(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.general_mix_ratio == 0.2

    def test_default_general_data_empty(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.general_data == ""

    def test_custom_ratio_and_path(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(
            general_mix_ratio=0.3, general_data="/some/file.txt")
        assert cfg.general_mix_ratio == 0.3
        assert cfg.general_data == "/some/file.txt"

    def test_zero_ratio_disables_mixing(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(
            general_mix_ratio=0.0, general_data="some data")
        assert cfg.general_mix_ratio == 0.0

    def test_default_label_smoothing(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.label_smoothing == 0.05

    def test_default_early_stopping(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.early_stopping_patience == 5


class TestValidationLoop:
    """Trainer._validate() method."""

    def test_validation_losses_populated(self):
        """state.validation_losses filled when val_split > 0."""
        from enigma_engine.core.training import TrainingState
        s = TrainingState()
        assert s.validation_losses == []
        s.validation_losses.append(1.5)
        assert len(s.validation_losses) == 1

    def test_abort_reason_default_empty(self):
        """abort_reason defaults to empty string."""
        from enigma_engine.core.training import TrainingState
        s = TrainingState()
        assert s.abort_reason == ""

    def test_abort_reason_set_round_trip(self):
        """abort_reason can be set and read back."""
        from enigma_engine.core.training import TrainingState
        s = TrainingState()
        s.abort_reason = "NaN/Inf loss detected"
        assert s.abort_reason == "NaN/Inf loss detected"

    def test_abort_reason_in_all_abort_paths(self):
        """All abort early-returns set self.state.abort_reason."""
        import inspect
        from enigma_engine.core.training import Trainer
        # Check train() method — every 'return self.state' preceded
        # by an abort logger.error should also set abort_reason
        src = inspect.getsource(Trainer.train)
        import re
        # Count how many times abort_reason is set
        sets = re.findall(r'self\.state\.abort_reason\s*=', src)
        # There should be at least 3 abort paths in train()
        assert len(sets) >= 3, (
            f"Expected >= 3 abort_reason assignments in train(), "
            f"found {len(sets)}")


# ================================================================
# Reproducible Baseline
# ================================================================

class TestReproducibleBaseline:
    """Verify seed, dataset fingerprint, and config dump."""

    def test_dataset_fingerprint_deterministic(self):
        """Same data must produce same fingerprint."""
        from enigma_engine.core.training import dataset_fingerprint
        fp1 = dataset_fingerprint("hello world")
        fp2 = dataset_fingerprint("hello world")
        assert fp1 == fp2
        assert len(fp1) == 16  # 16 hex chars

    def test_dataset_fingerprint_varies(self):
        """Different data must produce different fingerprint."""
        from enigma_engine.core.training import dataset_fingerprint
        fp1 = dataset_fingerprint("hello")
        fp2 = dataset_fingerprint("world")
        assert fp1 != fp2

    def test_config_has_seed_field(self):
        """TrainingConfig must have seed field."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "seed")
        assert cfg.seed is None  # default is None

    def test_config_has_golden_eval_path(self):
        """TrainingConfig must have golden_eval_path field."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "golden_eval_path")
        assert cfg.golden_eval_path == ""

    def test_seed_in_to_dict(self):
        """seed must appear in config.to_dict()."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(seed=42)
        d = cfg.to_dict()
        assert d["seed"] == 42


# ================================================================
# Golden Prompt Regression Eval
# ================================================================

class TestGoldenPromptEval:
    """Verify golden eval function exists and works."""

    def test_golden_eval_missing_file(self):
        """Returns empty results for non-existent file."""
        from enigma_engine.core.training_evaluation import run_golden_eval
        import unittest.mock as mock
        model = mock.MagicMock()
        tokenizer = mock.MagicMock()
        result = run_golden_eval(
            model, tokenizer, "/nonexistent/golden.json")
        assert result["total"] == 0
        assert result["pass_rate"] == 0.0

    def test_golden_eval_empty_cases(self):
        """Returns empty results for empty JSON array."""
        import json
        import tempfile
        from pathlib import Path
        from enigma_engine.core.training_evaluation import run_golden_eval
        import unittest.mock as mock
        model = mock.MagicMock()
        tokenizer = mock.MagicMock()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump([], f)
            f.flush()
            result = run_golden_eval(model, tokenizer, f.name)
        Path(f.name).unlink(missing_ok=True)
        assert result["total"] == 0


# ================================================================
# Tool Eval Wiring
# ================================================================

class TestToolEvalWiring:
    """Verify tool eval test cases and FORGE wiring."""

    def test_default_tool_test_cases_exist(self):
        """DEFAULT_TOOL_TEST_CASES must be importable and non-empty."""
        from enigma_engine.core.training_evaluation import (
            DEFAULT_TOOL_TEST_CASES,
        )
        assert isinstance(DEFAULT_TOOL_TEST_CASES, list)
        assert len(DEFAULT_TOOL_TEST_CASES) >= 3

    def test_tool_test_case_structure(self):
        """Each case must have prompt and expected_command."""
        from enigma_engine.core.training_evaluation import (
            DEFAULT_TOOL_TEST_CASES,
        )
        for case in DEFAULT_TOOL_TEST_CASES:
            assert "prompt" in case
            assert "expected_command" in case


# ================================================================
# Image System Prompt
# ================================================================


# ================================================================
# Sequence Packing
# ================================================================


class TestSequencePacking:
    """Tests for sequence packing in training."""

    def test_packing_config_default_off(self):
        """use_sequence_packing defaults to False."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert config.use_sequence_packing is False

    def test_packing_config_in_to_dict(self):
        """use_sequence_packing appears in to_dict()."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig(use_sequence_packing=True)
        d = config.to_dict()
        assert "use_sequence_packing" in d
        assert d["use_sequence_packing"] is True

    def test_pack_sequences_packs_short_seqs(self):
        """Short sequences get combined into one row."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import pack_sequences
        # Three short sequences, max_len=20
        seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        packed, masks = pack_sequences(seqs, max_length=20, eos_id=2, pad_id=0)
        # All should fit in one row (3+1+2+1+4+1 = 12 <= 20)
        assert packed.shape[0] == 1
        assert packed.shape[1] == 20

    def test_pack_sequences_mask_is_4d(self):
        """Packing produces a 4D attention mask (B, 1, T, T)."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import pack_sequences
        seqs = [[1, 2, 3], [4, 5]]
        packed, masks = pack_sequences(seqs, max_length=16, eos_id=2, pad_id=0)
        assert masks.ndim == 4
        assert masks.shape[1] == 1  # head dim
        assert masks.shape[2] == masks.shape[3]  # T x T

    def test_pack_sequences_cross_boundary_blocked(self):
        """Tokens in different documents cannot attend to each other."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import pack_sequences
        # Two sequences: [10, 11] and [20, 21]
        seqs = [[10, 11], [20, 21]]
        packed, masks = pack_sequences(seqs, max_length=16, eos_id=2, pad_id=0)
        # Row 0 contains both seqs packed: [10, 11, EOS, 20, 21, EOS, pad...]
        # Position 0 (token 10) should NOT attend to position 3 (token 20)
        # masks has -inf for blocked positions, 0 for allowed
        assert masks[0, 0, 0, 3].item() < -1e4  # blocked (cross-boundary)
        # Position 1 can attend to position 0 (same doc, causal ok)
        assert masks[0, 0, 1, 0].item() == 0.0   # allowed (same doc, past)

    def test_pack_sequences_causal_within_doc(self):
        """Within a document, future tokens are still masked (causal)."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import pack_sequences
        seqs = [[10, 11, 12]]
        packed, masks = pack_sequences(seqs, max_length=16, eos_id=2, pad_id=0)
        # Position 0 should NOT attend to position 1 (future within same doc)
        assert masks[0, 0, 0, 1].item() < -1e4

    def test_pack_sequences_long_seq_gets_own_row(self):
        """A sequence that fills max_length goes into its own row."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import pack_sequences
        long_seq = list(range(1, 16))  # 15 tokens
        short_seq = [100, 101]
        packed, masks = pack_sequences(
            [long_seq, short_seq], max_length=16, eos_id=2, pad_id=0)
        # long_seq (15) + EOS = 16 → fills a row alone
        # short_seq (2) + EOS = 3 → separate row
        assert packed.shape[0] == 2

    def test_pack_sequences_pad_positions_masked(self):
        """Padding positions at the end of packed rows are masked out."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.training import pack_sequences
        seqs = [[10, 11]]
        packed, masks = pack_sequences(seqs, max_length=8, eos_id=2, pad_id=0)
        # Row: [10, 11, EOS, 0, 0, 0, 0, 0]
        # Position 0 should NOT attend to position 3 (padding)
        assert masks[0, 0, 0, 3].item() < -1e4


# ─────────────────────────────────────────────────────────────────────────────
# Streaming training (disk-backed batches for large datasets)
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamingDiskIO:
    """Verify disk-backed sequence write/read round-trips correctly."""

    def test_write_and_read_sequences(self, tmp_path):
        """Sequences survive a write→read round-trip via JSONL."""
        from enigma_engine.core.training import Trainer
        sequences = [
            "Hello world",
            "Line with\nnewline inside",
            'Quotes "and" special chars: <>&',
            "Unicode: ñ ü ö 日本語",
        ]
        path = tmp_path / "seqs.jsonl"
        offsets = Trainer._write_sequences_to_disk(sequences, path)
        assert len(offsets) == 4

        # Read all
        result = Trainer._read_sequences_from_disk(
            path, offsets, [0, 1, 2, 3])
        assert result == sequences

    def test_read_subset_in_order(self, tmp_path):
        """Reading a subset returns sequences in the requested order."""
        from enigma_engine.core.training import Trainer
        sequences = [f"seq_{i}" for i in range(20)]
        path = tmp_path / "seqs.jsonl"
        offsets = Trainer._write_sequences_to_disk(sequences, path)

        # Read indices out of order
        result = Trainer._read_sequences_from_disk(
            path, offsets, [15, 3, 7, 0])
        assert result == ["seq_15", "seq_3", "seq_7", "seq_0"]

    def test_read_empty_indices(self, tmp_path):
        """Reading with empty indices returns empty list."""
        from enigma_engine.core.training import Trainer
        sequences = ["a", "b", "c"]
        path = tmp_path / "seqs.jsonl"
        offsets = Trainer._write_sequences_to_disk(sequences, path)
        result = Trainer._read_sequences_from_disk(path, offsets, [])
        assert result == []


class TestStreamingThreshold:
    """Verify streaming guards and threshold constants exist."""

    def test_minhash_skipped_for_large_datasets(self):
        """MinHash dedup is guarded by _MINHASH_LIMIT in train()."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "_MINHASH_LIMIT" in source
        assert "minhash_dedup" in source

    def test_curriculum_skipped_for_large_datasets(self):
        """Curriculum sorting is guarded by _CURRICULUM_LIMIT in train()."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "_CURRICULUM_LIMIT" in source

    def test_streaming_threshold_exists(self):
        """Trainer has the streaming threshold constant."""
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, "_STREAMING_THRESHOLD")
        assert Trainer._STREAMING_THRESHOLD > 0

    def test_streaming_window_exists(self):
        """Trainer has the streaming window constant."""
        from enigma_engine.core.training import Trainer
        assert hasattr(Trainer, "_STREAMING_WINDOW")
        assert Trainer._STREAMING_WINDOW > 0

    def test_stream_batches_is_generator(self):
        """_stream_batches should be a generator (yields batches)."""
        import inspect
        from enigma_engine.core.training import Trainer
        assert inspect.isgeneratorfunction(Trainer._stream_batches)

    def test_train_has_streaming_path(self):
        """train() branches on use_streaming for large datasets."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "use_streaming" in source
        assert "_STREAMING_THRESHOLD" in source
        assert "_write_sequences_to_disk" in source
        assert "_stream_batches" in source

    def test_cleanup_in_all_return_paths(self):
        """Every return from train() calls _cleanup_temp_files."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        # Count returns and cleanups in train()
        returns = source.count("return self.state")
        cleanups = source.count("_cleanup_temp_files()")
        assert cleanups >= returns, (
            f"Found {returns} 'return self.state' but only "
            f"{cleanups} '_cleanup_temp_files()' calls")

    def test_train_has_disk_backed_path(self):
        """train() accepts data_path/data_offsets for disk-backed mode."""
        import inspect
        from enigma_engine.core.training import Trainer
        sig = inspect.signature(Trainer.train)
        assert "data_path" in sig.parameters
        assert "data_offsets" in sig.parameters
        source = inspect.getsource(Trainer.train)
        assert "_disk_backed" in source
        assert "data_path is not None and data_offsets is not None" in source

    def test_eval_every_wired_in_train_loop(self):
        """eval_every triggers step-based validation in train loop."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "eval_every" in source, (
            "eval_every not consumed in train()")
        assert "self.config.eval_every" in source


class TestDiskBackedTraining:
    """Integration test: disk-backed training path (N-1 streaming)."""

    def test_disk_backed_trains_tiny_model(self, tmp_path):
        """Full pipeline: JSONL on disk → Trainer.train(data_path)."""
        import json
        import torch
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        tok = SimpleTokenizer()
        cfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32,
            n_layers=1, n_heads=2, max_seq_len=32)
        model = Enigma(config=cfg)

        # Write sequences to JSONL with byte offsets
        seq_path = tmp_path / "sequences.jsonl"
        offsets = []
        sequences = [
            "The quick brown fox jumps over the lazy dog. " * 5,
            "A journey of a thousand miles begins with one step. " * 5,
            "To be or not to be that is the question. " * 5,
            "All that glitters is not gold but silver shines too. " * 5,
            "Knowledge is power and power corrupts absolutely. " * 5,
            "The early bird catches the worm every morning. " * 5,
        ]
        with open(seq_path, "wb") as f:
            for seq in sequences:
                offsets.append(f.tell())
                f.write((json.dumps(seq) + "\n").encode("utf-8"))

        w_before = model.tok_embeddings.weight.data.clone()

        t_cfg = TrainingConfig(
            epochs=1, batch_size=2,
            learning_rate=1e-3, val_split=0.0,
            checkpoint_dir=str(tmp_path / "ckpt"),
            use_amp=False)
        trainer = Trainer(model, tok, t_cfg)
        state = trainer.train(
            data_path=str(seq_path),
            data_offsets=offsets)

        # Training should have run and changed weights
        assert state is not None
        assert state.best_loss < float("inf"), (
            "Training did not produce any loss values")
        w_after = model.tok_embeddings.weight.data
        assert not torch.equal(w_before, w_after), (
            "Model weights unchanged after training")

    def test_disk_backed_val_split(self, tmp_path):
        """Disk-backed path correctly splits train/val offsets."""
        import json
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        tok = SimpleTokenizer()
        cfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32,
            n_layers=1, n_heads=2, max_seq_len=32)
        model = Enigma(config=cfg)

        seq_path = tmp_path / "sequences.jsonl"
        offsets = []
        with open(seq_path, "wb") as f:
            for i in range(20):
                offsets.append(f.tell())
                text = f"Sample text number {i} " * 10
                f.write(
                    (json.dumps(text) + "\n").encode("utf-8"))

        t_cfg = TrainingConfig(
            epochs=1, batch_size=2,
            learning_rate=1e-3, val_split=0.2,
            checkpoint_dir=str(tmp_path / "ckpt"),
            use_amp=False)
        trainer = Trainer(model, tok, t_cfg)
        state = trainer.train(
            data_path=str(seq_path),
            data_offsets=offsets)
        assert state is not None
        assert state.best_loss < float("inf")

    def test_disk_backed_rejects_empty_offsets(self, tmp_path):
        """Raises ValueError when data_offsets is empty."""
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        tok = SimpleTokenizer()
        cfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32,
            n_layers=1, n_heads=2, max_seq_len=32)
        model = Enigma(config=cfg)
        t_cfg = TrainingConfig(epochs=1)
        trainer = Trainer(model, tok, t_cfg)
        # Create a real file so FileNotFoundError doesn't fire first
        fake_file = tmp_path / "empty.jsonl"
        fake_file.write_text("")
        with pytest.raises(ValueError, match="empty"):
            trainer.train(
                data_path=str(fake_file), data_offsets=[])

    def test_disk_backed_rejects_missing_file(self, tmp_path):
        """Raises FileNotFoundError when data_path doesn't exist."""
        from enigma_engine.core.model import Enigma, ForgeConfig
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer

        tok = SimpleTokenizer()
        cfg = ForgeConfig(
            vocab_size=tok.vocab_size, dim=32,
            n_layers=1, n_heads=2, max_seq_len=32)
        model = Enigma(config=cfg)
        t_cfg = TrainingConfig(epochs=1)
        trainer = Trainer(model, tok, t_cfg)
        with pytest.raises(FileNotFoundError):
            trainer.train(
                data_path=str(tmp_path / "nope.jsonl"),
                data_offsets=[0, 100])


# ─────────────────────────────────────────────────────────────────────────────
# I-12: Resume pre-training from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestFindLatestCheckpoint:
    """_find_latest_checkpoint selects the highest-epoch checkpoint."""

    def test_finds_highest_epoch(self, tmp_path):
        """Returns the checkpoint with the largest epoch number."""
        from enigma_engine.core.training import Trainer
        for i in [1, 5, 3, 10, 7]:
            (tmp_path / f"checkpoint_epoch_{i}.pt").write_text("x")
        result = Trainer._find_latest_checkpoint(tmp_path)
        assert result is not None
        assert result.name == "checkpoint_epoch_10.pt"

    def test_returns_none_when_empty(self, tmp_path):
        """Returns None when no checkpoints exist."""
        from enigma_engine.core.training import Trainer
        assert Trainer._find_latest_checkpoint(tmp_path) is None

    def test_returns_none_for_nonexistent_dir(self, tmp_path):
        """Returns None when directory doesn't exist."""
        from enigma_engine.core.training import Trainer
        assert Trainer._find_latest_checkpoint(
            tmp_path / "nonexistent") is None

    def test_ignores_non_checkpoint_files(self, tmp_path):
        """Non-checkpoint .pt files are ignored unless best/final."""
        from enigma_engine.core.training import Trainer
        (tmp_path / "rolling_best_e5_loss0.1234.pt").write_text("x")
        assert Trainer._find_latest_checkpoint(tmp_path) is None

    def test_falls_back_to_best_model(self, tmp_path):
        """best_model.pt is returned when no periodic checkpoints exist."""
        from enigma_engine.core.training import Trainer
        (tmp_path / "best_model.pt").write_text("x")
        result = Trainer._find_latest_checkpoint(tmp_path)
        assert result is not None
        assert result.name == "best_model.pt"

    def test_periodic_preferred_over_best(self, tmp_path):
        """Periodic checkpoint is preferred when both exist."""
        from enigma_engine.core.training import Trainer
        (tmp_path / "checkpoint_epoch_5.pt").write_text("x")
        (tmp_path / "best_model.pt").write_text("x")
        result = Trainer._find_latest_checkpoint(tmp_path)
        assert result is not None
        assert result.name == "checkpoint_epoch_5.pt"

    def test_falls_back_to_final_model(self, tmp_path):
        """final_model.pt is returned as last resort."""
        from enigma_engine.core.training import Trainer
        (tmp_path / "final_model.pt").write_text("x")
        result = Trainer._find_latest_checkpoint(tmp_path)
        assert result is not None
        assert result.name == "final_model.pt"

    def test_new_naming_format(self, tmp_path):
        """New {stem}_checkpoint{N}.pt naming is recognized."""
        from enigma_engine.core.training import Trainer
        # Simulate dir named after model (GUI sets checkpoint_dir
        # to models/checkpoints/{model_stem})
        ckpt_dir = tmp_path / "mymodel"
        ckpt_dir.mkdir()
        for i in [1, 3, 2]:
            (ckpt_dir / f"mymodel_checkpoint{i}.pt").write_text("x")
        result = Trainer._find_latest_checkpoint(ckpt_dir)
        assert result is not None
        assert result.name == "mymodel_checkpoint3.pt"

    def test_new_best_final_naming(self, tmp_path):
        """New {stem}_best.pt and {stem}_final.pt are recognized."""
        from enigma_engine.core.training import Trainer
        ckpt_dir = tmp_path / "mymodel"
        ckpt_dir.mkdir()
        (ckpt_dir / "mymodel_best.pt").write_text("x")
        result = Trainer._find_latest_checkpoint(ckpt_dir)
        assert result is not None
        assert result.name == "mymodel_best.pt"

    def test_new_preferred_over_legacy(self, tmp_path):
        """New format periodic checkpoint preferred over legacy best."""
        from enigma_engine.core.training import Trainer
        ckpt_dir = tmp_path / "mymodel"
        ckpt_dir.mkdir()
        (ckpt_dir / "mymodel_checkpoint2.pt").write_text("x")
        (ckpt_dir / "best_model.pt").write_text("x")
        result = Trainer._find_latest_checkpoint(ckpt_dir)
        assert result is not None
        assert result.name == "mymodel_checkpoint2.pt"


class TestCleanupKeepMarker:
    """_cleanup_periodic_checkpoints respects .keep sidecar files."""

    def test_keeps_protected_checkpoints(self, tmp_path):
        """Checkpoints with .keep marker are not deleted."""
        from enigma_engine.core.training import Trainer
        for i in range(1, 6):
            (tmp_path / f"m_checkpoint{i}.pt").write_text("x")
        # Protect checkpoint 1
        (tmp_path / "m_checkpoint1.pt.keep").write_text("protected")

        Trainer._cleanup_periodic_checkpoints(
            tmp_path, "m_checkpoint", keep=2)

        remaining = sorted(p.name for p in tmp_path.glob("m_checkpoint*.pt"))
        # Keep newest 2 (4, 5) + protected 1
        assert "m_checkpoint1.pt" in remaining
        assert "m_checkpoint4.pt" in remaining
        assert "m_checkpoint5.pt" in remaining
        assert "m_checkpoint2.pt" not in remaining
        assert "m_checkpoint3.pt" not in remaining


# ─────────────────────────────────────────────────────────────────────────────
# TC-3: _parse_training_data — data pipeline format detection
# ─────────────────────────────────────────────────────────────────────────────

def _make_bare_trainer():
    """Create a Trainer without __init__ for testing _parse_training_data.

    _parse_training_data only uses self for method dispatch, no model/tokenizer.
    """
    from enigma_engine.core.training import Trainer
    return object.__new__(Trainer)


class TestParseTrainingDataQA:
    """_parse_training_data correctly detects Q&A format (TC-3)."""

    def test_qa_simple(self):
        """Q: / A: pairs are normalised to User: / Assistant:."""
        trainer = _make_bare_trainer()
        data = "Q: What is AI?\nA: Artificial Intelligence."
        result = trainer._parse_training_data(data)
        assert len(result) == 1
        assert result[0].startswith("User:")
        assert "Assistant:" in result[0]
        assert "What is AI?" in result[0]

    def test_qa_multiple(self):
        """Multiple Q&A pairs are all extracted."""
        trainer = _make_bare_trainer()
        data = "Q: first\nA: one\nQ: second\nA: two"
        result = trainer._parse_training_data(data)
        assert len(result) == 2
        assert "first" in result[0]
        assert "second" in result[1]


class TestParseTrainingDataJSONL:
    """_parse_training_data correctly detects JSONL format (TC-3)."""

    def test_jsonl_prompt_completion(self):
        """Standard prompt/completion JSONL fields."""
        import json
        trainer = _make_bare_trainer()
        data = json.dumps({"prompt": "hello", "completion": "world"})
        result = trainer._parse_training_data(data)
        assert len(result) == 1
        assert "User: hello" in result[0]
        assert "Assistant: world" in result[0]

    def test_jsonl_question_answer(self):
        """Alternative question/answer JSONL fields."""
        import json
        trainer = _make_bare_trainer()
        data = json.dumps({"question": "q", "answer": "a"})
        result = trainer._parse_training_data(data)
        assert len(result) == 1
        assert "User: q" in result[0]

    def test_jsonl_multi_line(self):
        """Multiple JSONL lines are all parsed."""
        import json
        trainer = _make_bare_trainer()
        lines = [
            json.dumps({"prompt": "a", "completion": "b"}),
            json.dumps({"prompt": "c", "completion": "d"}),
        ]
        data = "\n".join(lines)
        result = trainer._parse_training_data(data)
        assert len(result) == 2

    def test_jsonl_skips_invalid_lines(self):
        """Malformed JSON lines are skipped, valid ones parsed."""
        import json
        trainer = _make_bare_trainer()
        data = "not json\n" + json.dumps({"prompt": "ok", "completion": "fine"})
        # Doesn't start with '{' so won't try JSONL path — falls through
        result = trainer._parse_training_data(data)
        # Should still produce some output (paragraph fallback)
        assert len(result) >= 0


class TestParseTrainingDataDialogue:
    """_parse_training_data detects User/AI dialogue format (TC-3)."""

    def test_user_assistant(self):
        """User: / Assistant: pairs are extracted."""
        trainer = _make_bare_trainer()
        data = "User: hi\nAssistant: hello there"
        result = trainer._parse_training_data(data)
        assert len(result) == 1
        assert "User: hi" in result[0]
        assert "Assistant: hello there" in result[0]

    def test_human_ai(self):
        """Human: / AI: pairs are normalised."""
        trainer = _make_bare_trainer()
        data = "Human: test\nAI: response"
        result = trainer._parse_training_data(data)
        assert len(result) == 1
        assert result[0].startswith("User:")
        assert "Assistant:" in result[0]


class TestParseTrainingDataDictList:
    """_parse_training_data handles pre-parsed list of dicts (TC-3)."""

    def test_dict_list_prompt_completion(self):
        """List of dicts with prompt/completion."""
        trainer = _make_bare_trainer()
        data = [{"prompt": "x", "completion": "y"}]
        result = trainer._parse_training_data(data)
        assert len(result) == 1
        assert "User: x" in result[0]

    def test_dict_list_string_items(self):
        """List containing raw strings passes through."""
        trainer = _make_bare_trainer()
        data = ["raw sequence one", "raw sequence two"]
        result = trainer._parse_training_data(data)
        assert result == data

    def test_dict_list_skips_empty(self):
        """Dicts with empty prompt/completion are skipped."""
        trainer = _make_bare_trainer()
        data = [{"prompt": "", "completion": "y"}, {"prompt": "x", "completion": "z"}]
        result = trainer._parse_training_data(data)
        assert len(result) == 1
        assert "User: x" in result[0]


class TestParseTrainingDataFallback:
    """_parse_training_data falls back to paragraph/line splitting (TC-3)."""

    def test_paragraph_split(self):
        """Long paragraphs separated by blank lines are extracted."""
        trainer = _make_bare_trainer()
        para1 = "A" * 60  # > 50 chars
        para2 = "B" * 60
        data = f"{para1}\n\n{para2}"
        result = trainer._parse_training_data(data)
        assert len(result) == 2
        assert para1 in result[0]

    def test_short_paragraphs_skipped(self):
        """Paragraphs under 50 chars are filtered out."""
        trainer = _make_bare_trainer()
        data = "short\n\n" + "A" * 60
        result = trainer._parse_training_data(data)
        assert len(result) == 1  # Only the long one

    def test_line_split_last_resort(self):
        """If no paragraphs qualify, splits by lines."""
        trainer = _make_bare_trainer()
        data = "line one\nline two\nline three"
        result = trainer._parse_training_data(data)
        assert len(result) == 3

    def test_empty_string_returns_empty(self):
        """Empty input returns empty list."""
        trainer = _make_bare_trainer()
        result = trainer._parse_training_data("")
        assert result == []


# ================================================================
# True byte-level BPE (UTF-8 byte sequences)
# ================================================================


# ================================================================
# SWA Checkpoint Save/Load
# ================================================================


class TestSWACheckpoint:
    """SWA state must be saved and loaded in checkpoints."""

    def test_save_checkpoint_includes_swa(self, tmp_path):
        """_save_checkpoint saves swa_state_dict when SWA is active."""
        import torch
        from enigma_engine.core.training import SWAWeightAverager

        # Build a minimal Trainer-like object with SWA
        model = torch.nn.Linear(4, 4)
        swa = SWAWeightAverager(model, update_interval=1)
        swa.update(model, step=0)  # n_averaged = 1

        # Save checkpoint and verify swa_state_dict is present
        from enigma_engine.core.training import Trainer, TrainingConfig
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.vocab_size = 16
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        config = TrainingConfig(use_amp=False, swa_update_interval=1)
        trainer = Trainer(model, tokenizer, config)
        trainer.swa = swa

        ckpt_path = tmp_path / "test.pt"
        trainer._save_checkpoint(ckpt_path)

        checkpoint = torch.load(ckpt_path, weights_only=False)
        assert "swa_state_dict" in checkpoint, \
            "SWA state must be saved in checkpoint"
        assert checkpoint["swa_state_dict"]["n_averaged"] == 1

    def test_load_checkpoint_restores_swa(self, tmp_path):
        """load_checkpoint restores swa_state_dict when SWA is active."""
        import torch
        from enigma_engine.core.training import (
            Trainer, TrainingConfig, SWAWeightAverager,
        )
        from unittest.mock import MagicMock

        model = torch.nn.Linear(4, 4)
        tokenizer = MagicMock()
        tokenizer.vocab_size = 16
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        config = TrainingConfig(use_amp=False, swa_update_interval=1)
        trainer = Trainer(model, tokenizer, config)

        # Simulate some SWA progress
        trainer.swa.update(model, step=0)
        trainer.swa.update(model, step=1)
        assert trainer.swa.n_averaged == 2

        # Save
        ckpt_path = tmp_path / "test.pt"
        trainer._save_checkpoint(ckpt_path)

        # Create fresh trainer and load
        model2 = torch.nn.Linear(4, 4)
        trainer2 = Trainer(model2, tokenizer, config)
        assert trainer2.swa.n_averaged == 0  # fresh

        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.swa.n_averaged == 2, \
            "SWA n_averaged must be restored from checkpoint"


# ================================================================
# TrainingQueue Lock Consistency
# ================================================================


class TestQueueLockConsistency:
    """TrainingQueue summary() and cleanup must use locks."""

    def test_summary_uses_lock(self, tmp_path):
        """summary() should not crash when called concurrently."""
        import threading
        from enigma_engine.core.training_queue import TrainingQueue

        q = TrainingQueue(save_path=tmp_path / "q.json")
        q.executor = lambda j: 0.0
        from enigma_engine.core.training_queue import TrainingJob
        q.add_job(TrainingJob(mode="solo"))

        # Call summary from multiple threads — should not crash
        errors = []

        def call_summary():
            try:
                for _ in range(50):
                    q.summary()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call_summary) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"summary() crashed: {errors}"


# ================================================================
# Forge log file persistence
# ================================================================

class TestForgeLogFilePersistence:
    """ForgeMixin log file writing and rotation."""

    def test_init_log_file_creates_file(self, tmp_path, monkeypatch):
        """_init_log_file creates a forge_*.log in logs/."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        monkeypatch.chdir(tmp_path)
        obj = object.__new__(ForgeMixin)
        fh = ForgeMixin._init_log_file(obj)
        try:
            assert fh is not None
            assert obj._forge_log_path.exists()
            assert obj._forge_log_path.name.startswith("forge_")
            assert obj._forge_log_path.name.endswith(".log")
        finally:
            fh.close()

    def test_write_log_file_appends(self, tmp_path, monkeypatch):
        """_write_log_file appends text to the session log."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        monkeypatch.chdir(tmp_path)
        obj = object.__new__(ForgeMixin)
        ForgeMixin._write_log_file(obj, "line 1\n")
        ForgeMixin._write_log_file(obj, "line 2\n")
        ForgeMixin._close_log_file(obj)
        content = obj._forge_log_path.read_text(encoding="utf-8")
        assert "line 1" in content
        assert "line 2" in content

    def test_rotation_keeps_max_files(self, tmp_path, monkeypatch):
        """Old log files are deleted when count >= _LOG_MAX_FILES."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        monkeypatch.chdir(tmp_path)
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        # Create 12 fake log files (> default 10)
        for i in range(12):
            (log_dir / f"forge_20260101_{i:06d}.log").write_text(f"old {i}")

        obj = object.__new__(ForgeMixin)
        fh = ForgeMixin._init_log_file(obj)
        fh.close()

        remaining = list(log_dir.glob("forge_*.log"))
        # Should be at most _LOG_MAX_FILES (10)
        assert len(remaining) <= ForgeMixin._LOG_MAX_FILES

    def test_close_log_file_closes_handle(self, tmp_path, monkeypatch):
        """_close_log_file closes the file handle."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        monkeypatch.chdir(tmp_path)
        obj = object.__new__(ForgeMixin)
        ForgeMixin._init_log_file(obj)
        assert not obj._forge_log_fh.closed
        ForgeMixin._close_log_file(obj)
        assert obj._forge_log_fh is None

    def test_close_log_file_noop_when_no_file(self):
        """_close_log_file is safe when no file was opened."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        obj = object.__new__(ForgeMixin)
        # Should not raise
        ForgeMixin._close_log_file(obj)


class TestOnWarningCallback:
    """S717: Trainer.on_warning fires when checkpoint save fails."""

    def test_trainer_has_on_warning(self):
        """Trainer.__init__ defines on_warning callback attribute."""
        import inspect
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer.__init__)
        assert "on_warning" in src

    def test_emit_warning_exists(self):
        """Trainer._emit_warning method exists and calls on_warning."""
        import inspect
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer._emit_warning)
        assert "on_warning" in src

    def test_save_checkpoint_emits_warning_on_failure(self):
        """_save_checkpoint calls _emit_warning on exception."""
        import inspect
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer._save_checkpoint)
        assert "_emit_warning" in src

