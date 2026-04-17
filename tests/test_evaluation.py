"""Tests for training evaluation features."""
import pytest
from unittest.mock import MagicMock


class TestTrainingEvaluation:
    """Test before/after training evaluation."""

    def test_evaluate_model_returns_dict(self):
        """evaluate_model returns dict with expected keys."""
        from enigma_engine.core.training_evaluation import evaluate_model
        result = evaluate_model(None, None, [], device="cpu")
        assert isinstance(result, dict)
        assert "perplexity" in result
        assert "loss" in result
        assert "num_prompts" in result

    def test_default_test_prompts_exist(self):
        """DEFAULT_TEST_PROMPTS is available and populated."""
        from enigma_engine.core.training_evaluation import DEFAULT_TEST_PROMPTS
        assert isinstance(DEFAULT_TEST_PROMPTS, list)
        assert len(DEFAULT_TEST_PROMPTS) > 0
        assert all(isinstance(p, str) for p in DEFAULT_TEST_PROMPTS)

    def test_training_config_evaluation_defaults(self):
        """TrainingConfig has evaluation fields with correct defaults."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert isinstance(config.run_evaluation, bool)
        assert hasattr(config, "eval_test_prompts")

    def test_evaluate_tool_usage_returns_dict(self):
        """evaluate_tool_usage returns dict with success metrics."""
        from enigma_engine.core.training_evaluation import evaluate_tool_usage
        result = evaluate_tool_usage(None, None, None, [], device="cpu")
        assert isinstance(result, dict)
        assert "success_rate" in result
        assert "total_tests" in result
        assert "successes" in result
        assert "failures" in result


class TestEvalModelModeRestore:
    """Evaluation functions must restore model.training mode."""

    def _make_model_and_tokenizer(self):
        """Create a tiny model + tokenizer for testing."""
        import torch.nn as nn

        model = nn.Linear(4, 4)

        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        tok.eos_token_id = 0
        return model, tok

    def test_evaluate_model_restores_train_mode(self):
        """evaluate_model restores model.train() after running."""
        from enigma_engine.core.training_evaluation import evaluate_model

        model, tok = self._make_model_and_tokenizer()
        model.train()
        assert model.training

        # Non-empty prompts so model.eval() is actually called
        evaluate_model(model, tok, ["hello world"], device="cpu")
        assert model.training, "model should be back in training mode"

    def test_evaluate_model_preserves_eval_mode(self):
        """evaluate_model leaves model in eval mode if it started that way."""
        from enigma_engine.core.training_evaluation import evaluate_model

        model, tok = self._make_model_and_tokenizer()
        model.eval()
        assert not model.training

        evaluate_model(model, tok, ["hello world"], device="cpu")
        assert not model.training, "model should remain in eval mode"

    def test_evaluate_tool_usage_restores_train_mode(self):
        """evaluate_tool_usage restores model.train() after running."""
        from enigma_engine.core.training_evaluation import evaluate_tool_usage

        model, _ = self._make_model_and_tokenizer()
        model.train()
        assert model.training

        engine = MagicMock()
        engine.generate.return_value = "search.web result"
        cases = [{"prompt": "test", "expected_command": "search.web"}]
        evaluate_tool_usage(model, None, engine, cases, device="cpu")
        assert model.training, "model should be back in training mode"

    def test_run_golden_eval_restores_train_mode(self, tmp_path):
        """run_golden_eval restores model.train() after running."""
        import json
        from enigma_engine.core.training_evaluation import run_golden_eval

        model, tok = self._make_model_and_tokenizer()
        model.train()
        assert model.training

        # Provide a real golden file with cases so model.eval() is reached
        golden = tmp_path / "golden.json"
        golden.write_text(
            json.dumps([{"prompt": "test", "expected": ["x"]}]),
            encoding="utf-8")
        run_golden_eval(model, tok, golden, device="cpu")
        assert model.training, "model should be back in training mode"


class TestEvalBugFixes:
    """Tests for S730 and S731 bug fixes."""

    def test_num_prompts_excludes_skipped(self):
        """S730: num_prompts should only count actually evaluated prompts."""
        import torch
        from enigma_engine.core.training_evaluation import evaluate_model

        # Model that returns valid logits for any input
        model = MagicMock()
        model.training = False
        model.eval = MagicMock()
        model.train = MagicMock()
        # Return logits shaped (1, seq_len-1, vocab_size=10)
        model.return_value = torch.randn(1, 4, 10)

        tok = MagicMock()
        # First prompt: 1 token → skipped (need ≥ 2)
        # Second prompt: 5 tokens → evaluated
        tok.encode.side_effect = [[1], [1, 2, 3, 4, 5]]

        result = evaluate_model(model, tok, ["a", "hello world"],
                                device="cpu")
        assert result["num_prompts"] == 1  # Only the 5-token prompt

    def test_golden_eval_skips_empty_expected(self, tmp_path):
        """S731: Cases with empty 'expected' should be skipped entirely."""
        import json
        from enigma_engine.core.training_evaluation import run_golden_eval
        import torch.nn as nn

        model = nn.Linear(4, 4)
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3]
        tok.decode.return_value = "anything"
        tok.eos_token_id = 0  # stops immediately

        golden = tmp_path / "golden.json"
        golden.write_text(
            json.dumps([{"prompt": "test", "expected": []}]),
            encoding="utf-8")

        result = run_golden_eval(model, tok, golden, device="cpu")
        # Empty expected = no constraint = meaningless test → skip
        assert result["total"] == 0
