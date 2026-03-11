"""Tests for training evaluation features."""
import pytest


class TestTrainingEvaluation:
    """Test before/after training evaluation."""

    def test_evaluate_model_exists(self):
        """evaluate_model function is importable."""
        from enigma_engine.core.training_evaluation import evaluate_model
        assert callable(evaluate_model)

    def test_evaluate_model_returns_dict(self):
        """evaluate_model returns dict with expected keys."""
        from enigma_engine.core.training_evaluation import evaluate_model
        # Test with empty prompts
        result = evaluate_model(None, None, [], device="cpu")
        assert isinstance(result, dict)
        assert "perplexity" in result
        assert "loss" in result
        assert "num_prompts" in result

    def test_default_test_prompts_exist(self):
        """DEFAULT_TEST_PROMPTS is available."""
        from enigma_engine.core.training_evaluation import DEFAULT_TEST_PROMPTS
        assert isinstance(DEFAULT_TEST_PROMPTS, list)
        assert len(DEFAULT_TEST_PROMPTS) > 0
        assert all(isinstance(p, str) for p in DEFAULT_TEST_PROMPTS)

    def test_training_config_has_evaluation_option(self):
        """TrainingConfig has run_evaluation field."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert hasattr(config, "run_evaluation")
        assert isinstance(config.run_evaluation, bool)

    def test_training_config_has_eval_prompts_option(self):
        """TrainingConfig has eval_test_prompts field."""
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert hasattr(config, "eval_test_prompts")

    def test_trainer_calls_evaluate_when_enabled(self):
        """Trainer.train() calls evaluate_model if run_evaluation=True."""
        import inspect
        from enigma_engine.core.training import Trainer
        source = inspect.getsource(Trainer.train)
        assert "run_evaluation" in source
        assert "evaluate_model" in source or "before_eval" in source

    def test_solo_training_enables_evaluation(self):
        """_start_solo_training enables run_evaluation=True."""
        import inspect
        from enigma_engine.gui.gui_forge_training import ForgeTrainingMixin
        source = inspect.getsource(ForgeTrainingMixin._start_solo_training)
        assert "run_evaluation" in source

    def test_lora_training_enables_evaluation(self):
        """_start_lora_training enables run_evaluation=True."""
        import inspect
        from enigma_engine.gui.gui_forge_training import ForgeTrainingMixin
        source = inspect.getsource(ForgeTrainingMixin._start_lora_training)
        assert "run_evaluation" in source

    def test_solo_training_logs_evaluation_results(self):
        """_start_solo_training logs before/after evaluation."""
        import inspect
        from enigma_engine.gui.gui_forge_training import ForgeTrainingMixin
        source = inspect.getsource(ForgeTrainingMixin._start_solo_training)
        assert "before_eval" in source
        assert "after_eval" in source
        assert "EVALUATION RESULTS" in source or "perplexity" in source.lower()

    def test_lora_training_logs_evaluation_results(self):
        """_start_lora_training logs before/after evaluation."""
        import inspect
        from enigma_engine.gui.gui_forge_training import ForgeTrainingMixin
        source = inspect.getsource(ForgeTrainingMixin._start_lora_training)
        assert "before_eval" in source
        assert "after_eval" in source

    def test_evaluate_tool_usage_exists(self):
        """evaluate_tool_usage function is available."""
        from enigma_engine.core.training_evaluation import evaluate_tool_usage
        assert callable(evaluate_tool_usage)

    def test_evaluate_tool_usage_returns_dict(self):
        """evaluate_tool_usage returns dict with success metrics."""
        from enigma_engine.core.training_evaluation import evaluate_tool_usage
        result = evaluate_tool_usage(None, None, None, [], device="cpu")
        assert isinstance(result, dict)
        assert "success_rate" in result
        assert "total_tests" in result
        assert "successes" in result
        assert "failures" in result
