"""Structural wiring tests for Forge training launchers.

GUI launcher methods spin threads and depend on large runtime state,
so these tests gate key call-site wiring using source inspection.
"""

from __future__ import annotations

import inspect
import re

from enigma_engine.gui.gui_forge_training import ForgeTrainingMixin


def test_solo_training_routes_through_dispatcher() -> None:
    src = inspect.getsource(ForgeTrainingMixin._start_solo_training)

    assert "run_training(" in src
    assert re.search(r'"mode"\s*:\s*"sft"', src)



def test_dpo_training_routes_through_dispatcher_and_forwards_loss_type() -> None:
    src = inspect.getsource(ForgeTrainingMixin._start_dpo_training)

    assert "run_training(" in src
    assert re.search(r'"mode"\s*:\s*"dpo"', src)
    assert re.search(r'"loss_type"\s*:\s*loss_type', src)
    assert not re.search(r'\btrainer\.train_dpo\(', src)


def test_vision_training_routes_through_dispatcher() -> None:
    src = inspect.getsource(ForgeTrainingMixin._start_vision_training)

    assert "run_training(" in src
    assert re.search(r'"mode"\s*:\s*"vision"', src)


def test_lora_training_routes_primary_path_through_dispatcher() -> None:
    src = inspect.getsource(ForgeTrainingMixin._start_lora_training)

    assert "run_training(" in src
    assert re.search(r'"mode"\s*:\s*"lora"', src)


def test_lora_fallback_uses_strict_dispatcher_payload_shape() -> None:
    """ImportError fallback must emit mode/data/training payload shape.

    This guards against flat keys like data_text/epochs that fail
    TrainingJobConfig(extra='forbid') validation.
    """
    src = inspect.getsource(ForgeTrainingMixin._start_lora_training)

    assert re.search(r'"mode"\s*:\s*"sft"', src)
    assert re.search(r'"data"\s*:\s*text', src)
    assert re.search(r'"training"\s*:\s*\{', src)
    assert '"data_text"' not in src
