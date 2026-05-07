"""Structural wiring tests for Forge new training modes launchers."""

from __future__ import annotations

import inspect
import re

from enigma_engine.gui.gui_forge_new_modes import ForgeNewModesMixin


def test_start_grpo_training_delegates_to_shared_rl_handler() -> None:
    src = inspect.getsource(ForgeNewModesMixin._start_grpo_training)

    assert "_start_rl_variant_training(\"GRPO\")" in src


def test_start_remax_training_delegates_to_shared_rl_handler() -> None:
    src = inspect.getsource(ForgeNewModesMixin._start_remax_training)

    assert "_start_rl_variant_training(\"ReMax\")" in src



def test_shared_rl_handler_routes_grpo_through_dispatcher() -> None:
    src = inspect.getsource(ForgeNewModesMixin._start_rl_variant_training)

    assert "run_training(" in src
    assert 'mode_name = "grpo" if algo == "GRPO" else "remax"' in src
    assert re.search(r'"mode"\s*:\s*mode_name', src)
