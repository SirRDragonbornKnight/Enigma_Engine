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


# ---------------------------------------------------------------------------
# ARCH-1d: API-routing branch wiring tests
# ---------------------------------------------------------------------------

def test_solo_training_has_api_routing_branch() -> None:
    """_start_solo_training must contain the ARCH-1d API-routing gate.

    Guards against a regression where use_api_chat check or the client.train
    call is accidentally removed.
    """
    src = inspect.getsource(ForgeTrainingMixin._start_solo_training)

    assert "use_api_chat" in src, "API routing gate missing"
    # The code uses getattr(self, "_get_api_chat_client", None) —
    # gate the string literal form which is the canonical pattern.
    assert '"_get_api_chat_client"' in src, (
        "_get_api_chat_client reference missing in solo training")
    assert re.search(r'client\.train\(', src), (
        "client.train( call missing in solo training")
    assert re.search(r'_poll_api_training_status\(', src), (
        "_poll_api_training_status( call missing in solo training")


def test_poll_api_training_status_helper_gates_right_calls() -> None:
    """_poll_api_training_status must call training_status() and update progress."""
    src = inspect.getsource(ForgeTrainingMixin._poll_api_training_status)

    assert re.search(r'client\.training_status\(\)', src), (
        "training_status() call missing in polling helper")
    assert re.search(r'_update_forge_progress\(', src), (
        "_update_forge_progress( call missing in polling helper")
    assert re.search(r'_refresh_models', src), (
        "_refresh_models call missing in polling helper")
    assert re.search(r'training_active', src), (
        "training_active loop guard missing in polling helper")


def test_lora_training_has_api_routing_branch() -> None:
    """_start_lora_training must include API routing and polling calls."""
    src = inspect.getsource(ForgeTrainingMixin._start_lora_training)

    assert "use_api_chat" in src
    assert '"_get_api_chat_client"' in src
    assert re.search(r'client\.train\(', src)
    assert re.search(r'_poll_api_training_status\(', src)
    assert re.search(r'mode_label\s*=\s*"LoRA"', src)


# ================================================================
# ARCH-1d Slice 3: RL/Preference Mode API Routing Tests
# ================================================================

def test_grpo_training_has_api_routing_branch() -> None:
    """_start_grpo_training must route through API when enabled."""
    from enigma_engine.gui.gui_forge_new_modes import (
        ForgeNewModesMixin)
    src = inspect.getsource(ForgeNewModesMixin._start_rl_variant_training)

    assert "use_api_chat" in src
    assert re.search(r'client\.train\(', src)
    assert re.search(r'_poll_api_training_status\(', src)
    assert re.search(r'mode_label\s*=\s*algo\.upper\(\)', src)


def test_remax_training_uses_shared_rl_handler() -> None:
    """_start_remax_training calls _start_rl_variant_training."""
    from enigma_engine.gui.gui_forge_new_modes import (
        ForgeNewModesMixin)
    src = inspect.getsource(ForgeNewModesMixin._start_remax_training)

    assert "_start_rl_variant_training" in src
    assert '"ReMax"' in src


def test_rlhf_training_has_api_routing_branch() -> None:
    """_start_rlhf_training must route through API when enabled."""
    from enigma_engine.gui.gui_forge_new_modes import (
        ForgeNewModesMixin)
    src = inspect.getsource(ForgeNewModesMixin._start_rlhf_training)

    assert "use_api_chat" in src
    assert re.search(r'client\.train\(', src)
    assert re.search(r'_poll_api_training_status\(', src)
    assert re.search(r'mode_label\s*=\s*"RLHF"', src)
    assert re.search(r'"mode"\s*:\s*"rlhf"', src)


def test_selfplay_training_has_api_routing_branch() -> None:
    """_start_selfplay_training must route through API when enabled."""
    from enigma_engine.gui.gui_forge_new_modes import (
        ForgeNewModesMixin)
    src = inspect.getsource(ForgeNewModesMixin._start_selfplay_training)

    assert "use_api_chat" in src
    assert re.search(r'client\.train\(', src)
    assert re.search(r'_poll_api_training_status\(', src)
    assert re.search(r'mode_label\s*=\s*"SELF-PLAY"', src)
    assert re.search(r'"mode"\s*:\s*"self_play"', src)


def test_simpo_training_uses_shared_pref_handler() -> None:
    """_start_simpo_training calls _start_preference_variant_training."""
    from enigma_engine.gui.gui_forge_new_modes import (
        ForgeNewModesMixin)
    src = inspect.getsource(ForgeNewModesMixin._start_simpo_training)

    assert "_start_preference_variant_training" in src
    assert '"SimPO"' in src


def test_orpo_training_uses_shared_pref_handler() -> None:
    """_start_orpo_training calls _start_preference_variant_training."""
    from enigma_engine.gui.gui_forge_new_modes import (
        ForgeNewModesMixin)
    src = inspect.getsource(ForgeNewModesMixin._start_orpo_training)

    assert "_start_preference_variant_training" in src
    assert '"ORPO"' in src


def test_preference_variant_handler_has_api_routing() -> None:
    """_start_preference_variant_training (SimPO/ORPO) routes through API."""
    from enigma_engine.gui.gui_forge_new_modes import (
        ForgeNewModesMixin)
    src = inspect.getsource(
        ForgeNewModesMixin._start_preference_variant_training)

    assert "use_api_chat" in src
    assert re.search(r'client\.train\(', src)
    assert re.search(r'_poll_api_training_status\(', src)
    assert re.search(r'algo\.upper\(\)', src)
    assert re.search(r'algo\.lower\(\)', src)
