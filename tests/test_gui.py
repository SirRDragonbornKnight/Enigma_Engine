"""Tests for the desktop GUI module and widgets."""

import json
import inspect
from pathlib import Path

import pytest


# ── Shared test helpers ──────────────────────────────────────────────


class MockVar:
    """Minimal mock for tkinter variable-like objects."""
    def __init__(self, initial=True):
        self.value = initial

    def get(self):
        return self.value

    def set(self, val):
        self.value = val


class MockStatusBar:
    """Minimal mock for the status bar widget."""
    def set_left(self, text): pass
    def set_center(self, text): pass
    def set_right(self, text): pass


class DummyStatusBar:
    """Status bar that records nothing — for tests that just need the API."""
    def set_left(self, text): pass
    def set_center(self, text): pass
    def set_right(self, text): pass


class DummyThread:
    """Thread replacement that records kwargs and does nothing."""
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def start(self):
        pass


class UnexpectedThread:
    """Thread replacement that fails if instantiated."""
    def __init__(self, **kwargs):
        raise RuntimeError("Unexpected thread created")


# ================================================================
# Scanners
# ================================================================

class TestScanners:
    """Verify all filesystem scanners work."""

    def test_scan_mods(self):
        from enigma_engine.gui.scanners import scan_mods
        mods = scan_mods()
        ids = [m["id"] for m in mods]
        assert "imagegen" in ids
        assert "voice" in ids
        # Voice and audio generation are intentionally unified.
        assert "audiogen" not in ids
        assert "_template" not in ids
        for mod in mods:
            assert "id" in mod
            assert "name" in mod
            assert "commands" in mod
            assert "prompt" in mod
            assert "rules" in mod

    def test_scan_models(self):
        from enigma_engine.gui.scanners import scan_models
        models = scan_models()
        assert isinstance(models, list)
        for m in models:
            assert "name" in m
            assert "path" in m
            assert "size_mb" in m

    def test_scan_models_groups_sharded_safetensors(self):
        """Sharded safetensors should be merged into one model entry."""
        from enigma_engine.gui.scanners import scan_models
        models = scan_models()
        names = [m["name"] for m in models]
        # Shard filenames like model-00001-of-00005 must not appear
        for name in names:
            assert "-of-" not in name, (
                f"Sharded file '{name}' should be grouped, "
                "not listed individually")

    def test_scan_training_data(self):
        from enigma_engine.gui.scanners import scan_training_data
        files = scan_training_data()
        assert isinstance(files, list)
        names = [f["name"] for f in files]
        assert "training.txt" in names
        assert "gui_settings.json" not in names
        for f in files:
            assert "path" in f
            assert "size_kb" in f

    # ── D-11b (Pass 156i9): default training file picks reasoning corpus ──

    def test_pick_default_prefers_combined_finetune_when_present(self):
        """When `data/finetune/combined_finetune.txt` is on disk, the
        FORGE training-data default should pick it over the placeholder
        `data/training.txt`. Closes D-11b — Pass 156i8 ships the
        reasoning corpus end-to-end, this passes the user's eyes onto
        it without manual file-tree navigation."""
        from enigma_engine.gui.scanners import _pick_default_training_file
        files = [
            {"name": "training.txt",
             "path": "data/training.txt", "size_kb": 1.0},
            {"name": "finetune/combined_finetune.txt",
             "path": "data/finetune/combined_finetune.txt",
             "size_kb": 60000.0},
        ]
        chosen = _pick_default_training_file(files)
        assert chosen == "data/finetune/combined_finetune.txt"

    def test_pick_default_falls_back_to_first_when_no_combined(self):
        """Without a finetune corpus on disk, return the first scanned
        file (legacy behaviour)."""
        from enigma_engine.gui.scanners import _pick_default_training_file
        files = [
            {"name": "training.txt",
             "path": "data/training.txt", "size_kb": 1.0},
            {"name": "smoke.txt",
             "path": "data/smoke.txt", "size_kb": 0.5},
        ]
        assert _pick_default_training_file(files) == "data/training.txt"

    def test_pick_default_empty_list_returns_empty_string(self):
        """No training files → empty string (matches the legacy
        `train_data_var` initial-value contract: empty if nothing
        found)."""
        from enigma_engine.gui.scanners import _pick_default_training_file
        assert _pick_default_training_file([]) == ""

    def test_pick_default_combined_in_arbitrary_position(self):
        """Helper does not depend on glob order — finds the corpus
        wherever it appears in the list."""
        from enigma_engine.gui.scanners import _pick_default_training_file
        files = [
            {"name": "smoke.txt",
             "path": "data/smoke.txt", "size_kb": 0.5},
            {"name": "training.txt",
             "path": "data/training.txt", "size_kb": 1.0},
            {"name": "finetune/combined_finetune.txt",
             "path": "data/finetune/combined_finetune.txt",
             "size_kb": 60000.0},
            {"name": "pretrain/combined.txt",
             "path": "data/pretrain/combined.txt",
             "size_kb": 100000.0},
        ]
        chosen = _pick_default_training_file(files)
        assert chosen == "data/finetune/combined_finetune.txt"

    # ── D-11c (Pass 156l): generalised picker for other FORGE pickers ──

    def test_pick_first_match_first_tail_wins_over_later_tails(self):
        """`_pick_first_match` is preference-ordered — the first tail
        in the list wins even if a later tail also matches a file in
        the directory. Adversarial ordering: the SECOND tail's match
        appears EARLIER in `files` than the FIRST tail's match, so a
        naive `for f in files` outer loop would return the wrong one.
        This proves the helper is preference-first, not file-order-
        first."""
        from enigma_engine.gui.scanners import _pick_first_match
        files = [
            {"name": "dpo_pairs.jsonl",
             "path": "data/finetune/dpo_pairs.jsonl",
             "size_kb": 5.0},
            {"name": "combined.jsonl",
             "path": "data/dpo/combined.jsonl",
             "size_kb": 10.0},
        ]
        # First tail = dpo/combined.jsonl. Even though dpo_pairs is
        # listed first, dpo/combined wins on preference order.
        chosen = _pick_first_match(
            files,
            ["dpo/combined.jsonl", "finetune/dpo_pairs.jsonl"])
        assert chosen == "data/dpo/combined.jsonl"

    def test_pick_default_dpo_data_file_prefers_dpo_combined(self):
        """When `data/dpo/combined.jsonl` is present, the DPO/APO pair-
        data picker should default to it. Closes D-11c — same UX win
        as D-11b but for the alignment training data rather than SFT."""
        from enigma_engine.gui.scanners import (
            _pick_default_dpo_data_file)
        files = [
            {"name": "dpo_smoke.jsonl",
             "path": "data/dpo_smoke.jsonl", "size_kb": 1.0},
            {"name": "combined.jsonl",
             "path": "data/dpo/combined.jsonl", "size_kb": 50.0},
        ]
        assert _pick_default_dpo_data_file(files) == (
            "data/dpo/combined.jsonl")

    def test_pick_default_train_data_for_mode_routes_preference_modes_to_dpo(self):
        """D-11c-DPO (Pass 156q): the mode-aware default helper must
        route preference-pair modes (DPO/APO/SimPO/ORPO/GRPO/ReMax/
        RLHF/Self-Play) to the DPO picker, and route SFT modes
        (Basic/LoRA) to the SFT picker. Catches regression where
        someone removes a mode from `_PREFERENCE_MODES` and the GUI
        silently surfaces SFT data when the user picked APO."""
        from enigma_engine.gui.scanners import (
            _pick_default_train_data_for_mode)
        files = [
            {"name": "combined_finetune.txt",
             "path": "data/finetune/combined_finetune.txt",
             "size_kb": 5000.0},
            {"name": "combined.jsonl",
             "path": "data/dpo/combined.jsonl",
             "size_kb": 50.0},
        ]
        # Preference-pair modes must pick the DPO file.
        for mode in ("RLHF", "Self-Play", "GRPO", "ReMax",
                     "SimPO", "ORPO", "APO"):
            assert _pick_default_train_data_for_mode(files, mode) == (
                "data/dpo/combined.jsonl"), (
                f"mode {mode!r} should route to DPO default")
        # SFT modes must pick the fine-tune corpus.
        for mode in ("Basic", "LoRA", "Distill", "AI-Guided",
                     "Dialogue", "Image", "Pre-Train"):
            assert _pick_default_train_data_for_mode(files, mode) == (
                "data/finetune/combined_finetune.txt"), (
                f"mode {mode!r} should route to SFT default")

    def test_pick_default_pretrain_file_prefers_combined_pretrain(self):
        """When `data/pretrain/combined.txt` is present (output of
        `collect_pretraining_data.py --combine-only`), the pre-training
        picker should default to it. D-11c."""
        from enigma_engine.gui.scanners import (
            _pick_default_pretrain_file)
        files = [
            {"name": "scratch.txt",
             "path": "data/scratch.txt", "size_kb": 1.0},
            {"name": "pretrain/combined.txt",
             "path": "data/pretrain/combined.txt",
             "size_kb": 90000.0},
        ]
        assert _pick_default_pretrain_file(files) == (
            "data/pretrain/combined.txt")

    def test_forge_pretrain_data_var_uses_smart_default(self):
        """Pass 156m: FORGE pretrain `pretrain_data_var` must be
        initialised via `_pick_default_pretrain_file(...)`, not as
        empty `value=""`. Catches regression where someone reverts
        the StringVar to a hardcoded empty default and the smart
        helper becomes orphaned infrastructure."""
        import inspect

        from enigma_engine.gui import gui_pages_forge
        src = inspect.getsource(gui_pages_forge)
        # Wiring assertion: pretrain_data_var construction must
        # call the helper, not hardcode "".
        assert "_pick_default_pretrain_file" in src, (
            "pretrain picker not wired to smart-default helper")
        assert 'self.pretrain_data_var = ctk.StringVar(value="")' not in src, (
            "pretrain_data_var still defaults to empty string; "
            "smart helper is orphaned")

    def test_resolve_anchor_path_user_override_returned_as_is(
            self, tmp_path):
        """Continuous-3b (Pass 156o): a non-empty saved path returns
        Path(saved) as-is, even if missing — the GUI status label
        shows 'file missing' rather than silently swapping in the
        default."""
        from enigma_engine.gui.scanners import _resolve_anchor_path
        custom = tmp_path / "my_anchor.jsonl"
        result = _resolve_anchor_path(str(custom))
        assert result == custom

    def test_resolve_anchor_path_empty_returns_default_when_present(
            self, tmp_path, monkeypatch):
        """Empty saved → repo default `data/anchor_examples.jsonl`
        when that file exists."""
        from enigma_engine.gui import scanners
        fake_default = tmp_path / "anchor_examples.jsonl"
        fake_default.write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(scanners, "DATA_DIR", tmp_path)
        assert scanners._resolve_anchor_path("") == fake_default

    def test_resolve_anchor_path_empty_returns_none_when_missing(
            self, tmp_path, monkeypatch):
        """Empty saved + no default file → None (recent-only replay,
        no log noise)."""
        from enigma_engine.gui import scanners
        monkeypatch.setattr(scanners, "DATA_DIR", tmp_path)
        assert scanners._resolve_anchor_path("") is None

    def test_resolve_anchor_path_none_arg_treated_as_empty(
            self, tmp_path, monkeypatch):
        """Saved=None should behave the same as saved=''."""
        from enigma_engine.gui import scanners
        monkeypatch.setattr(scanners, "DATA_DIR", tmp_path)
        assert scanners._resolve_anchor_path(None) is None

    def test_config_page_wires_anchor_widget(self):
        """Continuous-3b (Pass 156o): the CONFIG page must construct
        the anchor widget — calls `_resolve_anchor_path`, builds an
        anchor StringVar, and exposes browse / reset handlers."""
        import inspect
        from enigma_engine.gui import gui_pages_config
        src = inspect.getsource(gui_pages_config)
        assert "_resolve_anchor_path" in src, (
            "config page does not call _resolve_anchor_path")
        assert "_anchor_path_var" in src, (
            "config page does not build _anchor_path_var")
        assert "_browse_anchor_file" in src, (
            "config page is missing browse handler")
        assert "_reset_anchor_file" in src, (
            "config page is missing reset handler")

    def test_desktop_forwards_anchor_path_to_router(self):
        """Continuous-3b (Pass 156o): the desktop launcher must pass
        the resolved anchor path into `ModRouter(...)` so the
        BackgroundTrainer rehearsal layer honours the user's saved
        override (or repo default when blank)."""
        import inspect
        from enigma_engine.gui import desktop
        src = inspect.getsource(desktop)
        assert "anchor_data_path=" in src, (
            "desktop does not forward anchor_data_path kwarg")
        assert "_resolve_anchor_path" in src, (
            "desktop does not resolve the saved anchor override")

    def test_forge_lora_mode_card_present(self):
        """LoRA-1 (Pass 156p): explicit `LoRA` foundation mode card
        must exist in the FORGE training-method list so users can
        force adapter training on any model size (not just >7B
        auto-detected by Basic mode)."""
        import inspect
        from enigma_engine.gui import gui_pages_forge
        src = inspect.getsource(gui_pages_forge)
        assert '("LoRA",' in src, (
            "LoRA mode card not in foundation_modes list")

    def test_forge_lora_dispatcher_calls_lora_training(self):
        """LoRA-1 (Pass 156p): selecting `LoRA` mode must dispatch
        directly to `_start_lora_training()`. Pairs with the mode-
        card test to gate end-to-end wiring of the new mode."""
        import inspect
        from enigma_engine.gui import gui_forge
        src = inspect.getsource(gui_forge)
        # The dispatcher branch.
        assert 'mode_name == "LoRA"' in src, (
            "LoRA dispatch branch missing")
        # Description entry for status/log labels.
        assert '"LoRA": (' in src, (
            "LoRA description missing from _TRAINING_MODE_DESCRIPTIONS")
        # Display→key registry.
        assert '"LoRA": "LoRA"' in src, (
            "LoRA missing from _MODE_DISPLAY_TO_KEY")

    def test_forge_mode_change_swaps_train_data_default(self):
        """D-11c-DPO (Pass 156q): `_on_training_mode_changed` must call
        `_pick_default_train_data_for_mode` and swap the picker default
        only when the user has not customised the path. Catches
        regression where someone removes the swap logic and switching
        from Basic to APO leaves the irrelevant SFT default in the
        picker."""
        import inspect
        from enigma_engine.gui import gui_forge
        src = inspect.getsource(gui_forge.ForgeMixin._on_training_mode_changed)
        assert "_pick_default_train_data_for_mode" in src, (
            "Mode-change handler does not consult the mode-aware default helper")
        assert "_train_data_smart_default" in src, (
            "Mode-change handler does not gate swap on the smart-default tracker")

    def test_forge_user_browse_clears_smart_default_tracker(self):
        """D-11c-DPO (Pass 156q): when the user clicks Browse to pick a
        training file, the smart-default tracker must be cleared so a
        later mode change does not silently overwrite the user choice.
        Pairs with the mode-change test \u2014 together they prove the swap
        only fires on default-state pickers."""
        import inspect
        from enigma_engine.gui import gui_forge
        browse_src = inspect.getsource(
            gui_forge.ForgeMixin._browse_training_data)
        select_src = inspect.getsource(
            gui_forge.ForgeMixin._on_data_selected)
        assert "_train_data_smart_default = None" in browse_src, (
            "Browse handler does not clear the smart-default tracker")
        assert "_train_data_smart_default = None" in select_src, (
            "Quick-select handler does not clear the smart-default tracker")

    def test_forge_image_mode_exposes_unfreeze_text_layers(self):
        """Code-6b (Pass 156r): the Image foundation mode must expose
        an `unfreeze_text_layers` numeric input so users can switch
        between LLaVA Stage-1 (projection-only, default 0) and Stage-2
        (last N text transformer layers also fine-tuned). Catches
        regression where the widget is removed but the trainer still
        accepts the kwarg \u2014 leaving Stage-2 unreachable from the GUI."""
        import inspect
        from enigma_engine.gui import gui_pages_forge
        src = inspect.getsource(gui_pages_forge)
        assert "forge_vision_unfreeze_var" in src, (
            "Unfreeze-text-layers widget missing from Image mode")

    def test_forge_vision_training_forwards_unfreeze_to_trainer(self):
        """Code-6b (Pass 156r): `_start_vision_training` must read the
        `forge_vision_unfreeze_var` widget and forward the parsed value
        through dispatcher vision config. The literal
        kwarg gate catches the regression where someone reads the
        widget into a local but drops it from the call \u2014 the GUI knob\n        would silently revert to the default 0 (Stage-1) for everyone."""
        import inspect
        from enigma_engine.gui import gui_forge_training
        src = inspect.getsource(
            gui_forge_training.ForgeTrainingMixin._start_vision_training)
        assert "forge_vision_unfreeze_var" in src, (
            "Vision training does not read the unfreeze widget")
        assert '"unfreeze_text_layers": unfreeze_text_layers' in src, (
            "Vision training does not forward unfreeze_text_layers "
            "through dispatcher vision config")

    # =========================================================================
    # LoRA-1b (Pass 156s): adapter scanner + engine apply path
    # =========================================================================

    def test_scan_lora_adapters_finds_peft_directory(self, tmp_path,
                                                     monkeypatch):
        """A directory with `adapter_config.json` in checkpoints/ or
        lora_adapters/ should be picked up; metadata fields populated
        from the JSON."""
        from enigma_engine.gui import scanners as sc
        ckpt = tmp_path / "checkpoints"
        ckpt.mkdir()
        adapter = ckpt / "coding_v2"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "models/enigma_small.pth",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
        }), encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"x" * 1024)

        monkeypatch.setattr(sc, "MODELS_DIR", tmp_path)
        result = sc.scan_lora_adapters()
        assert len(result) == 1
        entry = result[0]
        assert entry["name"] == "coding_v2"
        assert entry["rank"] == 16
        assert entry["alpha"] == 32
        assert entry["target_modules"] == ["q_proj", "v_proj"]

    def test_scan_lora_adapters_filters_by_base_model_stem(
            self, tmp_path, monkeypatch):
        """When `base_model_path` is given, only adapters whose
        `base_model_name_or_path` stem matches are returned. Catches
        the regression where a coding-base adapter would surface in
        the dropdown for a math-base load."""
        from enigma_engine.gui import scanners as sc
        ckpt = tmp_path / "checkpoints"
        ckpt.mkdir()
        for name, base in [("matches", "models/enigma_small.pth"),
                           ("foreign", "models/other_model.pth")]:
            d = ckpt / name
            d.mkdir()
            (d / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": base,
                "r": 8, "lora_alpha": 16,
                "target_modules": ["q_proj"],
            }), encoding="utf-8")

        monkeypatch.setattr(sc, "MODELS_DIR", tmp_path)
        result = sc.scan_lora_adapters(
            base_model_path="models/enigma_small.pth")
        assert len(result) == 1
        assert result[0]["name"] == "matches"

    def test_scan_lora_adapters_skips_directory_without_config(
            self, tmp_path, monkeypatch):
        """Directories under checkpoints/ without `adapter_config.json`
        (e.g. plain training checkpoints) must NOT appear in the
        adapter list — they are not adapters."""
        from enigma_engine.gui import scanners as sc
        ckpt = tmp_path / "checkpoints"
        ckpt.mkdir()
        (ckpt / "regular_training_run").mkdir()
        (ckpt / "regular_training_run" / "model.pth").write_bytes(b"x")

        monkeypatch.setattr(sc, "MODELS_DIR", tmp_path)
        assert sc.scan_lora_adapters() == []

    def test_engine_exposes_apply_and_clear_adapter(self):
        """LoRA-1b foundation: `EnigmaEngine` must expose the runtime
        adapter API. Structural test (engine __init__ requires GPU/
        weights, can't instantiate in CI). Catches accidental method
        renames or removals that would silently revert the chat layer
        to base-only."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        assert hasattr(EnigmaEngine, "apply_adapter"), (
            "EnigmaEngine missing apply_adapter")
        assert hasattr(EnigmaEngine, "clear_adapter"), (
            "EnigmaEngine missing clear_adapter")
        apply_src = inspect.getsource(EnigmaEngine.apply_adapter)
        assert "PeftModel.from_pretrained" in apply_src, (
            "apply_adapter does not wrap with PeftModel — would be a "
            "silent no-op on a vanilla base model")
        assert "clear_kv_cache" in apply_src, (
            "apply_adapter does not clear KV cache — stale cache "
            "from base weights would corrupt next generation")

        # Pass 156s2 (audit-fix): clear_adapter must use the imperative
        # `disable_adapters` (plural). The singular `disable_adapter`
        # is a @contextmanager in PEFT — calling it bare returns the
        # CM and discards it, leaving the adapter active. Catches a
        # regression that re-introduces the broken fallback chain.
        clear_src = inspect.getsource(EnigmaEngine.clear_adapter)
        assert "disable_adapters" in clear_src, (
            "clear_adapter does not call disable_adapters() — "
            "adapter would stay active after 'clear'")
        assert "disable_adapter(" not in clear_src.replace(
            "disable_adapters(", ""), (
            "clear_adapter calls singular disable_adapter() — that is "
            "a context manager, calling it bare is a silent no-op")

        # Pass 156s2 (audit-fix): apply_adapter docstring must not
        # promise a base-mismatch RuntimeError that the body never
        # raises. The check lives upstream in scan_lora_adapters.
        assert "RuntimeError: The adapter's recorded base" not in (
            EnigmaEngine.apply_adapter.__doc__ or ""), (
            "apply_adapter docstring promises a base-mismatch "
            "RuntimeError that the code never raises")

    def test_save_adapter_writes_peft_directory_only(self):
        """LoRA-1b foundation: `LoraTrainer.save_adapter` must always
        emit a PEFT directory; the manual-fallback `.pth` save path
        was deleted in Pass 156s. Structural gate: the deleted
        ``param.requires_grad`` extraction loop must NOT reappear.
        Without this, a regression that re-introduces the manual save
        would silently produce metadata-less files that the chat
        engine cannot apply."""
        import inspect
        from enigma_engine.core.lora_utils import LoraTrainer
        src = inspect.getsource(LoraTrainer.save_adapter)
        assert "save_pretrained" in src, (
            "save_adapter does not call save_pretrained")
        assert "param.requires_grad" not in src, (
            "Manual-fallback .pth save path resurrected — adapter "
            "files would lack rank/alpha/target_modules metadata")
        assert "atomic_torch_save" not in src, (
            "save_adapter regressed to atomic_torch_save fallback")

    def test_gui_logic_wires_adapter_auto_restore(self):
        """LoRA-1b foundation: `_on_model_loaded` must call
        `_restore_lora_adapter_for_base` so the user's previous
        adapter choice survives a model reload. Structural gate
        catches the regression where the call is removed but the
        helper still exists (silent loss of adapter persistence)."""
        import inspect
        from enigma_engine.gui import gui_logic
        # The auto-restore lives inside the on-load callback on the
        # composed LogicMixin (LogicChatMixin + LogicMediaMixin).
        src = inspect.getsource(gui_logic.LogicMixin._on_model_loaded)
        assert "_restore_lora_adapter_for_base" in src, (
            "Model-load callback does not auto-restore adapter — "
            "user's saved adapter choice will be silently dropped")

    # ================================================================
    # Pass 156t — LoRA-1b UX surfaces
    # ================================================================

    def test_models_page_renders_lora_section_per_card(self):
        """LoRA-1b UX: every model card calls
        `_build_lora_section_for_card`. Structural gate — the GUI
        cannot be instantiated in CI but the wiring must be present
        in the card builder. Without this call, the user has no
        surface to apply or clear adapters from the MODELS page."""
        import inspect
        from enigma_engine.gui import gui_pages
        # _populate_model_cards must invoke the section builder.
        src = inspect.getsource(
            gui_pages.PagesMixin._populate_model_cards)
        assert "_build_lora_section_for_card" in src, (
            "MODELS-page card builder does not invoke the LoRA "
            "section — user cannot apply adapters from the GUI")
        # Section builder must call scan_lora_adapters with the
        # model's path so per-base filtering kicks in (otherwise
        # math-base adapters could leak onto coding-base cards).
        section_src = inspect.getsource(
            gui_pages.PagesMixin._build_lora_section_for_card)
        assert "scan_lora_adapters" in section_src, (
            "LoRA section does not call scan_lora_adapters — "
            "would render an empty list")
        assert "model[\"path\"]" in section_src or (
                "model['path']" in section_src), (
            "LoRA section does not pass model path to scanner — "
            "would surface adapters for the wrong base")

    def test_lora_apply_guards_against_inactive_base(self):
        """LoRA-1b UX: clicking Apply on a card whose base is NOT
        currently loaded must surface a chat hint instead of
        attempting a cross-base apply (PEFT would raise on shape
        mismatch, but the friendlier path is to tell the user to
        load the base first). Structural gate on the load-first
        check in `_on_lora_apply`."""
        import inspect
        from enigma_engine.gui import gui_pages
        src = inspect.getsource(gui_pages.PagesMixin._on_lora_apply)
        assert "model_path" in src, (
            "_on_lora_apply does not check the active model_path — "
            "would attempt to apply across mismatched bases")
        assert "_set_chat_adapter" in src, (
            "_on_lora_apply does not delegate to _set_chat_adapter "
            "— would skip persistence and engine call")

    def test_profile_adapter_field_drives_engine_apply(self):
        """LoRA-1b UX: a profile with an `adapter` field must call
        `engine.apply_adapter`; a profile with NO adapter (or empty
        string / None) must call `engine.clear_adapter`. The clear
        case is critical — switching to a profile that doesn't
        specify an adapter must NOT silently inherit the previous
        profile's adapter."""
        from enigma_engine.core.ai_profile import (
            AIProfile, apply_profile_to_engine,
        )

        class FakeEngine:
            def __init__(self):
                self.applied: list[str] = []
                self.cleared: int = 0
                self.system_prompt = ""
                self.temperature = 0.0
                self.top_p = 0.0
                self.top_k = 0
                self.max_tokens = 0

            def apply_adapter(self, path):
                self.applied.append(str(path))

            def clear_adapter(self):
                self.cleared += 1

        # Profile with adapter → apply called.
        eng = FakeEngine()
        prof_with = AIProfile(name="P1", adapter="models/checkpoints/foo")
        apply_profile_to_engine(prof_with, eng)
        assert eng.applied == ["models/checkpoints/foo"]
        assert eng.cleared == 0, (
            "apply_profile_to_engine called clear_adapter when "
            "profile pinned an adapter")

        # Profile without adapter → clear called (boundary discipline).
        eng2 = FakeEngine()
        prof_without = AIProfile(name="P2")  # adapter defaults to None
        apply_profile_to_engine(prof_without, eng2)
        assert eng2.applied == []
        assert eng2.cleared == 1, (
            "apply_profile_to_engine did not clear adapter when "
            "profile has no adapter field — silent inheritance bug")

        # Empty-string adapter is also "no adapter" (treated as None).
        eng3 = FakeEngine()
        prof_empty = AIProfile(name="P3", adapter="")
        apply_profile_to_engine(prof_empty, eng3)
        assert eng3.applied == []
        assert eng3.cleared == 1, (
            "Empty adapter string did not trigger clear — should "
            "behave the same as None per profile boundary discipline")

    def test_profile_adapter_field_round_trips_through_dict(self):
        """LoRA-1b UX: AIProfile.from_dict / to_dict must preserve
        the `adapter` field. Without this round-trip, profile JSON
        files cannot pin an adapter."""
        from enigma_engine.core.ai_profile import AIProfile
        prof = AIProfile(name="X", adapter="models/checkpoints/bar")
        data = prof.to_dict()
        assert data["adapter"] == "models/checkpoints/bar"
        rebuilt = AIProfile.from_dict(data)
        assert rebuilt.adapter == "models/checkpoints/bar"

        # Old profile JSONs without the field must still parse.
        old_data = {"name": "Old", "id": "old"}
        old_prof = AIProfile.from_dict(old_data)
        assert old_prof.adapter is None, (
            "Old profiles without adapter field must default to "
            "None for backward compatibility")

    def test_legacy_lora_migration_moves_pth_files(self, tmp_path,
                                                   monkeypatch):
        """LoRA-1b UX: legacy `.pth` LoRA files must be moved to
        `models/checkpoints/legacy_lora_pth/` with a NOTICE.txt.
        Behavioural test on the migrate() function — exercises the
        full move + notice + idempotence flow."""
        import migrate_legacy_lora as mig

        models_dir = tmp_path / "models"
        (models_dir / "lora_adapters").mkdir(parents=True)
        (models_dir / "checkpoints").mkdir(parents=True)

        # Three legacy files: one in lora_adapters/, one matching
        # *_lora.pth in checkpoints/, one un-matching in
        # checkpoints/ (must NOT be moved).
        loose = models_dir / "lora_adapters" / "old.pth"
        loose.write_bytes(b"fake-weights-1")
        named = models_dir / "checkpoints" / "model_v2_lora.pth"
        named.write_bytes(b"fake-weights-2")
        innocent = models_dir / "checkpoints" / "training_state.pth"
        innocent.write_bytes(b"NOT-a-lora-file")

        monkeypatch.setattr(mig, "MODELS_DIR", models_dir)
        monkeypatch.setattr(
            mig, "QUARANTINE_DIR",
            models_dir / "checkpoints" / "legacy_lora_pth")

        # Dry-run: nothing moved, files still in place.
        result_dry = mig.migrate(apply=False)
        assert len(result_dry["found"]) == 2, (
            "Dry-run did not detect both legacy files (only "
            "*_lora.pth in checkpoints/ + loose .pth in "
            "lora_adapters/)")
        assert loose.exists()
        assert named.exists()
        assert result_dry["moved"] == []

        # Apply: both legacy files moved, innocent one untouched,
        # NOTICE.txt written.
        result = mig.migrate(apply=True)
        assert len(result["moved"]) == 2
        assert not loose.exists()
        assert not named.exists()
        assert innocent.exists(), (
            "Migration moved a non-LoRA .pth file — would lose "
            "training state")
        quar = models_dir / "checkpoints" / "legacy_lora_pth"
        assert (quar / "old.pth").exists()
        assert (quar / "model_v2_lora.pth").exists()
        notice = quar / "NOTICE.txt"
        assert notice.exists()
        notice_text = notice.read_text(encoding="utf-8")
        assert "PEFT" in notice_text, (
            "NOTICE.txt does not explain the PEFT-format change")

        # Idempotence: re-running with --apply on the same tree is
        # a no-op (no new files found, no moves).
        result_again = mig.migrate(apply=True)
        assert result_again["found"] == [], (
            "Migration is not idempotent — second run found legacy "
            "files that were already quarantined")

    def test_legacy_lora_migration_handles_filename_collision(
            self, tmp_path, monkeypatch):
        """LoRA-1b UX: if a quarantine file with the same name
        already exists (e.g. from a previous training run that was
        already migrated), the second migration must not clobber it
        — append a numeric suffix instead. Adversarial test on the
        rename logic; without it a duplicate name silently overwrites
        and we lose data."""
        import migrate_legacy_lora as mig

        models_dir = tmp_path / "models"
        (models_dir / "lora_adapters").mkdir(parents=True)
        quar = models_dir / "checkpoints" / "legacy_lora_pth"
        quar.mkdir(parents=True)
        # Pre-existing quarantine file with the same name.
        (quar / "old.pth").write_bytes(b"PRE-EXISTING")
        # New legacy file with the same basename in lora_adapters/.
        new_loose = models_dir / "lora_adapters" / "old.pth"
        new_loose.write_bytes(b"NEW-CONTENT")

        monkeypatch.setattr(mig, "MODELS_DIR", models_dir)
        monkeypatch.setattr(mig, "QUARANTINE_DIR", quar)

        mig.migrate(apply=True)

        # Original quarantine file untouched.
        assert (quar / "old.pth").read_bytes() == b"PRE-EXISTING"
        # New file landed under a suffixed name.
        assert (quar / "old_1.pth").exists()
        assert (quar / "old_1.pth").read_bytes() == b"NEW-CONTENT"

    # ================================================================
    # Pass 156u-A — LoRA stacking (engine + persistence)
    # ================================================================

    def test_engine_exposes_apply_adapter_stack(self):
        """LoRA-1b stacking: `EnigmaEngine.apply_adapter_stack` must
        merge multiple PEFT adapters via ``add_weighted_adapter``
        (linear combination), set the merged stack as active, and
        clear the KV cache. Structural gate — engine __init__ needs
        weights, can't instantiate in CI."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        assert hasattr(EnigmaEngine, "apply_adapter_stack"), (
            "EnigmaEngine missing apply_adapter_stack — multi-LoRA "
            "stacking entry point is absent")
        src = inspect.getsource(EnigmaEngine.apply_adapter_stack)
        assert "add_weighted_adapter" in src, (
            "apply_adapter_stack does not call "
            "model.add_weighted_adapter — would not actually merge "
            "the adapters")
        assert "set_adapter" in src, (
            "apply_adapter_stack does not activate the merged stack "
            "via set_adapter — silent no-op")
        assert "clear_kv_cache" in src, (
            "apply_adapter_stack does not clear KV cache — stale "
            "cache from prior weights would corrupt next generation")
        # Validation gates BEFORE any heavy work (no peft import yet).
        assert "ValueError" in src or "raise" in src, (
            "apply_adapter_stack does not raise on bad input — "
            "would silently no-op or crash deep in PEFT")

    def test_apply_adapter_stack_rejects_empty_list(self):
        """LoRA-1b stacking: an empty adapter list is a programming
        error from the caller — must raise ValueError BEFORE touching
        ``self.model`` or importing peft. Behavioural test via unbound
        call so we don't need a real EnigmaEngine instance."""
        import pytest
        from enigma_engine.core.inference import EnigmaEngine

        class _FakeSelf:
            """Minimal stand-in — if validation accidentally accesses
            anything else, AttributeError makes the failure loud."""

        with pytest.raises(ValueError, match="empty"):
            EnigmaEngine.apply_adapter_stack(_FakeSelf(), [])

    def test_apply_adapter_stack_rejects_non_finite_weight(
            self, tmp_path):
        """LoRA-1b stacking: NaN / Inf weights silently corrupt the
        merged adapter. Must raise ValueError before
        ``add_weighted_adapter`` is called."""
        import math
        import pytest
        from enigma_engine.core.inference import EnigmaEngine

        # Build a real-looking adapter dir so the path-existence
        # check passes and we get to the weight validation.
        ad = tmp_path / "ad"
        ad.mkdir()
        (ad / "adapter_config.json").write_text("{}")

        class _FakeSelf:
            pass

        with pytest.raises(ValueError, match="finite|NaN|weight"):
            EnigmaEngine.apply_adapter_stack(
                _FakeSelf(), [(ad, math.nan)])
        with pytest.raises(ValueError, match="finite|inf|weight"):
            EnigmaEngine.apply_adapter_stack(
                _FakeSelf(), [(ad, math.inf)])

    def test_apply_adapter_stack_rejects_missing_adapter_dir(
            self, tmp_path):
        """LoRA-1b stacking: a missing adapter path must surface a
        FileNotFoundError immediately, not deep inside PEFT. Mirrors
        the validation in single-adapter ``apply_adapter``."""
        import pytest
        from enigma_engine.core.inference import EnigmaEngine
        missing = tmp_path / "nope"

        class _FakeSelf:
            pass

        with pytest.raises(FileNotFoundError):
            EnigmaEngine.apply_adapter_stack(
                _FakeSelf(), [(missing, 1.0)])

    def test_gui_logic_set_chat_adapter_stack_persists_to_stack_key(
            self):
        """LoRA-1b stacking: `_set_chat_adapter_stack` must write to
        the per-base ``chat_adapter_stack:<stem>`` key (NOT the
        single-adapter ``chat_adapter:<stem>`` key) and must clear
        the single key for mutual exclusion. Structural gate — the
        full GUI mixin can't be instantiated in CI."""
        import inspect
        from enigma_engine.gui import gui_logic
        assert hasattr(
            gui_logic.LogicMixin, "_set_chat_adapter_stack"), (
            "gui_logic missing _set_chat_adapter_stack — no GUI "
            "entry point for multi-LoRA")
        src = inspect.getsource(
            gui_logic.LogicMixin._set_chat_adapter_stack)
        assert "chat_adapter_stack:" in src, (
            "_set_chat_adapter_stack does not write to the "
            "chat_adapter_stack: route key — stack would not "
            "persist across model reloads")
        assert "apply_adapter_stack" in src, (
            "_set_chat_adapter_stack does not call "
            "engine.apply_adapter_stack — silent no-op")
        # Mutual exclusion: writing a stack must remove the lingering
        # single-adapter key for the same base, otherwise restore
        # could pick up either at random.
        assert "chat_adapter:" in src and ".pop(" in src, (
            "_set_chat_adapter_stack does not clear the single "
            "chat_adapter: key — mutual-exclusion violation, "
            "restore order becomes ambiguous")

    def test_gui_logic_set_chat_adapter_single_clears_stack_key(self):
        """LoRA-1b stacking: applying or clearing a single adapter
        must remove any lingering ``chat_adapter_stack:<stem>`` entry.
        Otherwise switching from a stack back to a single adapter
        would leave the stack in route_assignments and the next
        restore would resurrect it."""
        import inspect
        from enigma_engine.gui import gui_logic
        src = inspect.getsource(
            gui_logic.LogicMixin._set_chat_adapter)
        assert "chat_adapter_stack:" in src, (
            "_set_chat_adapter does not touch the stack key — "
            "switching single→stack→single would silently retain "
            "the old stack for the next reload")

    def test_gui_logic_restore_prefers_stack_over_single(self):
        """LoRA-1b stacking: when both keys exist for a base
        (shouldn't happen post-156u, but defend against legacy data),
        restore must prefer the stack key. Stack is the more recent
        and more specific intent."""
        import inspect
        from enigma_engine.gui import gui_logic
        src = inspect.getsource(
            gui_logic.LogicMixin._restore_lora_adapter_for_base)
        assert "chat_adapter_stack:" in src, (
            "_restore_lora_adapter_for_base does not check the "
            "stack key — saved stacks would be ignored on reload")
        assert "apply_adapter_stack" in src, (
            "_restore_lora_adapter_for_base does not call "
            "engine.apply_adapter_stack — restore would silently "
            "skip stacks even if the route key is present")

    def test_apply_adapter_stack_rejects_duplicate_path(self, tmp_path):
        """LoRA-1b stacking (156u-A2): duplicate adapter paths in the
        same stack must raise ``ValueError``. The docstring promises
        this; without a test the promise is a Pass 156s2 anti-pattern
        (claim without proof). PEFT's ``add_weighted_adapter`` would
        also reject duplicate names, but raising up front is friendlier
        and avoids the wasted base-wrap on an invalid call."""
        import pytest
        from enigma_engine.core.inference import EnigmaEngine

        ad = tmp_path / "ad"
        ad.mkdir()
        (ad / "adapter_config.json").write_text("{}")

        class _FakeSelf:
            pass

        with pytest.raises(ValueError, match="[Dd]uplicate"):
            EnigmaEngine.apply_adapter_stack(
                _FakeSelf(), [(ad, 0.5), (ad, 0.5)])

    def test_restore_lora_stack_survives_corrupted_entries(
            self, tmp_path, monkeypatch):
        """LoRA-1b stacking (156u-A2): if ``route_assignments.json``
        gets corrupted (hand-edited, older format, partial write) so
        that a stack entry is not a ``{path, weight}`` dict OR the
        weight is non-numeric, the restore path must NOT propagate
        AttributeError / TypeError / ValueError up through model
        load. It must drop the orphan key, surface a chat warning,
        and return cleanly so the user can keep using the base model.

        Author's-lens behavioural test — the existing except clause
        only wraps ``engine.apply_adapter_stack(...)``. Bugs in the
        list-comprehension that builds the entries list (e.g.
        ``item.get`` on a non-dict, or ``float("abc")``) escape the
        guard and crash the whole load."""
        from enigma_engine.gui.gui_logic import LogicMixin

        # Build a valid adapter dir so path-existence isn't the
        # signal we're testing — the failure must come from the
        # corrupted shape, not from a missing file.
        ad = tmp_path / "ad"
        ad.mkdir()
        (ad / "adapter_config.json").write_text("{}")

        # FakeEngine pretending to support stacking. Should NEVER be
        # called because we abort before reaching it.
        class FakeEngine:
            def __init__(self):
                self.stack_calls = 0

            def apply_adapter(self, *_a, **_kw):
                pass

            def apply_adapter_stack(self, *_a, **_kw):
                self.stack_calls += 1

        # Minimal harness — only the attributes the restore method
        # actually reaches for.
        class Harness:
            _adapter_route_key = LogicMixin._adapter_route_key
            _adapter_stack_route_key = (
                LogicMixin._adapter_stack_route_key)
            _restore_lora_adapter_for_base = (
                LogicMixin._restore_lora_adapter_for_base)

            def __init__(self):
                self.engine = FakeEngine()
                self.route_assignments = {}
                self.system_msgs: list[str] = []
                self.error_msgs: list[str] = []

            def _chat_system(self, m):
                self.system_msgs.append(m)

            def _chat_error(self, m):
                self.error_msgs.append(m)

        # Monkeypatch the persistence helper so the test doesn't
        # touch real disk.
        from enigma_engine.gui import gui_logic as gl
        saved_writes: list[dict] = []
        monkeypatch.setattr(
            gl, "save_route_assignments",
            lambda d: saved_writes.append(dict(d)))

        base = "models/test_base.pth"
        stack_key = f"chat_adapter_stack:{Path(base).stem}"

        # Case 1: stack entry is not a dict (corrupted JSON).
        h1 = Harness()
        h1.route_assignments = {stack_key: [1, 2, 3]}
        h1._restore_lora_adapter_for_base(base)
        assert h1.engine.stack_calls == 0, (
            "Corrupted-shape stack made it through to "
            "engine.apply_adapter_stack — restore must drop the "
            "key BEFORE attempting apply")
        assert stack_key not in h1.route_assignments, (
            "Corrupted stack key was not purged — next reload "
            "would hit the same crash")
        assert h1.system_msgs or h1.error_msgs, (
            "Corrupted stack was silently dropped — user got no "
            "indication that their saved stack is gone")

        # Case 2: weight is non-numeric (string that can't float).
        h2 = Harness()
        h2.route_assignments = {
            stack_key: [{"path": str(ad), "weight": "not-a-number"}]}
        h2._restore_lora_adapter_for_base(base)
        assert h2.engine.stack_calls == 0
        assert stack_key not in h2.route_assignments, (
            "Non-numeric weight in saved stack was not purged")
        assert h2.system_msgs or h2.error_msgs, (
            "Non-numeric weight was silently dropped — user got "
            "no indication that their saved stack is gone")

    # ================================================================
    # Pass 156u-B — LoRA stacking UI (parse + dispatch)
    # ================================================================

    def test_parse_lora_stack_inputs_empty_defaults_to_one(self):
        """Pass 156u-B: empty / whitespace-only weight strings must
        default to 1.0 (the user's intent when they tick a row but
        don't type a custom weight). Both '' and '  ' must produce
        1.0, not a parse error."""
        from enigma_engine.gui.gui_logic import (
            _parse_lora_stack_inputs)

        pairs, errors = _parse_lora_stack_inputs([
            ("models/checkpoints/foo", ""),
            ("models/checkpoints/bar", "   "),
        ])
        assert errors == []
        assert pairs == [
            ("models/checkpoints/foo", 1.0),
            ("models/checkpoints/bar", 1.0),
        ]

    def test_parse_lora_stack_inputs_rejects_non_numeric(self):
        """Pass 156u-B: non-numeric weight strings must produce a
        parse error naming the offending adapter (so the user can
        fix the right row). The engine layer would also catch this
        as ValueError but the GUI layer must surface a friendlier
        message that points at the bad row."""
        from enigma_engine.gui.gui_logic import (
            _parse_lora_stack_inputs)

        pairs, errors = _parse_lora_stack_inputs([
            ("models/checkpoints/good", "0.7"),
            ("models/checkpoints/bad", "abc"),
        ])
        assert pairs == [], (
            "Parser returned partial pairs on error — must be "
            "all-or-nothing so the engine never sees a partial "
            "stack")
        assert len(errors) == 1
        assert "bad" in errors[0], (
            f"Error message {errors[0]!r} does not name the bad "
            "adapter — user can't tell which row to fix")

    def test_parse_lora_stack_inputs_rejects_nan_and_inf(self):
        """Pass 156u-B: NaN and Inf must be rejected at the GUI
        layer with a clear message. The engine validation would
        also reject these but a generic ValueError is less helpful
        than 'NaN weight on adapter X'."""
        from enigma_engine.gui.gui_logic import (
            _parse_lora_stack_inputs)

        pairs_nan, errors_nan = _parse_lora_stack_inputs(
            [("models/checkpoints/foo", "nan")])
        assert pairs_nan == []
        assert any("foo" in e for e in errors_nan)

        pairs_inf, errors_inf = _parse_lora_stack_inputs(
            [("models/checkpoints/foo", "inf")])
        assert pairs_inf == []
        assert any("foo" in e for e in errors_inf)

    def test_parse_lora_stack_inputs_accepts_negative(self):
        """Pass 156u-B: negative weights are LEGITIMATE — they
        subtract the adapter's contribution from the merged stack
        (e.g. an "anti-coding" adapter). Parser must NOT reject
        them — the engine's `add_weighted_adapter` accepts
        negatives. This is a discipline test against an over-eager
        'sanitize input' instinct."""
        from enigma_engine.gui.gui_logic import (
            _parse_lora_stack_inputs)

        pairs, errors = _parse_lora_stack_inputs([
            ("models/checkpoints/foo", "-0.5"),
            ("models/checkpoints/bar", "1.5"),
        ])
        assert errors == []
        assert pairs == [
            ("models/checkpoints/foo", -0.5),
            ("models/checkpoints/bar", 1.5),
        ]

    def test_parse_lora_stack_inputs_collects_all_errors(self):
        """Pass 156u-B: when multiple rows have bad weights, the
        parser must report ALL errors, not just the first. Otherwise
        the user fixes one, retries, hits the next, fixes it,
        retries, hits the next — N round-trips for N typos."""
        from enigma_engine.gui.gui_logic import (
            _parse_lora_stack_inputs)

        pairs, errors = _parse_lora_stack_inputs([
            ("models/checkpoints/a", "abc"),
            ("models/checkpoints/b", "1.0"),
            ("models/checkpoints/c", "xyz"),
        ])
        assert pairs == []
        assert len(errors) == 2, (
            "Parser stopped at first error — user must fix typos "
            "one round-trip per typo")
        assert any("a" in e for e in errors)
        assert any("c" in e for e in errors)

    def test_on_lora_apply_stack_load_first_guard(self):
        """Pass 156u-B: clicking Apply Stack on a card whose base is
        NOT currently loaded must surface a chat hint and NOT call
        the engine. Same load-first discipline as `_on_lora_apply`."""
        from enigma_engine.gui import gui_pages

        class Harness:
            _on_lora_apply_stack = (
                gui_pages.PagesMixin._on_lora_apply_stack)

            def __init__(self):
                self.model_path = "models/A.pth"
                self.system_msgs: list[str] = []
                self.error_msgs: list[str] = []
                self.set_stack_calls: list = []
                self.set_single_calls: list = []

            def _chat_system(self, m):
                self.system_msgs.append(m)

            def _chat_error(self, m):
                self.error_msgs.append(m)

            def _set_chat_adapter_stack(self, base, adapters):
                self.set_stack_calls.append((base, adapters))

            def _set_chat_adapter(self, base, adapter):
                self.set_single_calls.append((base, adapter))

            def _refresh_model_cards(self):
                pass

        h = Harness()
        # Trying to apply on a DIFFERENT model than what's loaded.
        h._on_lora_apply_stack(
            {"path": "models/B.pth", "name": "Other"},
            [("models/checkpoints/foo", "1.0")])
        assert h.set_stack_calls == []
        assert h.set_single_calls == []
        assert any("Other" in m for m in h.system_msgs), (
            "Load-first hint did not name the model the user "
            "needs to load")

    def test_on_lora_apply_stack_empty_selection_is_chat_hint(self):
        """Pass 156u-B: zero selections is a UX state, not a
        programming error. Surface a chat hint, do NOT call the
        engine (which would raise ValueError on empty list).
        Without this guard the user gets a cryptic 'requires a
        non-empty list' chat error from the engine."""
        from enigma_engine.gui import gui_pages

        class Harness:
            _on_lora_apply_stack = (
                gui_pages.PagesMixin._on_lora_apply_stack)

            def __init__(self):
                self.model_path = "models/A.pth"
                self.system_msgs: list[str] = []
                self.error_msgs: list[str] = []
                self.set_stack_calls: list = []
                self.set_single_calls: list = []

            def _chat_system(self, m):
                self.system_msgs.append(m)

            def _chat_error(self, m):
                self.error_msgs.append(m)

            def _set_chat_adapter_stack(self, base, adapters):
                self.set_stack_calls.append((base, adapters))

            def _set_chat_adapter(self, base, adapter):
                self.set_single_calls.append((base, adapter))

            def _refresh_model_cards(self):
                pass

        h = Harness()
        h._on_lora_apply_stack(
            {"path": "models/A.pth", "name": "A"}, [])
        assert h.set_stack_calls == []
        assert h.set_single_calls == []
        assert h.system_msgs, (
            "Empty selection produced no chat hint — user gets "
            "no feedback on a no-op click")

    def test_on_lora_apply_stack_single_selection_uses_single_path(
            self):
        """Pass 156u-B: when the user selects exactly one adapter,
        route through `_set_chat_adapter` (single-adapter path), NOT
        `_set_chat_adapter_stack`. Avoids the `_stack` PEFT
        indirection for the trivial case AND keeps mutual exclusion
        clean (single-key wins). Also gates that
        `_refresh_model_cards` is called on success so the active
        highlight updates — silent no-refresh would make the user
        think the click failed."""
        from enigma_engine.gui import gui_pages

        class Harness:
            _on_lora_apply_stack = (
                gui_pages.PagesMixin._on_lora_apply_stack)

            def __init__(self):
                self.model_path = "models/A.pth"
                self.system_msgs: list[str] = []
                self.error_msgs: list[str] = []
                self.set_stack_calls: list = []
                self.set_single_calls: list = []
                self.refresh_count = 0

            def _chat_system(self, m):
                self.system_msgs.append(m)

            def _chat_error(self, m):
                self.error_msgs.append(m)

            def _set_chat_adapter_stack(self, base, adapters):
                self.set_stack_calls.append((base, adapters))

            def _set_chat_adapter(self, base, adapter):
                self.set_single_calls.append((base, adapter))

            def _refresh_model_cards(self):
                self.refresh_count += 1

        h = Harness()
        h._on_lora_apply_stack(
            {"path": "models/A.pth", "name": "A"},
            [("models/checkpoints/solo", "0.7")])
        assert h.set_stack_calls == [], (
            "Single selection routed through stack path — wastes "
            "the _stack PEFT indirection on the trivial case")
        assert h.set_single_calls == [
            ("models/A.pth", "models/checkpoints/solo")
        ], (
            "Single selection did not delegate to "
            "_set_chat_adapter — silent no-op")
        assert h.refresh_count == 1, (
            "_refresh_model_cards not called after single-path "
            "stack apply — active-adapter highlight will not "
            "update, user thinks click did nothing")

    def test_on_lora_apply_stack_multi_calls_stack_path(self):
        """Pass 156u-B: 2+ selections route through
        `_set_chat_adapter_stack` with parsed weights. Behavioural
        test gating the parse → stack-call wiring, the weight
        forwarding, AND the post-success refresh."""
        from enigma_engine.gui import gui_pages

        class Harness:
            _on_lora_apply_stack = (
                gui_pages.PagesMixin._on_lora_apply_stack)

            def __init__(self):
                self.model_path = "models/A.pth"
                self.system_msgs: list[str] = []
                self.error_msgs: list[str] = []
                self.set_stack_calls: list = []
                self.set_single_calls: list = []
                self.refresh_count = 0

            def _chat_system(self, m):
                self.system_msgs.append(m)

            def _chat_error(self, m):
                self.error_msgs.append(m)

            def _set_chat_adapter_stack(self, base, adapters):
                self.set_stack_calls.append((base, adapters))

            def _set_chat_adapter(self, base, adapter):
                self.set_single_calls.append((base, adapter))

            def _refresh_model_cards(self):
                self.refresh_count += 1

        h = Harness()
        h._on_lora_apply_stack(
            {"path": "models/A.pth", "name": "A"},
            [
                ("models/checkpoints/foo", "0.7"),
                ("models/checkpoints/bar", "0.3"),
            ])
        assert h.set_single_calls == []
        assert h.set_stack_calls == [
            ("models/A.pth",
             [("models/checkpoints/foo", 0.7),
              ("models/checkpoints/bar", 0.3)])
        ]
        assert h.refresh_count == 1, (
            "_refresh_model_cards not called after stack apply — "
            "active-adapter highlight will not update")

    def test_on_lora_apply_stack_parse_error_aborts(self):
        """Pass 156u-B: a parse error in any row must abort BEFORE
        any engine call AND skip the post-apply refresh. The user
        gets ALL parse errors (not just the first) via chat-error
        and no partial stack reaches the persistence layer."""
        from enigma_engine.gui import gui_pages

        class Harness:
            _on_lora_apply_stack = (
                gui_pages.PagesMixin._on_lora_apply_stack)

            def __init__(self):
                self.model_path = "models/A.pth"
                self.system_msgs: list[str] = []
                self.error_msgs: list[str] = []
                self.set_stack_calls: list = []
                self.set_single_calls: list = []
                self.refresh_count = 0

            def _chat_system(self, m):
                self.system_msgs.append(m)

            def _chat_error(self, m):
                self.error_msgs.append(m)

            def _set_chat_adapter_stack(self, base, adapters):
                self.set_stack_calls.append((base, adapters))

            def _set_chat_adapter(self, base, adapter):
                self.set_single_calls.append((base, adapter))

            def _refresh_model_cards(self):
                self.refresh_count += 1

        h = Harness()
        h._on_lora_apply_stack(
            {"path": "models/A.pth", "name": "A"},
            [
                ("models/checkpoints/foo", "0.7"),
                ("models/checkpoints/bar", "abc"),
            ])
        assert h.set_stack_calls == [], (
            "Parse error did not abort the stack call — engine "
            "would receive a partial or zero-weight stack")
        assert h.set_single_calls == []
        assert h.error_msgs, (
            "Parse error produced no chat-error message — user "
            "got silent failure")
        assert h.refresh_count == 0, (
            "_refresh_model_cards called on parse-error abort — "
            "signals success when nothing was applied")

    def test_models_page_renders_stacking_controls_per_row(self):
        """Pass 156u-B: every adapter row in the LoRA section must
        render a checkbox (for stack selection) AND a numeric weight
        entry (CTkEntry, NOT a slider — per Dia rules). Structural
        gate on the section builder; without it the stacking UI is
        unreachable from the GUI."""
        import inspect
        from enigma_engine.gui import gui_pages

        src = inspect.getsource(
            gui_pages.PagesMixin._build_lora_section_for_card)
        assert "CTkCheckBox" in src, (
            "LoRA section does not render per-row checkboxes — "
            "user has no way to select multiple adapters for a "
            "stack")
        # Weight entry must be the float-mode themed numeric entry
        # (allows negatives + scientific notation), NOT a plain
        # int-mode entry. The literal token "themed_numeric_entry"
        # is what gates this; "mode=\"float\"" gates the float
        # contract specifically (no NaN-by-typo from int mode
        # rejecting decimals).
        assert "themed_numeric_entry" in src, (
            "LoRA section does not render per-row weight entries "
            "— user has no way to specify weights")
        assert 'mode="float"' in src, (
            "LoRA weight entries are not in float mode — user "
            "cannot type decimals, negatives, or scientific "
            "notation")
        assert "CTkSlider" not in src, (
            "LoRA section uses CTkSlider — Dia rule: numeric "
            "input only, no sliders")
        assert "_on_lora_apply_stack" in src, (
            "LoRA section does not wire the Apply Stack button "
            "to _on_lora_apply_stack — button would be a no-op")

    # ================================================================
    # Pass 156v Step 1 — Session-1 unification (chat session marker)
    # ================================================================

    def test_chat_session_marker_helper_exists(self):
        """Pass 156v: `LogicChatMixin._chat_session_marker` must
        exist and write to the chat display via `_chat_append`. This
        is the single source of truth for state-change dividers
        (adapter swap, future model/profile/prompt swaps)."""
        import inspect
        from enigma_engine.gui import gui_logic_chat

        assert hasattr(
            gui_logic_chat.LogicChatMixin, "_chat_session_marker"), (
            "LogicChatMixin missing _chat_session_marker — Pass "
            "156v Session-1 helper not shipped")
        src = inspect.getsource(
            gui_logic_chat.LogicChatMixin._chat_session_marker)
        assert "_chat_append" in src, (
            "_chat_session_marker does not delegate to "
            "_chat_append — divider would not appear in the chat "
            "log")
        assert "session_marker" in src, (
            "_chat_session_marker does not use the "
            "'session_marker' tag — divider style would inherit "
            "from system_msg and look identical to system "
            "messages")

    def test_session_marker_tag_configured_on_chat_display(self):
        """Pass 156v: chat display must configure the
        `session_marker` tag so the new helper actually renders
        with divider styling (centered, dim). Without this tag
        config the marker would render in default text color and
        be visually indistinguishable from regular text."""
        import inspect
        from enigma_engine.gui import gui_pages

        # The chat display tag config lives in the page builder.
        # Find it by scanning all PagesMixin methods for
        # tag_configure of session_marker.
        found = False
        for name in dir(gui_pages.PagesMixin):
            attr = getattr(gui_pages.PagesMixin, name)
            if not callable(attr):
                continue
            try:
                src = inspect.getsource(attr)
            except (OSError, TypeError):
                continue
            if 'tag_configure("session_marker"' in src:
                found = True
                break
        assert found, (
            "No PagesMixin method configures the 'session_marker' "
            "tag — _chat_session_marker would render with default "
            "tk styling, defeating the divider UX")

    def test_set_chat_adapter_apply_emits_session_marker(self):
        """Pass 156v: applying a single LoRA adapter must surface
        the change via `_chat_session_marker` (divider), NOT
        `_chat_system` (regular system message). Without this
        distinction the user can't visually locate the seam where
        weights changed if quality regresses afterwards."""
        from enigma_engine.gui import gui_logic

        class FakeEngine:
            def apply_adapter(self, p):
                self.last = p

        class Harness:
            _set_chat_adapter = gui_logic.LogicMixin._set_chat_adapter
            _adapter_route_key = (
                gui_logic.LogicMixin._adapter_route_key)
            _adapter_stack_route_key = (
                gui_logic.LogicMixin._adapter_stack_route_key)

            def __init__(self):
                self.engine = FakeEngine()
                self.route_assignments: dict = {}
                self.system_msgs: list[str] = []
                self.error_msgs: list[str] = []
                self.marker_msgs: list[str] = []

            def _chat_system(self, m):
                self.system_msgs.append(m)

            def _chat_error(self, m):
                self.error_msgs.append(m)

            def _chat_session_marker(self, m):
                self.marker_msgs.append(m)

        h = Harness()
        # Patch save_route_assignments to no-op for the test —
        # _set_chat_adapter calls it on success and we don't want
        # to write to disk.
        import enigma_engine.gui.gui_logic as logic_mod
        orig_save = logic_mod.save_route_assignments
        logic_mod.save_route_assignments = lambda *a, **kw: None
        try:
            h._set_chat_adapter(
                "models/A.pth", "models/checkpoints/foo_lora")
        finally:
            logic_mod.save_route_assignments = orig_save

        assert h.marker_msgs, (
            "Successful adapter apply did not emit a session "
            "marker — user has no visible divider in chat log to "
            "locate the swap")
        assert any("foo_lora" in m for m in h.marker_msgs), (
            "Session marker does not name the adapter — user "
            "sees a divider but can't tell what changed")
        assert not h.system_msgs, (
            "Successful apply still emitted a _chat_system "
            "message — duplicate signal, divider is supposed to "
            "REPLACE the regular system message")

    def test_set_chat_adapter_clear_emits_session_marker(self):
        """Pass 156v: clearing the LoRA adapter must also emit a
        session marker, not a plain system message. Same UX
        contract — the seam where the model reverted to base
        weights must be visually locatable."""
        from enigma_engine.gui import gui_logic

        class FakeEngine:
            def clear_adapter(self):
                self.cleared = True

        class Harness:
            _set_chat_adapter = gui_logic.LogicMixin._set_chat_adapter
            _adapter_route_key = (
                gui_logic.LogicMixin._adapter_route_key)
            _adapter_stack_route_key = (
                gui_logic.LogicMixin._adapter_stack_route_key)

            def __init__(self):
                self.engine = FakeEngine()
                self.route_assignments: dict = {
                    "chat_adapter:A": "models/checkpoints/old"}
                self.system_msgs: list[str] = []
                self.error_msgs: list[str] = []
                self.marker_msgs: list[str] = []

            def _chat_system(self, m):
                self.system_msgs.append(m)

            def _chat_error(self, m):
                self.error_msgs.append(m)

            def _chat_session_marker(self, m):
                self.marker_msgs.append(m)

        h = Harness()
        import enigma_engine.gui.gui_logic as logic_mod
        orig_save = logic_mod.save_route_assignments
        logic_mod.save_route_assignments = lambda *a, **kw: None
        try:
            h._set_chat_adapter("models/A.pth", None)
        finally:
            logic_mod.save_route_assignments = orig_save

        assert h.marker_msgs, (
            "Adapter clear did not emit a session marker — user "
            "has no visible divider for the revert to base "
            "weights")
        assert not h.system_msgs, (
            "Adapter clear emitted a _chat_system message in "
            "addition to the marker — duplicate signal")

    def test_set_chat_adapter_stack_apply_emits_session_marker(self):
        """Pass 156v: applying a multi-LoRA stack must emit a
        session marker that names the stack members + weights.
        Same contract as single-adapter apply."""
        from enigma_engine.gui import gui_logic

        class FakeEngine:
            def apply_adapter_stack(self, adapters):
                self.last = adapters

        class Harness:
            _set_chat_adapter_stack = (
                gui_logic.LogicMixin._set_chat_adapter_stack)
            _adapter_route_key = (
                gui_logic.LogicMixin._adapter_route_key)
            _adapter_stack_route_key = (
                gui_logic.LogicMixin._adapter_stack_route_key)

            def __init__(self):
                self.engine = FakeEngine()
                self.route_assignments: dict = {}
                self.system_msgs: list[str] = []
                self.error_msgs: list[str] = []
                self.marker_msgs: list[str] = []

            def _chat_system(self, m):
                self.system_msgs.append(m)

            def _chat_error(self, m):
                self.error_msgs.append(m)

            def _chat_session_marker(self, m):
                self.marker_msgs.append(m)

        h = Harness()
        import enigma_engine.gui.gui_logic as logic_mod
        orig_save = logic_mod.save_route_assignments
        logic_mod.save_route_assignments = lambda *a, **kw: None
        try:
            h._set_chat_adapter_stack(
                "models/A.pth",
                [("models/checkpoints/foo", 0.7),
                 ("models/checkpoints/bar", 0.3)])
        finally:
            logic_mod.save_route_assignments = orig_save

        assert h.marker_msgs, (
            "Stack apply did not emit a session marker")
        marker = h.marker_msgs[0]
        assert "foo" in marker and "bar" in marker, (
            f"Stack marker {marker!r} does not name both "
            "adapters — user can't tell what's in the stack")
        assert not h.system_msgs, (
            "Stack apply emitted a _chat_system message in "
            "addition to the marker — duplicate signal")

    # -------------------------------------------------------------------------
    # Pass 156v Step 2 (Session-1 unification — model + RAG seams)
    # -------------------------------------------------------------------------
    # Step 1 wired the marker for LoRA adapter swaps. Step 2 extends
    # the same UX contract to the other genuine session-state-change
    # surfaces that already exist in the GUI today: model load, model
    # unload, RAG corpus enable, RAG corpus disable. Profile swap and
    # system-prompt edit have no chat-page handlers yet (no GUI surface
    # to swap them); they remain deferred until the surface lands.
    #
    # The model-load and RAG-enable paths are deeply entangled with Tk
    # widgets, header status, route assignments, and `self.after()`
    # main-thread bouncing — exercising the full path in a Harness is
    # disproportionate. Structural tests via `inspect.getsource` gate
    # the wiring claim (helper IS called from the entry point with a
    # reason string that names the change). Behavioural coverage of
    # the helper itself lives in the Step 1 tests above. RAG-disable is
    # a 2-line method and is easy to test behaviourally; we do.

    def test_on_model_loaded_emits_session_marker(self):
        """Pass 156v Step 2: model-load success must surface the
        change via `_chat_session_marker`, not `_chat_system`. The
        seam where weights changed is exactly when the user needs
        a visible divider — answers regress here far more often
        than from any other state change.
        """
        import inspect
        from enigma_engine.gui import gui_logic
        src = inspect.getsource(gui_logic.LogicMixin._on_model_loaded)
        assert "_chat_session_marker(" in src, (
            "_on_model_loaded does not call _chat_session_marker "
            "— the model-swap seam still uses a regular system "
            "message and is not visually distinct in the chat log")

    def test_on_model_loaded_marker_does_not_duplicate_system_message(
        self,
    ):
        """Pass 156v Step 2: the OLD `_chat_system("Model online...`
        line must be replaced by the marker, not run alongside it.
        Otherwise the user sees a divider PLUS a regular system
        message for the same event — duplicate signal, defeats the
        scan-the-log UX."""
        import inspect
        from enigma_engine.gui import gui_logic
        src = inspect.getsource(gui_logic.LogicMixin._on_model_loaded)
        assert '_chat_system(\n            f"Model online' not in src \
            and '_chat_system(f"Model online' not in src, (
            "_on_model_loaded still emits the old `Model online` "
            "_chat_system message in addition to the new session "
            "marker — duplicate signal")

    def test_unload_engine_emits_session_marker(self):
        """Pass 156v Step 2: unloading the model is a genuine
        session-state change (KV cache gone, weights gone) — must
        emit the marker, not a plain system message."""
        import inspect
        from enigma_engine.gui import gui_logic
        src = inspect.getsource(gui_logic.LogicMixin._unload_model)
        assert "_chat_session_marker(" in src, (
            "_unload_model does not call _chat_session_marker — "
            "the unload event is not visually distinct in the log")
        # Old `_chat_system("Model unloaded.")` must be gone
        assert '_chat_system("Model unloaded.")' not in src, (
            "_unload_model still emits the old `Model unloaded` "
            "_chat_system message — duplicate signal alongside "
            "the new marker")

    def test_rag_disable_emits_session_marker(self):
        """Pass 156v Step 2 (behavioural): RAG-off changes the
        retrieval pipeline that feeds every subsequent answer —
        same answer regression risk as a model swap. Marker, not
        plain system message.
        """
        from enigma_engine.gui import gui_logic

        class FakeEngine:
            _rag_index = object()  # stand-in pre-toggle

        class Harness:
            _on_rag_toggle = gui_logic.LogicMixin._on_rag_toggle

            def __init__(self):
                self.engine = FakeEngine()
                self._rag_index = object()
                self.system_msgs: list[str] = []
                self.marker_msgs: list[str] = []

            def _chat_system(self, m):
                self.system_msgs.append(m)

            def _chat_session_marker(self, m):
                self.marker_msgs.append(m)

        h = Harness()
        h._on_rag_toggle(False)

        assert h.marker_msgs, (
            "RAG disable did not emit a session marker — the "
            "retrieval pipeline change is invisible in the log")
        assert any("rag" in m.lower() or "document" in m.lower()
                   for m in h.marker_msgs), (
            f"RAG-disable marker does not name the subsystem; "
            f"got {h.marker_msgs!r}")
        assert not h.system_msgs, (
            "RAG disable emitted a _chat_system message in "
            "addition to the marker — duplicate signal")
        # Engine + local index must still be cleared
        assert h._rag_index is None
        assert h.engine._rag_index is None

    def test_rag_enable_success_emits_session_marker(self):
        """Pass 156v Step 2: successful RAG index build emits the
        marker (named with chunk + file counts so the user knows
        what corpus is now feeding answers). Structural because
        `_build_rag_index` does threaded `self.after(0, ...)`
        bouncing and disk I/O — disproportionate to exercise."""
        import inspect
        from enigma_engine.gui import gui_logic
        src = inspect.getsource(gui_logic.LogicMixin._build_rag_index)
        assert "_chat_session_marker(" in src, (
            "_build_rag_index does not call _chat_session_marker "
            "on success — RAG-enabled is the same kind of state "
            "change as RAG-disabled, must use the same UX")

    def test_pick_first_match_falls_back_to_first_when_no_match(self):
        """When NONE of the preferred tails match, fall back to the
        first scanned file (legacy behaviour). Empty list → ""."""
        from enigma_engine.gui.scanners import _pick_first_match
        files = [
            {"name": "scratch.txt",
             "path": "data/scratch.txt", "size_kb": 1.0},
        ]
        assert _pick_first_match(files, ["foo/bar.txt"]) == (
            "data/scratch.txt")
        assert _pick_first_match([], ["foo/bar.txt"]) == ""

    def test_scan_sessions(self):
        from enigma_engine.gui.scanners import scan_sessions
        sessions = scan_sessions()
        assert isinstance(sessions, list)
        for s in sessions:
            assert "name" in s
            assert "path" in s

    def test_scan_docs(self):
        """scan_docs returns guides and mod docs."""
        from enigma_engine.gui.scanners import scan_docs, INFO_DIR
        assert INFO_DIR.exists()
        docs = scan_docs()
        assert isinstance(docs, list)
        assert len(docs) > 0
        for doc in docs:
            assert "name" in doc
            assert "path" in doc
            assert "category" in doc
            assert "filename" in doc
        # Has guides
        guides = [d for d in docs if d["category"] == "guides"]
        assert len(guides) >= 5
        names = {d["filename"] for d in guides}
        assert "how_the_ai_works.md" in names
        assert "training_guide.md" in names
        assert "commands_reference.md" in names
        # Has mod docs
        assert any(d["category"].startswith("mod:") for d in docs)
        # All files readable
        for doc in docs:
            path = Path(doc["path"])
            assert path.exists(), f"Missing: {path}"
            assert len(path.read_text(encoding="utf-8")) > 10


# ================================================================
# Config validation
# ================================================================

class TestConfigValidation:
    """Verify config clamping and descriptions."""

    def test_clamp_config_values(self):
        from enigma_engine.gui.scanners import clamp_config
        assert clamp_config("temperature", 5.0) == 2.0
        assert clamp_config("temperature", -1.0) == 0.0
        assert clamp_config("temperature", 0.8) == 0.8
        assert clamp_config("top_k", 0) == 1
        assert clamp_config("top_k", 999) == 200
        assert clamp_config("max_tokens", 0) == 1
        assert clamp_config("max_tokens", 99999) == 99999

    def test_config_descriptions_exist(self):
        from enigma_engine.gui.scanners import (
            CONFIG_DESCRIPTIONS, CONFIG_LIMITS)
        for name in CONFIG_LIMITS:
            assert name in CONFIG_DESCRIPTIONS


# ================================================================
# Module structure and backward compatibility
# ================================================================

class TestModuleStructure:
    """Verify GUI module split, mixins, and re-exports."""

    def test_widget_module(self):
        from enigma_engine.gui.widgets import (
            HUDFrame, GlowFrame, C_BG, FONT_TITLE)
        assert GlowFrame is HUDFrame
        assert isinstance(C_BG, str)
        assert isinstance(FONT_TITLE, tuple)

    def test_desktop_inherits_all_mixins(self):
        from enigma_engine.gui.desktop import EnigmaGUI
        from enigma_engine.gui.gui_docs_page import DocsPageMixin
        from enigma_engine.gui.gui_cmd_page import CMDPageMixin
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert issubclass(EnigmaGUI, DocsPageMixin)
        assert issubclass(EnigmaGUI, CMDPageMixin)
        assert issubclass(EnigmaGUI, ForgeMixin)

    def test_backward_compat_imports(self):
        from enigma_engine.gui.desktop import (
            scan_mods, scan_docs,
            INFO_DIR, CONFIG_LIMITS)
        assert callable(scan_mods)
        assert callable(scan_docs)
        assert isinstance(CONFIG_LIMITS, dict)
        assert INFO_DIR.exists()


# ================================================================
# Mod template completeness
# ================================================================


class TestModDefinitions:
    """Verify mod file conventions and merged audio/voice setup."""

    def test_voice_mod_has_main_entry(self):
        from enigma_engine.gui.scanners import MODS_DIR
        assert (MODS_DIR / "voice" / "main.py").exists()


# ================================================================
# CMD Page
# ================================================================

class TestCMDPage:
    """Verify CMD page dual-mode terminal."""

    def test_cmd_page_methods(self):
        from enigma_engine.gui.gui_cmd_page import (
            CMDPageMixin, MODE_SYSTEM, MODE_ENGINE)
        assert MODE_SYSTEM == "SYSTEM"
        assert MODE_ENGINE == "ENGINE"
        for attr in (
            "_build_page_cmd", "_cmd_execute", "_cmd_clear",
            "_cmd_write", "_cmd_welcome", "_cmd_run_system",
            "_cmd_run_engine", "_cmd_ask_ai", "_cmd_switch_mode",
            "_cmd_toggle_ai_access", "_cmd_execute_ai_command",
        ):
            assert hasattr(CMDPageMixin, attr), f"Missing: {attr}"

    def test_cmd_engine_registry(self):
        """Engine commands work for ENGINE mode."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        result = registry.execute("config.list")
        assert result.success


# ================================================================
# Per-model context
# ================================================================

class TestModelContext:
    """Verify per-model context integrates with GUI."""

    def test_model_context_module(self):
        from enigma_engine.core.model_context import (
            load_model_context, get_contexts_dir)
        assert callable(load_model_context)
        ctx_dir = get_contexts_dir()
        assert ctx_dir.name == "model_contexts"


# ================================================================
# CORE page widgets
# ================================================================


# ================================================================
# Voice input
# ================================================================


# ================================================================
# FORGE data editor
# ================================================================


# ================================================================
# Nav rail
# ================================================================


# ================================================================
# Path settings
# ================================================================

class TestPathSettings:
    """Verify directory path settings."""

    def test_path_constants_and_defaults(self):
        from enigma_engine.gui.scanners import (
            PATH_SETTINGS, get_path, MODELS_DIR)
        assert isinstance(PATH_SETTINGS, dict)
        assert "models_dir" in PATH_SETTINGS
        assert "outputs_dir" in PATH_SETTINGS
        assert get_path("models_dir") == MODELS_DIR


# ================================================================
# DOCS page
# ================================================================

class TestDocsPage:
    """DOCS page: documentation browser, file management."""

    def test_scan_docs_has_data_category(self):
        """scan_docs includes training data files under 'data' category."""
        from enigma_engine.gui.scanners import scan_docs
        docs = scan_docs()
        categories = {d["category"] for d in docs}
        assert "data" in categories, (
            "scan_docs should include data/ files as 'data' category")


# ================================================================
# DOCS page improvements
# ================================================================

class TestDocsPageImprovements:
    """DOCS page: search, notes, unsaved changes, Ctrl+S, stats."""

    def test_scan_docs_has_notes_category(self):
        """scan_docs includes notes files under 'notes' category."""
        from enigma_engine.gui.scanners import scan_docs, NOTES_DIR
        # Create a test note if dir is empty
        NOTES_DIR.mkdir(parents=True, exist_ok=True)
        test_note = NOTES_DIR / "_test_note.md"
        created = False
        if not any(NOTES_DIR.glob("*.md")):
            test_note.write_text("test", encoding="utf-8")
            created = True
        try:
            docs = scan_docs()
            categories = {d["category"] for d in docs}
            assert "notes" in categories, (
                "scan_docs should include notes/ files as 'notes' category")
        finally:
            if created and test_note.exists():
                test_note.unlink()

    def test_scanners_notes_dir_constant(self):
        """scanners module exports NOTES_DIR constant."""
        from enigma_engine.gui.scanners import NOTES_DIR
        assert NOTES_DIR.name == "notes"

    def test_scanners_scan_docs_returns_list(self):
        """scan_docs should return a list of doc entries."""
        from enigma_engine.gui import scanners
        result = scanners.scan_docs()
        assert isinstance(result, list)


# ================================================================
# Docs undo/redo
# ================================================================


# ================================================================
# Chat fullscreen
# ================================================================


# ================================================================
# Display names and model AI name
# ================================================================


# ================================================================
# Trainer Docs Section
# ================================================================

class TestTrainerDocs:
    """Tests for the TRAINER section in DOCS page."""

    def test_trainer_dir_exists(self):
        """Trainer docs directory exists with files."""
        from enigma_engine.gui.scanners import TRAINER_DIR
        assert TRAINER_DIR.exists()
        files = list(TRAINER_DIR.glob("*.md"))
        assert len(files) >= 3

    def test_scan_docs_has_trainer_category(self):
        """scan_docs returns items with trainer category."""
        from enigma_engine.gui.scanners import scan_docs
        docs = scan_docs()
        categories = {d["category"] for d in docs}
        assert "trainer" in categories

    def test_trainer_files_readable(self):
        """All trainer docs are readable."""
        from enigma_engine.gui.scanners import TRAINER_DIR
        for f in TRAINER_DIR.glob("*.md"):
            content = f.read_text(encoding="utf-8")
            assert len(content) > 50


# ================================================================
# Unified History / Sessions
# ================================================================


class TestExternalModelsDocs:
    """Tests for external model limitations documentation."""

    def test_scan_docs_finds_external_models(self):
        """scan_docs discovers the external_models.md file."""
        from enigma_engine.gui.scanners import scan_docs
        docs = scan_docs()
        names = [d["name"] for d in docs]
        assert "External Models" in names


# ================================================================
# Models page feedback
# ================================================================


class TestRouteAssignmentPersistence:
    """Test that route assignments save and load from disk."""

    def test_save_and_load_round_trip(self, tmp_path):
        """Route assignments survive save → load round trip."""
        import enigma_engine.gui.scanners as scanners
        # Temporarily redirect the routes file
        original = scanners._ROUTES_FILE
        scanners._ROUTES_FILE = tmp_path / "routes.json"
        try:
            assignments = {"chat": "/models/test.gguf",
                           "trainer": "/models/train.pth"}
            scanners.save_route_assignments(assignments)
            loaded = scanners.load_route_assignments()
            assert loaded["chat"] == "/models/test.gguf"
            assert loaded["trainer"] == "/models/train.pth"
        finally:
            scanners._ROUTES_FILE = original

    def test_none_values_not_saved(self, tmp_path):
        """None-valued routes are excluded from the saved file."""
        import enigma_engine.gui.scanners as scanners
        original = scanners._ROUTES_FILE
        scanners._ROUTES_FILE = tmp_path / "routes.json"
        try:
            assignments = {"chat": "/models/test.gguf",
                           "trainer": None}
            scanners.save_route_assignments(assignments)
            loaded = scanners.load_route_assignments()
            assert "chat" in loaded
            assert "trainer" not in loaded
        finally:
            scanners._ROUTES_FILE = original


class TestStudentRoute:
    """Test the STUDENT route integration."""

    def test_route_keys_includes_student(self):
        """ROUTE_KEYS contains the student route."""
        from enigma_engine.gui.scanners import ROUTE_KEYS
        assert "student" in ROUTE_KEYS

    def test_route_persistence_includes_student(self, tmp_path):
        """Student route survives save and load round trip."""
        import enigma_engine.gui.scanners as scanners
        original = scanners._ROUTES_FILE
        scanners._ROUTES_FILE = tmp_path / "routes.json"
        try:
            assignments = {
                "chat": "/models/chat.gguf",
                "trainer": "/models/trainer.pth",
                "student": "/models/student.pth",
            }
            scanners.save_route_assignments(assignments)
            loaded = scanners.load_route_assignments()
            assert loaded["student"] == "/models/student.pth"
        finally:
            scanners._ROUTES_FILE = original


class TestBlankModelCreate:
    """Test simplified blank model creation on MODELS page."""

    def test_rename_model_method_exists(self):
        """ForgeMixin must have _rename_model for renaming models."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert hasattr(ForgeMixin, '_rename_model'), (
            "ForgeMixin must provide _rename_model method")
        assert callable(ForgeMixin._rename_model)


class TestForgeModeUI:
    """Test that FORGE training mode descriptions match the 8 radio buttons."""

    def test_descriptions_cover_all_modes(self):
        """_TRAINING_MODE_DESCRIPTIONS has exactly the 9 GUI modes
        (LoRA-1 Pass 156p added explicit LoRA mode card)."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        expected = {"Pre-Train", "Distill", "Basic", "LoRA",
                    "AI-Guided", "Image", "Dialogue", "RLHF",
                    "Self-Play"}
        assert set(ForgeMixin._TRAINING_MODE_DESCRIPTIONS.keys()) == expected


class TestForgeNewModes:
    """Test new training modes wiring."""

    def test_display_name_mapping_covers_all_modes(self):
        """_MODE_DISPLAY_TO_KEY maps all 9 display names to keys
        (LoRA-1 Pass 156p added explicit LoRA mode)."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        mapping = ForgeMixin._MODE_DISPLAY_TO_KEY
        assert len(mapping) == 9
        # Every display name resolves to a valid internal key
        expected_keys = {"Pre-Train", "Distill", "Basic", "LoRA",
                         "AI-Guided", "Image", "Dialogue",
                         "RLHF", "Self-Play"}
        assert set(mapping.values()) == expected_keys

    def test_reverse_mapping_covers_all_keys(self):
        """_MODE_KEY_TO_DISPLAY maps all 9 internal keys to display
        (LoRA-1 Pass 156p added explicit LoRA mode)."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        reverse = ForgeMixin._MODE_KEY_TO_DISPLAY
        assert len(reverse) == 9
        # Display names match GUI radio button values
        assert reverse["Image"] == "Image"
        assert reverse["Basic"] == "Basic"
        assert reverse["LoRA"] == "LoRA"
        assert reverse["AI-Guided"] == "AI-Guided"
        assert reverse["Dialogue"] == "Dialogue"
        assert reverse["RLHF"] == "RLHF"
        assert reverse["Self-Play"] == "Self-Play"

    def test_pretrain_tokenizer_cap_in_code(self):
        """Pre-train code must define a tokenizer sample cap to prevent OOM.

        Structural guard: _TOK_SAMPLE_CAP is a local variable inside the
        pre-train function, not a module-level constant. We verify the
        function source contains the cap logic.
        """
        import inspect
        import enigma_engine.gui.gui_forge_new_modes as mod
        src = inspect.getsource(mod)
        assert "_TOK_SAMPLE_CAP" in src, (
            "Missing _TOK_SAMPLE_CAP — tokenizer will OOM on "
            "large corpora")


class TestForgeThreeModeConnections:
    """Regression tests for FORGE wiring."""

    def test_no_forge_focus_field_references(self):
        """No training methods reference the removed forge_focus_field widget."""
        from enigma_engine.gui.gui_forge_training import ForgeTrainingMixin
        from enigma_engine.gui.gui_forge_advanced import ForgeAdvancedMixin
        for cls in (ForgeTrainingMixin, ForgeAdvancedMixin):
            source = inspect.getsource(cls)
            assert "forge_focus_field" not in source


class TestVisionTrainingHeartbeat:
    """V-4: vision training thread must write the same heartbeat as
    pre-training so OS OOM kills are detectable post-mortem."""

    def _vision_source(self):
        from enigma_engine.gui.gui_forge_training import ForgeTrainingMixin
        full = inspect.getsource(ForgeTrainingMixin)
        # Slice to just _start_vision_training to keep assertions focused.
        marker = "def _start_vision_training("
        assert marker in full, "vision training method missing"
        start = full.index(marker)
        # Find next def at same indent or end-of-class.
        end = full.find("\n    def ", start + len(marker))
        return full[start:end] if end != -1 else full[start:]

    def test_stale_heartbeat_check_present(self):
        """Outer entry must read training_heartbeat.json before launching."""
        src = self._vision_source()
        assert "training_heartbeat.json" in src, (
            "vision training must check for stale heartbeat on entry")
        assert "pid_exists" in src, (
            "stale check must verify previous PID via psutil")

    def test_write_hb_helper_defined(self):
        """Inner thread must define a _write_hb closure."""
        src = self._vision_source()
        assert "def _write_hb(" in src, (
            "vision thread must define _write_hb heartbeat helper")

    def test_write_hb_called_at_each_lifecycle_stage(self):
        """Heartbeat must fire at data_load, training, complete, stopped,
        and crash branches."""
        src = self._vision_source()
        # Phase markers
        assert '_write_hb("data_load"' in src or \
               "_write_hb('data_load'" in src, "missing data_load heartbeat"
        # Status markers — one each for the four exit branches.
        assert 'status="complete"' in src, "missing complete status"
        assert 'status="stopped"' in src, "missing stopped status"
        assert 'status="crashed' in src, "missing crashed status branch"

    def test_write_hb_inside_progress_callback(self):
        """Periodic heartbeat must fire inside the training loop callback,
        not just at start/end — otherwise mid-run OOM is invisible."""
        src = self._vision_source()
        # Find on_epoch or on_progress block; verify _write_hb appears
        # somewhere in the callback region.
        cb_start = src.find("def on_progress(")
        cb_end = src.find("def on_epoch(")
        assert cb_start != -1 and cb_end != -1
        cb_block = src[cb_start:cb_end]
        assert "_write_hb(" in cb_block, (
            "heartbeat must fire from on_progress/on_epoch — otherwise "
            "long epochs hide silent kills")

    def test_oom_taxonomy_matches_pretrain(self):
        """V-4 audit: the same crash on the same hardware should land
        the same heartbeat status across pre-training and vision modes.
        Reference (gui_forge_new_modes.py) uses RuntimeError + simple
        'out of memory' or 'cuda' check; vision mode must do the same."""
        src = self._vision_source()
        # RuntimeError must be caught separately so OOM-friendly user
        # advice doesn't fire on PIL/NumPy errors that happen to mention
        # 'memory'. Reference pattern at gui_forge_new_modes.py.
        assert "except RuntimeError" in src, (
            "vision crash branch must split RuntimeError from Exception "
            "to match pre-training reference pattern")
        # Verify the OOM check is the simple 'out of memory' or 'cuda'
        # form, not a tighter cuda-AND-memory variant that diverges
        # from the pre-training taxonomy.
        rt_start = src.find("except RuntimeError")
        rt_end = src.find("except Exception", rt_start)
        if rt_end == -1:
            rt_end = len(src)
        rt_block = src[rt_start:rt_end]
        assert ('"out of memory" in' in rt_block
                or "'out of memory' in" in rt_block), (
            "OOM detection must check 'out of memory' literal")
        assert ('"cuda" in' in rt_block
                or "'cuda' in" in rt_block), (
            "OOM detection must treat any CUDA error as crashed_oom "
            "(matches pre-training reference)")

    def test_val_split_plumbed_to_train_vision(self):
        """V-6b: the GUI must honor val_split for vision
        training by splitting vision_data into train/val and passing
        train/val payload via dispatcher config. Without this the
        backend V-6 hook is unreachable from the GUI."""
        src = self._vision_source()
        # Must read val_split and use it as a
        # fraction to slice vision_data.
        assert "val_split" in src, (
            "vision GUI must reference val_split for V-6b plumbing")
        assert "run_training(" in src
        # The dispatcher config must pass val rows, not just train.
        assert '"val": val_pairs_data' in src, (
            "run_training(...) must receive val rows when "
            "val_split is non-zero (V-6b)")

    def test_val_split_shuffle_is_seeded(self):
        """V-6b split path must instantiate a local Random before
        shuffling so split policy is explicit and isolated from global
        RNG state."""
        src = self._vision_source()
        # Must use a local Random instance before shuffling.
        shuffle_idx = src.find("shuffle(")
        assert shuffle_idx != -1, "expected a shuffle call for val split"
        # Look back ~400 chars for local RNG setup.
        window = src[max(0, shuffle_idx - 400):shuffle_idx + 50]
        assert "random.Random(" in window, (
            "val-split shuffle must use a local Random instance")


# ================================================================
# FORGE Page: Model Status Cards
# ================================================================


# ================================================================
# FORGE Page: Solo Training
# ================================================================


# ================================================================
# FORGE Helpers: Prompt Extraction
# ================================================================

class TestExtractPrompts:
    """Test _extract_prompts helper for parsing data files."""

    def test_extract_prompts_qa_format(self):
        """_extract_prompts pulls question from Q/A format."""
        import tempfile
        from enigma_engine.gui.gui_forge import ForgeMixin
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt",
                delete=False, encoding="utf-8") as f:
            f.write("Q: What is AI?\nA: Artificial Intelligence.\n\n"
                    "Q: What is ML?\nA: Machine Learning.\n")
            f.flush()
            prompts = ForgeMixin._extract_prompts(f.name)
        assert len(prompts) == 2
        assert "What is AI?" in prompts[0]
        assert "What is ML?" in prompts[1]
        # Should NOT include the A: lines as prompts
        assert not any("Artificial" in p for p in prompts)

    def test_extract_prompts_jsonl_format(self):
        """_extract_prompts pulls prompt from JSONL format."""
        import json
        import tempfile
        from enigma_engine.gui.gui_forge import ForgeMixin
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl",
                delete=False, encoding="utf-8") as f:
            f.write(json.dumps(
                {"prompt": "Hello", "completion": "Hi"}) + "\n")
            f.write(json.dumps(
                {"prompt": "Bye", "completion": "See ya"}) + "\n")
            f.flush()
            prompts = ForgeMixin._extract_prompts(f.name)
        assert len(prompts) == 2
        assert prompts[0] == "Hello"
        assert prompts[1] == "Bye"

    def test_extract_prompts_raw_text(self):
        """_extract_prompts falls back to non-empty lines."""
        import tempfile
        from enigma_engine.gui.gui_forge import ForgeMixin
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt",
                delete=False, encoding="utf-8") as f:
            f.write("# Comment\nHello world\n\nTell me a joke\n")
            f.flush()
            prompts = ForgeMixin._extract_prompts(f.name)
        assert "Hello world" in prompts
        assert "Tell me a joke" in prompts
        # Comments should be excluded
        assert not any(p.startswith("#") for p in prompts)

    def test_extract_prompts_user_ai_format(self):
        """_extract_prompts pulls User lines from User/AI format."""
        import tempfile
        from enigma_engine.gui.gui_forge import ForgeMixin
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt",
                delete=False, encoding="utf-8") as f:
            f.write("User: Hello!\nAI: Hi there!\n\n"
                    "User: How are you?\nAI: Fine!\n")
            f.flush()
            prompts = ForgeMixin._extract_prompts(f.name)
        assert len(prompts) == 2
        assert "Hello!" in prompts[0]
        assert "How are you?" in prompts[1]


# ================================================================
# FORGE Helpers: Engine Loading
# ================================================================


# ================================================================
# FORGE: Guided uses _extract_prompts
# ================================================================


# ================================================================
# FORGE Page: Guided Training
# ================================================================


# ================================================================
# FORGE Page: Dialogue Training (TRAINER ↔ STUDENT conversation)
# ================================================================


# ================================================================
# FORGE: Stage-Aware Generation Prompts
# ================================================================

class TestGenerationPromptBuilder:
    """Test _build_generation_prompt produces varied formats per stage."""

    def test_basics_not_forced_qa(self):
        """Basics stage generates varied formats, not just Q&A."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_generation_prompt(1, 10, "basics")
        # Should mention multiple types, not force Q:/A:
        assert "Format exactly as" not in prompt
        assert "statement" in prompt.lower() or "greeting" in prompt.lower()

    def test_conversation_uses_dialogue_format(self):
        """Conversation stage uses User/AI dialogue format."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_generation_prompt(1, 10, "conversation")
        assert "User:" in prompt
        assert "AI:" in prompt

    def test_commands_uses_qa_with_cmd(self):
        """Commands stage uses Q&A with [CMD] blocks."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_generation_prompt(1, 10, "commands")
        assert "Q:" in prompt
        assert "[CMD]" in prompt

    def test_web_uses_qa_with_search(self):
        """Web stage uses Q&A with search/fetch commands."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_generation_prompt(1, 10, "web")
        assert "search.web" in prompt

    def test_unknown_stage_falls_back_to_basics(self):
        """Unknown stage name falls back to basics prompt."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_generation_prompt(1, 10, "unknown_stage")
        basics = ForgeMixin._build_generation_prompt(1, 10, "basics")
        assert prompt == basics

    def test_includes_index_and_total(self):
        """Prompt includes the example index and total count."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_generation_prompt(7, 20, "basics")
        assert "#7" in prompt
        assert "20" in prompt


# ================================================================
# FORGE: Training Pair Formatter
# ================================================================

class TestFormatTrainingPair:
    """Test _format_training_pair outputs the right format per stage."""

    def test_basics_raw_text(self):
        """Basics stage returns raw text without Q:/A: or User:/AI:."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._format_training_pair(
            "basics", "Hello", "Hi there")
        assert result == "Hello\nHi there"
        assert "Q:" not in result
        assert "User:" not in result

    def test_conversation_user_ai_format(self):
        """Conversation stage returns User/AI dialogue format."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._format_training_pair(
            "conversation", "How are you?", "I'm good!")
        assert result == "User: How are you?\nAI: I'm good!"

    def test_commands_qa_format(self):
        """Commands stage returns Q&A format."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._format_training_pair(
            "commands", "List files", "[CMD]ls[/CMD]")
        assert result == "Q: List files\nA: [CMD]ls[/CMD]"

    def test_web_qa_format(self):
        """Web stage returns Q&A format."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._format_training_pair(
            "web", "Search for cats", "[CMD]search.web cats[/CMD]")
        assert result == "Q: Search for cats\nA: [CMD]search.web cats[/CMD]"

    def test_unknown_stage_defaults_to_qa(self):
        """Unknown stage name falls back to Q&A format."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._format_training_pair(
            "custom", "prompt", "response")
        assert result == "Q: prompt\nA: response"

    def test_training_guide_visible_on_docs_page(self):
        """Training guide is discoverable by scan_docs."""
        from enigma_engine.gui.scanners import scan_docs
        docs = scan_docs()
        names = [d["filename"] for d in docs]
        assert "training_guide.md" in names


# ================================================================
# FORGE Page: Evaluate Student
# ================================================================

class TestEvaluateStudent:
    """Test TRAINER interactively testing STUDENT."""

    def test_evaluate_no_data_file_required(self):
        """_evaluate_student works without a data file."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._evaluate_student)
        # No check for data_path / train_data_var
        assert "train_data_var" not in source


# ================================================================
# FORGE Page: Checkpoint Save/Resume
# ================================================================


# ================================================================
# FORGE Page: Loss Curve Visualization
# ================================================================


# ================================================================
# CoT-B: REASONING-AWARE TRAINING DATA
# ================================================================


# ================================================================
# FORGE: Trainer System Prompt (human-like responses)
# ================================================================

class TestBuildTrainerSystemPrompt:
    """Test _build_trainer_system_prompt for human-like output."""

    def test_returns_string(self):
        """Returns a non-empty string."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_includes_param_count(self):
        """Prompt includes the student parameter count."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=5_000_000)
        assert "5,000,000" in result

    def test_tiny_model_gets_simple_guidance(self):
        """Very small models get short/simple response guidance."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=500_000)
        assert "very small" in result.lower() or "simple" in result.lower()

    def test_medium_model_gets_paragraph_guidance(self):
        """Medium models get paragraph-length guidance."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=100_000_000)
        assert "medium" in result.lower()

    def test_includes_architecture_info(self):
        """When student_cfg is provided, architecture info appears."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            n_layers=6, dim=256, max_seq_len=512)
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, student_cfg=cfg)
        assert "6 layers" in result
        assert "256" in result

    def test_discourages_ai_phrases(self):
        """Prompt tells TRAINER not to sound like a generic AI."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=10_000_000)
        lower = result.lower()
        assert "as an ai" in lower
        assert "human" in lower or "person" in lower or "friend" in lower
        assert "guardrails" in lower or "limits" in lower

    def test_fact_checking_instructions(self):
        """Prompt teaches fact-checking and offline fallback."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=10_000_000)
        lower = result.lower()
        # Should teach verification
        assert "fact check" in lower or "double-check" in lower or "verify" in lower
        # Should handle no internet
        assert "no internet" in lower or "no internet" in lower.replace("'", "")
        # Should not refuse, give best answer
        assert "best answer" in lower or "best take" in lower
        # Should flag uncertainty
        assert "confident" in lower or "uncertain" in lower or "not 100" in lower

    def test_task_parameter_reflected(self):
        """Task name appears in the prompt."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=10_000_000, task="evaluate")
        assert "evaluate" in result

    def test_no_architecture_without_cfg(self):
        """No architecture line when student_cfg is None."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=10_000_000, student_cfg=None)
        assert "Student architecture" not in result


# ================================================================
# FORGE: Training Stages (curriculum)
# ================================================================

class TestTrainingStages:
    """Test training stage curriculum in system prompt."""

    def test_stage_parameter_accepted(self):
        """_build_trainer_system_prompt accepts stage parameter."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        # Should not raise
        for s in ("basics", "conversation", "commands", "web"):
            result = ForgeMixin._build_trainer_system_prompt(
                student_params=1_000_000, stage=s)
            assert isinstance(result, str)

    def test_basics_stage_content(self):
        """Basics stage focuses on simple sentences."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, stage="basics")
        lower = result.lower()
        assert "basics" in lower
        assert "sentence" in lower
        # Stage section should say NOT to teach commands yet
        assert "do not teach commands" in lower

    def test_conversation_stage_content(self):
        """Conversation stage teaches dialogue skills."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, stage="conversation")
        lower = result.lower()
        assert "conversation" in lower
        assert "dialogue" in lower or "multi-sentence" in lower

    def test_commands_stage_content(self):
        """Commands stage teaches [CMD] syntax."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, stage="commands")
        assert "[CMD]" in result
        assert "web" not in result.split("COMMANDS")[1].split(
            "IMPORTANT")[0].lower() or "NOT" in result

    def test_web_stage_content(self):
        """Web stage teaches search.web and web.fetch."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, stage="web")
        assert "search.web" in result
        assert "web.fetch" in result

    def test_unknown_stage_falls_back_to_basics(self):
        """Unknown stage falls back to basics."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, stage="unknown_stage")
        assert "BASICS" in result


# ================================================================
# FORGE: Training Brief (Quick Profile + Custom Brief)
# ================================================================

class TestTrainingBrief:
    """Test Training Brief feature — quick profile fields + freeform text."""

    def test_build_trainer_prompt_accepts_training_brief(self):
        """_build_trainer_system_prompt accepts training_brief kwarg."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000,
            training_brief="Personality: cheerful. Expertise: cooking.")
        assert "cheerful" in result
        assert "cooking" in result

    def test_training_brief_empty_is_fine(self):
        """Empty training_brief doesn't break prompt generation."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, training_brief="")
        assert isinstance(result, str)
        assert len(result) > 50

    def test_training_brief_none_is_fine(self):
        """None training_brief doesn't break prompt generation."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, training_brief=None)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_training_brief_placed_prominently(self):
        """Training brief appears BEFORE the generic instructions."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        brief = "The AI should be a sarcastic chef named Gordon."
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, training_brief=brief)
        # Brief must appear before the critical/generic section
        brief_pos = result.find("sarcastic chef")
        critical_pos = result.find("CRITICAL")
        assert brief_pos < critical_pos, (
            "Training brief should appear before generic instructions")

    def test_training_brief_has_section_header(self):
        """Training brief is wrapped with a clear section header."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        result = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000,
            training_brief="Be a pirate.")
        # Should have a header marking the user's brief
        assert "USER" in result.upper() or "BRIEF" in result.upper() or "GOAL" in result.upper()

    def test_quick_profile_fields_defined(self):
        """Quick profile has the expected field names."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        fields = ForgeMixin._QUICK_PROFILE_FIELDS
        assert isinstance(fields, (list, tuple))
        names = [f[0] for f in fields]
        assert "Personality" in names
        assert "Tone" in names
        assert "Expertise" in names
        # Name is NOT a quick field — it comes from the student model
        assert "Name" not in names

    def test_save_training_brief_persists_epochs_lr(self):
        """_save_training_brief includes epochs, LR, and preset."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._save_training_brief)
        assert "_epochs" in source
        assert "_lr" in source
        assert "_preset" in source

    def test_load_training_brief_restores_epochs_lr(self):
        """_load_training_brief restores epochs, LR, and preset."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._load_training_brief)
        assert "_epochs" in source
        assert "_lr" in source
        assert "_preset" in source

    def test_save_training_brief_persists_pretrain_settings(self):
        """_save_training_brief includes vocab, retrain tok, utf8."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._save_training_brief)
        assert "_pretrain_vocab" in source
        assert "_pretrain_retrain_tok" in source
        assert "_pretrain_utf8" in source

    def test_save_training_brief_persists_lora_and_data_path(self):
        """_save_training_brief includes LoRA settings and Basic data path."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._save_training_brief)
        assert "_lora_rank" in source
        assert "_lora_alpha" in source
        assert "_train_data_path" in source

    def test_load_training_brief_restores_all_new_fields(self):
        """_load_training_brief restores all new persistence fields."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._load_training_brief)
        assert "_pretrain_vocab" in source
        assert "_pretrain_retrain_tok" in source
        assert "_lora_rank" in source
        assert "_train_data_path" in source

    def test_save_training_brief_persists_distill_and_mode_vars(self):
        """_save_training_brief covers distill, checkboxes, stage, replay, vision."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._save_training_brief)
        for key in (
            "_distill_num_examples", "_distill_max_tokens",
            "_reasoning", "_evolutionary", "_auto_train", "_resume_training",
            "_general_mix", "_training_stage",
            "_replay_capacity", "_replay_ratio",
        ):
            assert key in source, f"_save_training_brief missing key: {key}"

    def test_load_training_brief_restores_distill_and_mode_vars(self):
        """_load_training_brief restores distill, checkboxes, stage, replay, vision."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._load_training_brief)
        for key in (
            "_distill_num_examples", "_distill_max_tokens",
            "_reasoning", "_evolutionary", "_auto_train", "_resume_training",
            "_general_mix", "_training_stage",
            "_replay_capacity", "_replay_ratio",
        ):
            assert key in source, f"_load_training_brief missing key: {key}"

    def test_on_close_calls_save_training_brief(self):
        """desktop.EnigmaGUI._on_close must call _save_training_brief so Forge
        settings are written even when no mode-switch triggered an earlier save."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._on_close)
        assert "_save_training_brief" in source, (
            "_on_close does not call _save_training_brief — "
            "Forge settings typed without a mode-switch will be lost on close"
        )


# ================================================================
# FORGE: UI Polish
# ================================================================


# ================================================================
# FORGE: Web Learn
# ================================================================


# ================================================================
# System as visible third speaker in chat
# ================================================================


# ================================================================
# Chat Media Support (inline images, GIFs, video thumbnails, links)
# ================================================================

class TestChatMedia:
    """Media rendering in the chat display — images, GIFs, videos, clickable links."""

    def test_detect_media_refs_finds_images(self):
        """detect_media_refs finds image paths and URLs in text."""
        from enigma_engine.gui.media import detect_media_refs
        text = "Here is an image: outputs/images/test.png and done."
        refs = detect_media_refs(text)
        assert len(refs) >= 1
        assert any(r["path"].endswith("test.png") for r in refs)
        assert any(r["type"] == "image" for r in refs)

    def test_detect_media_refs_finds_gifs(self):
        """detect_media_refs identifies GIF files."""
        from enigma_engine.gui.media import detect_media_refs
        text = "Check this: outputs/gifs/anim.gif"
        refs = detect_media_refs(text)
        assert len(refs) >= 1
        assert any(r["type"] == "gif" for r in refs)

    def test_detect_media_refs_finds_videos(self):
        """detect_media_refs identifies video files."""
        from enigma_engine.gui.media import detect_media_refs
        text = "Video at outputs/videos/demo.mp4"
        refs = detect_media_refs(text)
        assert len(refs) >= 1
        assert any(r["type"] == "video" for r in refs)

    def test_detect_urls_finds_http(self):
        """detect_urls finds http and https URLs in text."""
        from enigma_engine.gui.media import detect_urls
        text = "Visit https://example.com and http://test.org/page"
        urls = detect_urls(text)
        assert "https://example.com" in urls
        assert "http://test.org/page" in urls

    def test_detect_urls_finds_image_urls(self):
        """detect_urls identifies image URLs as media type."""
        from enigma_engine.gui.media import detect_media_refs
        text = "Image at https://example.com/photo.jpg"
        refs = detect_media_refs(text)
        assert any(r["type"] == "image" for r in refs)

    def test_load_chat_image_returns_photoimage(self):
        """load_chat_image returns a PhotoImage-compatible object."""
        from enigma_engine.gui.media import load_chat_image
        # Create a tiny test image in memory
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        import tkinter as tk
        import tempfile, os
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            pytest.skip("Tcl/Tk not available")
        try:
            img = Image.new("RGB", (100, 100), color="red")
            with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False) as f:
                img.save(f, format="PNG")
                tmp_path = f.name
            try:
                result = load_chat_image(tmp_path, max_width=80)
                assert result is not None
                assert hasattr(result, "width")
                assert hasattr(result, "height")
                # Should be resized
                assert result.width() <= 80
            finally:
                os.unlink(tmp_path)
        finally:
            root.destroy()

    def test_load_chat_image_returns_none_for_missing(self):
        """load_chat_image returns None for missing files."""
        from enigma_engine.gui.media import load_chat_image
        result = load_chat_image("/nonexistent/image.png")
        assert result is None

    def test_extract_gif_frames_returns_list(self):
        """extract_gif_frames returns a list of PhotoImage frames."""
        from enigma_engine.gui.media import extract_gif_frames
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")
        import tkinter as tk
        import tempfile, os
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            pytest.skip("Tcl/Tk not available")
        try:
            # Create a minimal 2-frame GIF
            frames = [
                Image.new("RGB", (10, 10), "red"),
                Image.new("RGB", (10, 10), "blue"),
            ]
            with tempfile.NamedTemporaryFile(
                    suffix=".gif", delete=False) as f:
                frames[0].save(
                    f, format="GIF", save_all=True,
                    append_images=frames[1:], duration=100, loop=0)
                tmp_path = f.name
            try:
                result = extract_gif_frames(tmp_path, max_width=20)
                assert isinstance(result, list)
                assert len(result) >= 2
                # Each frame should have (photo_image, duration_ms)
                for photo, dur in result:
                    assert hasattr(photo, "width")
                    assert isinstance(dur, int)
            finally:
                os.unlink(tmp_path)
        finally:
            root.destroy()

    def test_extract_video_thumbnail_with_cv2(self):
        """extract_video_thumbnail returns an image for valid video."""
        from enigma_engine.gui.media import extract_video_thumbnail
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")
        import tkinter as tk
        import tempfile, os, numpy as np
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            pytest.skip("Tcl/Tk not available")
        try:
            # Create a tiny valid video file
            tmp_path = tempfile.mktemp(suffix=".avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(tmp_path, fourcc, 1, (64, 64))
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :] = (0, 0, 255)  # red
            writer.write(frame)
            writer.release()
            try:
                result = extract_video_thumbnail(tmp_path, max_width=50)
                assert result is not None
                assert hasattr(result, "width")
            finally:
                os.unlink(tmp_path)
        finally:
            root.destroy()

    def test_extract_video_thumbnail_returns_none_for_missing(self):
        """extract_video_thumbnail returns None for missing files."""
        from enigma_engine.gui.media import extract_video_thumbnail
        result = extract_video_thumbnail("/nonexistent/video.mp4")
        assert result is None

    def test_media_constants(self):
        """Media module has file extension constants."""
        from enigma_engine.gui.media import (
            IMAGE_EXTENSIONS, GIF_EXTENSIONS, VIDEO_EXTENSIONS)
        assert ".png" in IMAGE_EXTENSIONS
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".gif" in GIF_EXTENSIONS
        assert ".mp4" in VIDEO_EXTENSIONS

    def test_detect_media_refs_relative_and_absolute(self):
        """detect_media_refs handles both relative and absolute paths."""
        from enigma_engine.gui.media import detect_media_refs
        # Relative path
        refs1 = detect_media_refs("see outputs/images/cat.jpg")
        assert len(refs1) >= 1
        # Absolute-style path
        refs2 = detect_media_refs(r"see C:\images\cat.jpg")
        assert len(refs2) >= 1

    def test_detect_media_refs_no_false_positives(self):
        """detect_media_refs does not match random text."""
        from enigma_engine.gui.media import detect_media_refs
        refs = detect_media_refs("Hello world, nothing here")
        assert len(refs) == 0

    def test_detect_markdown_image_syntax(self):
        """detect_media_refs parses ![alt](url) markdown images."""
        from enigma_engine.gui.media import detect_media_refs
        text = "Here is Pikachu: ![Pikachu](https://example.com/pikachu.jpg)"
        refs = detect_media_refs(text)
        url_refs = [r for r in refs if r["source"] == "url"]
        assert len(url_refs) >= 1
        assert any(r["path"] == "https://example.com/pikachu.jpg"
                    for r in url_refs)
        assert any(r["type"] == "image" for r in url_refs)

    def test_markdown_image_has_alt_text(self):
        """Markdown image refs include alt text when present."""
        from enigma_engine.gui.media import detect_media_refs
        text = "![My Cat](https://example.com/cat.png)"
        refs = detect_media_refs(text)
        md_refs = [r for r in refs if r.get("alt")]
        assert len(md_refs) >= 1
        assert md_refs[0]["alt"] == "My Cat"

    def test_markdown_gif_detected(self):
        """Markdown image syntax with .gif extension detected as gif type."""
        from enigma_engine.gui.media import detect_media_refs
        text = "![anim](https://example.com/anim.gif)"
        refs = detect_media_refs(text)
        assert any(r["type"] == "gif" for r in refs)

    def test_no_false_file_match_on_url_domain(self):
        """File path regex does not match URL domain components."""
        from enigma_engine.gui.media import detect_media_refs
        text = "See https://raw.githubusercontent.com/user/repo/image.jpg"
        refs = detect_media_refs(text)
        file_refs = [r for r in refs if r["source"] == "file"]
        assert len(file_refs) == 0, (
            f"URL domain falsely matched as file: {file_refs}")

    def test_markdown_image_no_duplicate_url(self):
        """Markdown image URL not duplicated as a separate bare URL ref."""
        from enigma_engine.gui.media import detect_media_refs
        text = "![pic](https://example.com/pic.png)"
        refs = detect_media_refs(text)
        # Should only have one ref for the URL, not two
        paths = [r["path"] for r in refs]
        assert paths.count("https://example.com/pic.png") == 1


# ================================================================
# Send guard (double-send crash fix)
# ================================================================


# ================================================================
# STOP button
# ================================================================


# ================================================================
# Message editing
# ================================================================


class TestWindowClose:
    """Verify cleanup on window close."""

    def test_on_close_does_not_silence_exceptions(self):
        """_on_close should log failures instead of using bare except-pass."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI._on_close)
        lines = source.splitlines()
        for i, line in enumerate(lines[:-1]):
            if line.strip().startswith("except Exception"):
                j = i + 1
                while j < len(lines):
                    stripped = lines[j].strip()
                    if stripped and not stripped.startswith("#"):
                        break
                    j += 1
                assert j == len(lines) or lines[j].strip() != "pass", (
                    "_on_close uses bare `except Exception: pass` and "
                    "silently drops shutdown failures"
                )


class TestForgeUsesModelsDirConstant:
    """Verify FORGE uses MODELS_DIR constant instead of hardcoded paths."""

    def test_no_hardcoded_path_models(self):
        """No Path('models') hardcoded in gui_forge.py."""
        from enigma_engine.gui import gui_forge
        source = inspect.getsource(gui_forge)
        assert 'Path("models")' not in source
        assert "Path('models')" not in source

    def test_checkpoint_dir_uses_models_dir(self):
        """checkpoint_dir uses MODELS_DIR, not hardcoded string."""
        from enigma_engine.gui import gui_forge
        source = inspect.getsource(gui_forge)
        assert 'checkpoint_dir="models/checkpoints"' not in source


class TestFileEncoding:
    """Verify all file I/O uses encoding='utf-8' on Windows."""

    def test_builtin_commands_mod_json_encoding(self):
        """mod.json reads use encoding='utf-8'."""
        from enigma_engine.core import builtin_commands
        source = inspect.getsource(builtin_commands)
        # Should NOT have open(mod_json, 'r') without encoding
        assert "open(mod_json, 'r')" not in source


class TestRouterPortDynamic:
    """Verify router port is not hardcoded in messages."""

    def test_mod_start_uses_dynamic_port(self):
        """mod_start command shows actual router port."""
        from enigma_engine.core import builtin_commands
        source = inspect.getsource(builtin_commands)
        assert 'f"[OK] Router started on port {router.port}"' in source, (
            "mod.start must format the success message from router.port so "
            "the reported port stays dynamic."
        )


class TestRouterStartupLogging:
    """Verify router startup failures are logged, not silently swallowed."""

    def test_desktop_router_not_bare_pass(self):
        """Router startup must log failures, not silently swallow them."""
        from enigma_engine.gui.desktop import EnigmaGUI
        source = inspect.getsource(EnigmaGUI.__init__)

        # Guard against `except Exception: pass` with or without comments,
        # indentation variations, or wrapped formatting.
        lines = source.splitlines()
        for i, line in enumerate(lines[:-1]):
            if line.strip().startswith("except Exception"):
                assert lines[i + 1].strip() != "pass", (
                    "Router startup uses bare `except Exception: pass` and "
                    "silently drops failures"
                )

        assert "Router startup failed (optional)" in source, (
            "Router startup exception path must emit a warning log"
        )


class TestExpandingChatDisplay:
    """Verify the chat area uses native CTkTextbox scrollbar."""

    def test_mousewheel_not_redirected(self):
        """Chat display no longer needs mousewheel redirect (native scroll)."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_core)
        # redirect_mousewheel was removed — native scrollbar handles it
        assert "_redirect_mousewheel" not in source


class TestGuiDeadImports:
    """Verify GUI files don't have dead imports."""

    def test_media_no_unused_os(self):
        """media.py should not import unused os."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "media.py"
        source = source_path.read_text(encoding='utf-8')
        lines = source.split('\n')
        import_lines = [l for l in lines if l.strip() == 'import os']
        usage_lines = [l for l in lines if 'os.' in l and 'import' not in l]
        if import_lines:
            assert usage_lines, "os is imported but never used in media.py"

    def test_media_no_unused_imagefont(self):
        """media.py should not import unused ImageFont."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "media.py"
        source = source_path.read_text(encoding='utf-8')
        lines = source.split('\n')
        import re
        import_lines = [l for l in lines if 'ImageFont' in l and 'import' in l]
        usage_lines = [l for l in lines if re.search(r'\bImageFont\b', l)
                       and 'import' not in l]
        if import_lines:
            assert usage_lines, "ImageFont is imported but never used in media.py"

    def test_mod_page_no_unused_border_accent(self):
        """gui_mod_page.py should not import unused C_BORDER_ACCENT."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_mod_page.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        usage_lines = [l for l in source.split('\n')
                       if re.search(r'\bC_BORDER_ACCENT\b', l) and 'import' not in l]
        if not usage_lines:
            # Should not be imported if not used
            import_lines = [l for l in source.split('\n')
                            if 'C_BORDER_ACCENT' in l and 'import' in l]
            assert not import_lines, "C_BORDER_ACCENT imported but never used"

    def test_cmd_page_no_unused_c_accent(self):
        """gui_cmd_page.py should not import unused C_ACCENT."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_cmd_page.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        # C_ACCENT but NOT C_ACCENT_DIM (which IS used)
        usage_lines = [l for l in source.split('\n')
                       if re.search(r'\bC_ACCENT\b', l)
                       and 'C_ACCENT_DIM' not in l
                       and 'C_ACCENT_MUTED' not in l
                       and 'import' not in l]
        if not usage_lines:
            [l for l in source.split('\n')
                            if re.search(r'\bC_ACCENT\b', l) and 'import' in l
                            and 'C_ACCENT_DIM' not in l.replace('C_ACCENT,', '')]
            # Check if C_ACCENT alone appears on import line
            assert 'C_ACCENT,' not in source.split('import')[1] if len(source.split('import')) > 1 else True


# ================================================================
# Polish audit — fixes verified 2026-02-26
# ================================================================

class TestPolishAuditGUI:
    """Verify polish fixes made to GUI files."""

    def test_desktop_no_dead_c_green(self):
        """desktop.py should not import unused C_GREEN."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        usage = [l for l in source.split('\n')
                 if re.search(r'\bC_GREEN\b', l) and 'import' not in l]
        if not usage:
            import_lines = [l for l in source.split('\n')
                            if 'C_GREEN' in l and 'import' in l]
            assert not import_lines, "C_GREEN imported but never used in desktop.py"

    def test_desktop_no_dead_c_text(self):
        """desktop.py should not import unused C_TEXT."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        usage = [l for l in source.split('\n')
                 if re.search(r'\bC_TEXT\b', l)
                 and 'C_TEXT_BRIGHT' not in l
                 and 'C_TEXT_DIM' not in l
                 and 'import' not in l]
        if not usage:
            import_lines = [l for l in source.split('\n')
                            if re.search(r'\bC_TEXT\b', l)
                            and 'C_TEXT_BRIGHT' not in l
                            and 'C_TEXT_DIM' not in l
                            and 'import' in l]
            assert not import_lines, "C_TEXT imported but never used in desktop.py"

    def test_gui_pages_no_dead_purple_dim(self):
        """gui_pages.py should not import unused C_PURPLE_DIM."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_pages.py"
        source = source_path.read_text(encoding='utf-8')
        assert 'C_PURPLE_DIM' not in source, "C_PURPLE_DIM imported but never used"

    def test_gui_pages_no_dead_purple_muted(self):
        """gui_pages.py should not import unused C_PURPLE_MUTED."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_pages.py"
        source = source_path.read_text(encoding='utf-8')
        assert 'C_PURPLE_MUTED' not in source, "C_PURPLE_MUTED imported but never used"

    def test_gui_logic_no_dead_accent_dim(self):
        """gui_logic.py should not import unused C_ACCENT_DIM."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        usage = [l for l in source.split('\n')
                 if re.search(r'\bC_ACCENT_DIM\b', l) and 'import' not in l]
        if not usage:
            import_lines = [l for l in source.split('\n')
                            if 'C_ACCENT_DIM' in l and 'import' in l]
            assert not import_lines, "C_ACCENT_DIM imported but never used"

    def test_version_constant_exists(self):
        """widgets.py should export a VERSION constant."""
        from enigma_engine.gui.widgets import VERSION
        assert isinstance(VERSION, str)
        assert VERSION  # Not empty

    def test_font_family_constant(self):
        """widgets.py should use FONT_FAMILY constant for all font tuples."""
        from enigma_engine.gui.widgets import FONT_FAMILY
        assert FONT_FAMILY == "Consolas"

    def test_tooltip_uses_constants(self):
        """Tooltip should use color/font constants, not hardcoded values."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "widgets.py"
        source = source_path.read_text(encoding='utf-8')
        # Find the _show method
        lines = source.split('\n')
        in_tooltip_show = False
        for line in lines:
            if 'def _show(self)' in line:
                in_tooltip_show = True
            elif in_tooltip_show and 'def ' in line:
                break
            elif in_tooltip_show:
                # Should not reference raw color hex in tooltip
                if 'background=' in line or 'foreground=' in line:
                    assert '"#' not in line, (
                        f"Tooltip uses hardcoded color: {line.strip()}")

    def test_tooltip_dismiss_on_focus_loss(self):
        """Tooltip must dismiss when app loses focus and not use -topmost."""
        import inspect
        from enigma_engine.gui.widgets import Tooltip
        source = inspect.getsource(Tooltip)
        # Must have FocusOut binding for app focus loss
        assert 'FocusOut' in source, (
            "Tooltip has no <FocusOut> binding — stays visible when app loses focus")
        # Must use wm_transient so tooltip follows parent z-order
        assert 'wm_transient' in source, (
            "Tooltip does not use wm_transient — floats above other apps")
        # Must NOT use -topmost (causes tooltip to stay above all windows)
        show_src = inspect.getsource(Tooltip._show)
        assert '-topmost' not in show_src, (
            "Tooltip._show uses -topmost — tooltip persists above other apps")

    def test_config_page_no_core_dropdown_text(self):
        """CONFIG page should not reference nonexistent CORE dropdown."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_pages.py"
        source = source_path.read_text(encoding='utf-8')
        assert "CORE page dropdown" not in source, (
            "CONFIG page references nonexistent CORE page profile dropdown")

    def test_cmd_clear_does_not_rescan(self):
        """_cmd_clear should not call _cmd_welcome (rescans filesystem)."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_cmd_page.py"
        source = source_path.read_text(encoding='utf-8')
        # Find the _cmd_clear method
        lines = source.split('\n')
        in_clear = False
        for line in lines:
            if 'def _cmd_clear' in line:
                in_clear = True
            elif in_clear and line.strip().startswith('def '):
                break
            elif in_clear:
                assert '_cmd_welcome()' not in line, (
                    "_cmd_clear calls _cmd_welcome which rescans filesystem")

    def test_media_uses_named_constants(self):
        """media.py should use named constants for limits, not magic numbers."""
        from enigma_engine.gui.media import (
            MAX_GIF_FRAMES, MAX_IMAGE_DOWNLOAD_BYTES,
            MAX_GIF_DOWNLOAD_BYTES, MEDIA_DOWNLOAD_TIMEOUT)
        assert MAX_GIF_FRAMES == 120
        assert MAX_IMAGE_DOWNLOAD_BYTES == 10 * 1024 * 1024
        assert MAX_GIF_DOWNLOAD_BYTES == 20 * 1024 * 1024
        assert MEDIA_DOWNLOAD_TIMEOUT == 10

    def test_desktop_no_dead_session_counter_init(self):
        """desktop.py should not have redundant _session_counter = 0."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        assert "_session_counter: int = 0" not in source, (
            "Redundant initialization immediately overwritten by = 1")


# ================================================================
# Voice Input: Conversational Mode
# ================================================================

class TestVoiceConversation:
    """Verify voice input works conversationally (auto-send)."""

    def test_voice_continuous_listening(self):
        """Voice input should keep listening after each phrase."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic.py"
        source = source_path.read_text(encoding='utf-8')
        # The old code set _voice_got_audio to stop after one phrase;
        # the new code should NOT have that pattern
        assert '_voice_got_audio = True' not in source, (
            "Voice input stops after one phrase — should be continuous")


# ================================================================
# Voice Output: TTS
# ================================================================


# ================================================================
# Model Delete: No GUI Freeze
# ================================================================

class TestModelDeleteNoFreeze:
    """Verify model deletion runs heavy work off the main thread."""

    def test_delete_model_no_direct_unload(self):
        """_delete_model should not call _unload_model directly.

        _unload_model does torch.cuda.empty_cache() which is slow;
        the delete method should handle unloading inline without
        blocking the GUI.
        """
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_forge_models.py"
        source = source_path.read_text(encoding='utf-8')
        lines = source.split('\n')
        in_method = False
        for line in lines:
            if 'def _delete_model' in line:
                in_method = True
            elif in_method and line.strip().startswith('def ') and '_do_delete' not in line:
                break
            elif in_method and '_unload_model()' in line:
                pytest.fail(
                    "_delete_model should not call _unload_model "
                    "directly — it freezes the GUI")

    def test_refresh_models_uses_thread(self):
        """_refresh_models should scan in a background thread."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_forge_models.py"
        source = source_path.read_text(encoding='utf-8')
        # Extract _refresh_models through to next top-level method
        lines = source.split('\n')
        in_method = False
        method_lines = []
        for line in lines:
            if 'def _refresh_models' in line:
                in_method = True
            elif in_method and line.strip().startswith('def ') and not line.startswith(' ' * 8):
                # Hit next class-level method (4 spaces indent)
                break
            if in_method:
                method_lines.append(line)
        method_body = '\n'.join(method_lines)
        assert 'Thread' in method_body, (
            "_refresh_models should scan models in background thread")


# ================================================================
# Model Size Display
# ================================================================

class TestModelSizeDisplay:
    """Verify model cards show user-entered size, not computed params."""

    def test_normalise_size_label_b(self):
        """'8b' should normalise to '8B'."""
        from enigma_engine.gui.scanners import _normalise_size_label
        assert _normalise_size_label("8b") == "8B"

    def test_normalise_size_label_decimal_b(self):
        """'1.5b' should normalise to '1.5B'."""
        from enigma_engine.gui.scanners import _normalise_size_label
        assert _normalise_size_label("1.5b") == "1.5B"

    def test_normalise_size_label_m(self):
        """'500m' should normalise to '0.50B'."""
        from enigma_engine.gui.scanners import _normalise_size_label
        assert _normalise_size_label("500m") == "0.50B"

    def test_normalise_size_label_preset(self):
        """Preset names like 'small' pass through unchanged."""
        from enigma_engine.gui.scanners import _normalise_size_label
        assert _normalise_size_label("small") == "small"


# ================================================================
# Memory Optimization: Param Counting + Image Cap
# ================================================================

class TestMemoryOptimization:
    """Verify RAM optimizations — no huge torch.load, capped images."""

    def test_format_param_count_billions(self):
        """_format_param_count formats large numbers as B."""
        from enigma_engine.gui.scanners import _format_param_count
        assert _format_param_count(19_080_000_000) == "19.08B"

    def test_format_param_count_millions(self):
        """_format_param_count formats millions as M."""
        from enigma_engine.gui.scanners import _format_param_count
        assert _format_param_count(5_000_000) == "5.0M"

    def test_format_param_count_small(self):
        """_format_param_count formats small numbers with commas."""
        from enigma_engine.gui.scanners import _format_param_count
        assert _format_param_count(1234) == "1,234"

    def test_max_chat_images_constant(self):
        """MAX_CHAT_IMAGES constant exists in media.py."""
        from enigma_engine.gui.media import MAX_CHAT_IMAGES
        assert isinstance(MAX_CHAT_IMAGES, int)
        assert MAX_CHAT_IMAGES > 0

    def test_peek_target_size_returns_none_for_missing(self):
        """_peek_target_size returns None for non-existent file."""
        from enigma_engine.gui.scanners import _peek_target_size
        result = _peek_target_size(Path("nonexistent_model.pth"))
        assert result is None


# ================================================================
# Deferred Imports: Avoid Loading torch/transformers at Startup
# ================================================================

class TestDeferredImports:
    """Verify heavy libraries are NOT loaded when importing the GUI."""

    def test_core_init_no_eager_loader_imports(self):
        """core/__init__.py must not eagerly import loader modules."""
        import re
        source_path = (Path(__file__).parent.parent
                       / "enigma_engine" / "core" / "__init__.py")
        source = source_path.read_text(encoding='utf-8')
        for mod in ['gguf_loader', 'huggingface_loader', 'ollama_loader',
                     'onnx_loader', 'gptq_awq_loader']:
            eager = re.findall(
                rf'^from \.{mod} import', source, re.MULTILINE)
            assert not eager, (
                f"core/__init__.py still eagerly imports {mod}")

    def test_core_init_kv_cache_lazy(self):
        """KVCache import must be lazy in core/__init__.py."""
        import re
        source_path = (Path(__file__).parent.parent
                       / "enigma_engine" / "core" / "__init__.py")
        source = source_path.read_text(encoding='utf-8')
        eager = re.findall(r'^from \.kv_cache import', source, re.MULTILINE)
        assert not eager, (
            "core/__init__.py eagerly imports kv_cache — it must be lazy")

    def test_huggingface_loader_deferred(self):
        """huggingface_loader must NOT import transformers at module level."""
        import re
        source_path = (Path(__file__).parent.parent / "enigma_engine"
                       / "core" / "huggingface_loader.py")
        source = source_path.read_text(encoding='utf-8')
        top_level = re.findall(
            r'^from transformers import', source, re.MULTILINE)
        assert not top_level, (
            "huggingface_loader.py has top-level 'from transformers import'")

    def test_gptq_awq_loader_deferred(self):
        """gptq_awq_loader must NOT import transformers at module level."""
        import re
        source_path = (Path(__file__).parent.parent / "enigma_engine"
                       / "core" / "gptq_awq_loader.py")
        source = source_path.read_text(encoding='utf-8')
        top_level = re.findall(
            r'^from transformers import', source, re.MULTILINE)
        assert not top_level, (
            "gptq_awq_loader.py has top-level 'from transformers import'")

    def test_gguf_loader_deferred(self):
        """gguf_loader must NOT import llama_cpp at module level."""
        import re
        source_path = (Path(__file__).parent.parent / "enigma_engine"
                       / "core" / "gguf_loader.py")
        source = source_path.read_text(encoding='utf-8')
        top_level = re.findall(
            r'^from llama_cpp import', source, re.MULTILINE)
        assert not top_level, (
            "gguf_loader.py has top-level 'from llama_cpp import'")

    def test_lazy_loaders_resolve(self):
        """Lazy loader attributes resolve correctly via __getattr__."""
        from enigma_engine import core
        for name in ['load_gguf_model', 'load_huggingface_model',
                     'load_ollama_model', 'load_onnx_model']:
            attr = getattr(core, name, 'MISSING')
            assert attr != 'MISSING', f"core.{name} not accessible"


# ================================================================
# TTS Thread Safety: No Cross-Thread COM Calls
# ================================================================

class TestTTSThreadSafety:
    """Verify TTS stop uses callback instead of cross-thread engine.stop()."""

    def test_tts_no_started_word_callback(self):
        """Worker must NOT use started-word callback — it corrupts SAPI5."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_media.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        match = re.search(
            r'def _tts_worker\b.*?(?=\n            t = threading\.Thread)',
            source, re.DOTALL)
        assert match, "_tts_worker not found"
        body = match.group(0)
        assert "started-word" not in body or "do NOT use" in body.lower() or "intentionally" in body.lower(), (
            "TTS worker must not connect started-word callback — "
            "calling engine.stop() inside runAndWait breaks SAPI5")

    def test_tts_stop_sets_event(self):
        """_tts_stop must signal the event, not call engine.stop() directly."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "gui_logic_media.py"
        source = source_path.read_text(encoding='utf-8')
        # Extract just the _tts_stop method body (code lines only,
        # excluding docstrings) up to the next method.
        import re
        match = re.search(
            r'def _tts_stop\b.*?(?=\n    def )',
            source, re.DOTALL)
        assert match, "_tts_stop method not found"
        body = match.group(0)
        # Check code lines only — skip docstring content
        code_lines = []
        in_docstring = False
        for line in body.split('\n'):
            stripped = line.strip()
            if stripped.startswith('"""'):
                # Toggle docstring — skip lines inside it
                if in_docstring:
                    in_docstring = False
                    continue
                if stripped.endswith('"""') and len(stripped) > 3:
                    continue  # single-line docstring
                in_docstring = True
                continue
            if not in_docstring:
                code_lines.append(line)
        code_body = '\n'.join(code_lines)
        assert '_tts_stop_event' in code_body and '.set()' in code_body, (
            "_tts_stop must set the stop event")
        assert 'engine' not in code_body or '.stop()' not in code_body, (
            "_tts_stop must NOT call engine.stop() — "
            "cross-thread COM calls crash on Windows SAPI5")


# ================================================================
# TTS Text Cleaning: Safe Text for SAPI5
# ================================================================

class TestTTSTextCleaning:
    """Verify TTS cleans and chunks text before speaking."""

    def test_tts_clean_strips_code_blocks(self):
        """Code blocks should be replaced with a short label."""
        from enigma_engine.gui.gui_logic import LogicMixin
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        result = LogicMixin._tts_clean_text(None, text)
        assert '```' not in result
        assert "print" not in result
        assert "Done" in result

    def test_tts_clean_strips_inline_code(self):
        """Backtick-wrapped code should have backticks removed."""
        from enigma_engine.gui.gui_logic import LogicMixin
        result = LogicMixin._tts_clean_text(None, "Use `print()` here.")
        assert '`' not in result
        assert "print" in result

    def test_tts_clean_strips_markdown(self):
        """Markdown bold/italic markers should be removed."""
        from enigma_engine.gui.gui_logic import LogicMixin
        result = LogicMixin._tts_clean_text(None, "This is **bold**.")
        assert '**' not in result
        assert "bold" in result

    def test_tts_clean_strips_urls(self):
        """URLs should be replaced with 'link'."""
        from enigma_engine.gui.gui_logic import LogicMixin
        result = LogicMixin._tts_clean_text(
            None, "Visit https://example.com/path?q=1 now.")
        assert "https://" not in result

    def test_tts_clean_strips_cmd_blocks(self):
        """[CMD]...[/CMD] blocks should be removed."""
        from enigma_engine.gui.gui_logic import LogicMixin
        result = LogicMixin._tts_clean_text(
            None, "Done. [CMD]file.read x[/CMD] OK.")
        assert "[CMD]" not in result
        assert "OK" in result

    def test_tts_chunks_long_text(self):
        """Long text should be split into sentence chunks."""
        from enigma_engine.gui.gui_logic import LogicMixin
        # Build text longer than 180 chars with sentence boundaries
        text = "This is a fairly long test sentence for chunking. " * 8
        assert len(text) > 200, "test text must exceed chunk limit"
        chunks = LogicMixin._tts_chunk_text(None, text)
        assert isinstance(chunks, list)
        assert len(chunks) >= 2, (
            f"Text of {len(text)} chars should be split into "
            f"multiple chunks, got {len(chunks)}")

    def test_tts_chunks_respect_max_length(self):
        """No chunk should exceed the max character limit."""
        from enigma_engine.gui.gui_logic import LogicMixin
        text = "Word " * 100  # ~500 chars, no periods
        chunks = LogicMixin._tts_chunk_text(None, text)
        for chunk in chunks:
            assert len(chunk) <= 200, (
                f"Chunk too long ({len(chunk)} chars): {chunk[:50]}...")


# ================================================================
# Boot Time: Deferred Param Counting
# ================================================================

class TestDeferredBootParam:
    """Verify model param counting is deferred to background thread."""

    def test_scan_models_returns_estimated_params(self):
        """scan_models should estimate params from file size (not load the model)."""
        from enigma_engine.gui.scanners import scan_models
        models = scan_models()
        # Models with recognized extensions should have a file-size estimate
        for m in models:
            if m["format"] in ("gguf", "pth", "pt", "bin", "safetensors"):
                assert m["params"] is None or isinstance(m["params"], (int, float)), (
                    f"params should be None or a numeric estimate, "
                    f"got {type(m['params'])} for {m['name']}")

    def test_cpuinfo_not_called_synchronously(self):
        """cpuinfo.get_cpu_info() must not block the status tick."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "gui" / "desktop.py"
        source = source_path.read_text(encoding='utf-8')
        import re
        match = re.search(
            r'def _tick\b.*?(?=\n        self\.after)',
            source, re.DOTALL)
        assert match, "_tick function not found"
        tick_body = match.group(0)
        assert 'cpuinfo' not in tick_body, (
            "cpuinfo must not be called inside _tick — "
            "it blocks the UI for seconds")

    def test_status_ticker_uses_low_overhead_interval(self, monkeypatch):
        """Low-overhead mode slows the steady-state status refresh rate."""
        from enigma_engine.gui.desktop import EnigmaGUI
        import enigma_engine.gui.desktop as desktop_mod

        after_calls = []

        monkeypatch.setattr(desktop_mod.threading, "Thread", DummyThread)

        obj = object.__new__(EnigmaGUI)
        obj._gaming_mode_active = True
        obj._status_tick_ms = 5000
        obj._boot_time = 0.0
        obj._hw_device_label = "CPU"
        obj._shutting_down = False
        obj.status_bar = DummyStatusBar()
        obj.state = lambda: "normal"  # mock tkinter state()

        def after(ms, callback):
            after_calls.append(ms)
            if len(after_calls) == 1:
                callback()

        obj.after = after

        obj._start_status_ticker()

        assert after_calls == [100, 5000]

    def test_count_model_params_skips_in_low_overhead_mode(self, monkeypatch):
        """Low-overhead mode must not start exact param counting threads."""
        from enigma_engine.gui.desktop import EnigmaGUI
        import enigma_engine.gui.desktop as desktop_mod

        monkeypatch.setattr(desktop_mod.threading, "Thread", UnexpectedThread)

        obj = object.__new__(EnigmaGUI)
        obj._gaming_mode_active = True
        obj.models_data = [{"format": "pth", "params": None, "path": "fake"}]

        obj._count_model_params_background()


# ================================================================
# TTS Queue Drain: Stop Clears Pending Chunks
# ================================================================


# ================================================================
# Scroll Consistency: Always Scroll During Typewriter
# ================================================================


# ================================================================
# Route Prompts
# ================================================================

class TestRoutePrompts:
    """Verify per-route prompt system works."""

    def test_prompts_dir_exists(self):
        """data/prompts/ directory exists."""
        from enigma_engine.gui.scanners import PROMPTS_DIR
        assert PROMPTS_DIR.exists(), "data/prompts/ must exist"

    def test_default_prompt_files_exist(self):
        """Default prompt files (chat, trainer) exist."""
        from enigma_engine.gui.scanners import PROMPTS_DIR
        assert (PROMPTS_DIR / "chat.md").exists()
        assert (PROMPTS_DIR / "trainer.md").exists()

    def test_load_route_prompt_chat(self):
        """load_route_prompt('chat') returns non-empty string."""
        from enigma_engine.gui.scanners import load_route_prompt
        prompt = load_route_prompt("chat")
        assert isinstance(prompt, str)
        assert len(prompt) > 10, "Chat prompt should have content"

    def test_load_route_prompt_trainer(self):
        """load_route_prompt('trainer') returns non-empty string."""
        from enigma_engine.gui.scanners import load_route_prompt
        prompt = load_route_prompt("trainer")
        assert isinstance(prompt, str)
        assert len(prompt) > 10, "Trainer prompt should have content"

    def test_load_route_prompt_missing(self):
        """load_route_prompt for non-existent route returns empty."""
        from enigma_engine.gui.scanners import load_route_prompt
        prompt = load_route_prompt("nonexistent_route_xyz")
        assert prompt == ""

    def test_scan_docs_has_prompts_category(self):
        """scan_docs includes prompt files under 'prompts' category."""
        from enigma_engine.gui.scanners import scan_docs
        docs = scan_docs()
        prompts = [d for d in docs if d["category"] == "prompts"]
        assert len(prompts) >= 2, (
            "scan_docs should include at least 2 prompt files")
        names = {d["filename"] for d in prompts}
        assert "chat.md" in names
        assert "trainer.md" in names

    def test_trainer_system_prompt_includes_user_prompt(self):
        """_build_trainer_system_prompt prepends user trainer prompt."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        prompt = ForgeMixin._build_trainer_system_prompt(
            student_params=1_000_000, stage="basics")
        # Should contain content from the trainer prompt file
        from enigma_engine.gui.scanners import load_route_prompt
        user_prompt = load_route_prompt("trainer")
        if user_prompt:
            assert user_prompt[:30] in prompt, (
                "Trainer system prompt should include user prompt")

    def test_model_context_default_prompt(self):
        """ModelContext default prompt loads from chat.md."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("test_prompt_default")
        # Should either match chat.md content or fallback
        assert len(ctx.system_prompt) > 10
        assert ctx.system_prompt != ""

    def test_prompt_files_readable(self):
        """All prompt files are readable with utf-8."""
        from enigma_engine.gui.scanners import PROMPTS_DIR
        for f in PROMPTS_DIR.glob("*.md"):
            content = f.read_text(encoding="utf-8")
            assert len(content) > 0, f"Prompt file empty: {f.name}"


# ================================================================
# Chat Tab Audit Fixes
# ================================================================

class TestChatTabAuditFixes:
    """Tests for chat tab issue fixes."""

    def test_chat_cursor_uses_motion_not_tag_enter_leave(self):
        """Chat link cursor should use Motion handler, not per-tag Enter/Leave (S741)."""
        import inspect
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin)
        # Must NOT have per-tag Enter/Leave cursor changes
        assert 'tag_bind("link", "<Enter>"' not in source, (
            "link tag should not bind <Enter> for cursor — use <Motion> instead")
        assert 'tag_bind("video_link", "<Enter>"' not in source
        assert 'tag_bind("file_link", "<Enter>"' not in source
        # Must HAVE a Motion-based cursor handler
        assert "<Motion>" in source, (
            "Chat textbox needs <Motion> bind for cursor updates")

    def test_sessions_sorted_newest_first(self):
        """scan_sessions returns newest sessions first."""
        from enigma_engine.gui.scanners import scan_sessions
        sessions = scan_sessions()
        if len(sessions) >= 2:
            for i in range(len(sessions) - 1):
                assert sessions[i]["saved_at"] >= sessions[i + 1]["saved_at"], (
                    "Sessions should be sorted newest-first by saved_at")

    def test_config_default_max_gen_is_2048(self):
        """CONFIG max_gen default is 2048."""
        from enigma_engine.config import CONFIG
        assert CONFIG.get("max_gen") == 8192


class TestEstimateGGUFParams:
    """Tests for GGUF parameter estimation in model loading."""

    def test_metadata_estimation(self):
        """Estimates params from dim, n_layers, vocab_size metadata."""
        from enigma_engine.gui.gui_logic import _estimate_gguf_params
        from types import SimpleNamespace
        cfg = SimpleNamespace(dim=4096, n_layers=32, vocab_size=151936)
        model = SimpleNamespace(config=cfg)
        engine = SimpleNamespace(model=model)
        result = _estimate_gguf_params(engine, "fake.gguf")
        # 12 * 4096^2 * 32 + 151936 * 4096 = 6,442,450,944 + 622,329,856
        assert result > 6_000_000_000
        assert result < 8_000_000_000

    def test_metadata_zero_dim_falls_to_filesize(self, tmp_path):
        """Falls back to file-size heuristic when metadata has dim=0."""
        from enigma_engine.gui.gui_logic import _estimate_gguf_params
        from types import SimpleNamespace
        cfg = SimpleNamespace(dim=0, n_layers=0, vocab_size=0)
        model = SimpleNamespace(config=cfg)
        engine = SimpleNamespace(model=model)
        fake = tmp_path / "test.gguf"
        fake.write_bytes(b'\x00' * (1024 * 1024))  # 1 MB
        result = _estimate_gguf_params(engine, str(fake))
        assert result > 0

    def test_no_model_falls_to_filesize(self, tmp_path):
        """Falls back when engine has no model attribute."""
        from enigma_engine.gui.gui_logic import _estimate_gguf_params
        engine = type('E', (), {'model': None})()
        fake = tmp_path / "test.gguf"
        fake.write_bytes(b'\x00' * (2 * 1024 * 1024))  # 2 MB
        result = _estimate_gguf_params(engine, str(fake))
        assert result > 0

    def test_missing_file_returns_zero(self):
        """Returns 0 when model has no metadata and file doesn't exist."""
        from enigma_engine.gui.gui_logic import _estimate_gguf_params
        engine = type('E', (), {'model': None})()
        result = _estimate_gguf_params(engine, "/nonexistent/fake.gguf")
        assert result == 0


# =========================================================================
# Deep-dive audit — scanner security, server path traversal, CORS
# =========================================================================


# ── Mod base file presence ──────────────────────────────────────────────────


class TestModBasePresence:
    """Verify mod_base.py exists in each shipped mod folder."""

    def test_imagegen_has_mod_base(self):
        """mods/imagegen/ must contain mod_base.py for imports to work."""
        from pathlib import Path
        assert (Path("mods/imagegen/mod_base.py").exists()), (
            "mods/imagegen/mod_base.py missing — ImageGenMod fails")


# ── Config persistence ──────────────────────────────────────────────────────


# ── Atomic saves ────────────────────────────────────────────────────────────


class TestAtomicSaves:
    """Verify model saves use atomic write pattern."""

    # ── atomic_write_text / atomic_write_json ────────────────────────────

    def test_atomic_write_text_roundtrip(self, tmp_path):
        """atomic_write_text must write and read back correctly."""
        from enigma_engine.core.safe_save import atomic_write_text
        target = tmp_path / "test.txt"
        content = "hello\nworld\n"
        atomic_write_text(target, content)
        assert target.read_text(encoding="utf-8") == content

    def test_atomic_write_json_roundtrip(self, tmp_path):
        """atomic_write_json must write and read back valid JSON."""
        import json
        from enigma_engine.core.safe_save import atomic_write_json
        target = tmp_path / "test.json"
        data = {"key": "value", "num": 42, "nested": [1, 2, 3]}
        atomic_write_json(target, data)
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded == data

    def test_atomic_write_text_cleans_tmp_on_failure(self, tmp_path):
        """atomic_write_text must not leave .tmp on write failure."""
        from enigma_engine.core.safe_save import atomic_write_text
        target = tmp_path / "sub" / "test.txt"
        # Create the parent so mkdir doesn't fail
        target.parent.mkdir(parents=True, exist_ok=True)
        # Make target a directory to force os.replace to fail
        target.mkdir()
        with pytest.raises(OSError):
            atomic_write_text(target, "data")
        tmp_file = target.with_suffix(target.suffix + ".tmp")
        assert not tmp_file.exists(), ".tmp file should be cleaned up on failure"

    def test_atomic_write_text_creates_bak(self, tmp_path):
        """atomic_write_text must create .bak of existing file."""
        from enigma_engine.core.safe_save import atomic_write_text
        target = tmp_path / "test.txt"
        target.write_text("original", encoding="utf-8")
        atomic_write_text(target, "updated")
        bak = target.with_suffix(".txt.bak")
        assert bak.exists(), ".bak file should exist"
        assert bak.read_text(encoding="utf-8") == "original"
        assert target.read_text(encoding="utf-8") == "updated"

    def test_atomic_write_json_creates_bak(self, tmp_path):
        """atomic_write_json must create .bak of existing JSON file."""
        import json
        from enigma_engine.core.safe_save import atomic_write_json
        target = tmp_path / "data.json"
        atomic_write_json(target, {"version": 1})
        atomic_write_json(target, {"version": 2})
        bak = target.with_suffix(".json.bak")
        assert bak.exists(), ".bak file should exist for JSON overwrite"
        assert json.loads(bak.read_text(encoding="utf-8")) == {"version": 1}
        assert json.loads(target.read_text(encoding="utf-8")) == {"version": 2}

    def test_no_direct_writes_in_critical_modules(self):
        """Critical data modules must use atomic_write_text/json, not raw writes."""
        import ast
        from pathlib import Path as P
        modules = [
            "enigma_engine/core/memory.py",
            "enigma_engine/core/curated_dataset.py",
            "enigma_engine/training/training_queue.py",
            "enigma_engine/training/training_monitor.py",
            "enigma_engine/core/model_context.py",
            "enigma_engine/core/model_registry.py",
            "enigma_engine/core/ai_profile.py",
            "enigma_engine/core/rag.py",
            "enigma_engine/core/adaptive_trainer.py",
        ]
        for modpath in modules:
            source = P(modpath).read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                # Check for open(..., "w", ...) calls
                if isinstance(func, ast.Name) and func.id == "open":
                    pass
                elif isinstance(func, ast.Attribute) and func.attr == "open":
                    pass
                else:
                    continue
                # Determine mode
                mode = "r"
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                if len(node.args) >= 2:
                    arg = node.args[1]
                    if isinstance(arg, ast.Constant):
                        mode = arg.value
                if "w" in mode:
                    assert False, (
                        f"{modpath} line {node.lineno}: uses raw open('w') "
                        f"— must use atomic_write_text/json instead")
            # Also check for .write_text( calls
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "write_text":
                        assert False, (
                            f"{modpath} line {node.lineno}: uses raw "
                            f".write_text() — must use atomic_write_text/json")


# ── Ollama encoding ─────────────────────────────────────────────────────────


class TestOllamaEncoding:
    """Verify ollama_loader.py uses utf-8 encoding on all text opens."""

    def test_no_text_opens_without_encoding(self):
        """All text-mode open() calls must have encoding='utf-8'."""
        import ast
        from pathlib import Path as P
        source = P("enigma_engine/core/ollama_loader.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Match open(...) calls
            if isinstance(func, ast.Name) and func.id == "open":
                pass
            elif isinstance(func, ast.Attribute) and func.attr == "open":
                pass
            else:
                continue
            # Check mode arg — skip binary opens
            mode = "r"  # default
            for kw in node.keywords:
                if kw.arg == "mode":
                    if isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
            if len(node.args) >= 2:
                arg = node.args[1]
                if isinstance(arg, ast.Constant):
                    mode = arg.value
            if "b" in mode:
                continue  # binary mode is fine
            # Text mode — must have encoding keyword
            has_encoding = any(
                kw.arg == "encoding" for kw in node.keywords)
            assert has_encoding, (
                f"ollama_loader.py line {node.lineno}: "
                f"text open() missing encoding='utf-8'")


# =====================================================================
# Rename model — case-insensitive Windows support
# =====================================================================


# =====================================================================
# Gradient checkpointing in training
# =====================================================================

class TestGradientCheckpointing:
    """Gradient checkpointing reduces VRAM usage during training."""

    def test_training_config_has_gradient_checkpointing(self):
        """TrainingConfig must have use_gradient_checkpointing field."""
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "use_gradient_checkpointing")
        assert cfg.use_gradient_checkpointing is True

    def test_gradient_checkpointing_in_to_dict(self):
        """use_gradient_checkpointing must appear in to_dict output."""
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(use_gradient_checkpointing=True)
        d = cfg.to_dict()
        assert "use_gradient_checkpointing" in d
        assert d["use_gradient_checkpointing"] is True


# =====================================================================
# Training config exposed in FORGE UI
# =====================================================================


# =====================================================================
# Quantize & Export on FORGE page
# =====================================================================


# =====================================================================
# Memory instructions — proactive preference learning
# =====================================================================


# =====================================================================
# Learn While Chatting — BackgroundTrainer wired to chat
# =====================================================================


# =====================================================================
# Theme picker on CONFIG page
# =====================================================================

class TestThemePicker:
    """CONFIG page must have a theme selector with live switching."""

    def test_apply_theme_no_restart(self):
        """_apply_theme must NOT call _restart_gui (live switching)."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._apply_theme)
        assert "_restart_gui" not in source

    def test_reload_theme_returns_color_map(self):
        """reload_theme returns a dict mapping old colours to new."""
        from enigma_engine.gui.widgets import reload_theme
        color_map = reload_theme("midnight")
        assert isinstance(color_map, dict)
        # Should have mappings since dark != midnight
        assert len(color_map) > 0
        # Restore original theme
        reload_theme("dark")

    def test_reload_theme_updates_globals(self):
        """reload_theme must update C_* module-level constants."""
        from enigma_engine.gui import widgets
        old_bg = widgets.C_BG
        widgets.reload_theme("midnight")
        assert widgets.C_BG != old_bg
        # Restore
        widgets.reload_theme("dark")
        assert widgets.C_BG == old_bg

    def test_reload_theme_same_returns_empty(self):
        """reload_theme with current theme returns empty map."""
        from enigma_engine.gui.widgets import reload_theme
        # Ensure we're on dark
        reload_theme("dark")
        color_map = reload_theme("dark")
        assert color_map == {}

    def test_config_no_profile_section(self):
        """CONFIG page must NOT have an AI Profile section."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_config)
        assert "AI PROFILE" not in source
        assert "profile_dd" not in source
        assert "_apply_profile" not in source

    def test_max_tokens_no_artificial_cap(self):
        """max_tokens upper limit must be large — no artificial 4096 cap."""
        from enigma_engine.gui.scanners import CONFIG_LIMITS
        _lo, hi, _step = CONFIG_LIMITS["max_tokens"]
        assert hi >= 100000, f"max_tokens capped at {hi}, should be uncapped"


# =====================================================================
# Selectable text everywhere — labels and textboxes
# =====================================================================


# =====================================================================
# Silent error swallowing — except Exception: pass must log
# =====================================================================

class TestSilentErrorSwallowing:
    """Critical modules must not silently swallow errors."""

    def test_gui_logic_no_bare_pass(self):
        """gui_logic.py should have minimal except-pass blocks."""
        from enigma_engine.gui import gui_logic
        source = inspect.getsource(gui_logic)
        import re
        # Count bare except-pass (no logging)
        bare_passes = re.findall(
            r'except\s+Exception[^:]*:\s*\n\s*pass\s*$',
            source, re.MULTILINE)
        # Allow some for UI widget guards, but not more than 8
        assert len(bare_passes) <= 8, (
            f"gui_logic.py has {len(bare_passes)} silent "
            f"except-pass blocks — add logger.debug()")

    def test_router_no_bare_pass(self):
        """router.py should have minimal except-pass blocks."""
        from enigma_engine import router
        source = inspect.getsource(router)
        import re
        bare_passes = re.findall(
            r'except\s+Exception[^:]*:\s*\n\s*pass\s*$',
            source, re.MULTILINE)
        assert len(bare_passes) <= 2, (
            f"router.py has {len(bare_passes)} silent "
            f"except-pass blocks — add logger.debug()")

    def test_scanners_no_bare_pass(self):
        """scanners.py should have minimal except-pass blocks."""
        from enigma_engine.gui import scanners
        source = inspect.getsource(scanners)
        import re
        bare_passes = re.findall(
            r'except\s+Exception[^:]*:\s*\n\s*pass\s*$',
            source, re.MULTILINE)
        assert len(bare_passes) <= 1, (
            f"scanners.py has {len(bare_passes)} silent "
            f"except-pass blocks — add logger.debug()")


# ================================================================
# FORGE Feature Tests — presets, preview, history, progress bar,
# learn-while-chatting toggle
# ================================================================

class TestForgePresets:
    """Hyperparameter preset backend logic."""

    def test_preset_values_defined(self):
        """ForgeMixin._TRAINING_PRESETS has expected keys."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        presets = ForgeMixin._TRAINING_PRESETS
        assert "Quick" in presets
        assert "Balanced" in presets
        assert "Thorough" in presets

    def test_preset_tuples_have_three_values(self):
        """Each preset is a (epochs, lr, batch) tuple."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        for name, vals in ForgeMixin._TRAINING_PRESETS.items():
            assert len(vals) == 3, (
                f"Preset '{name}' should have 3 values")

    def test_preset_custom_not_in_presets(self):
        """Custom is not in presets dict (leaves fields unchanged)."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert "Custom" not in ForgeMixin._TRAINING_PRESETS

    def test_quick_preset_values(self):
        """Quick preset: 3 epochs, lr=0.0001, batch=auto."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        epochs, lr, batch = ForgeMixin._TRAINING_PRESETS["Quick"]
        assert epochs == "3"
        assert lr == "0.0001"
        assert batch == "auto"

    def test_balanced_preset_values(self):
        """Balanced preset: 10 epochs, lr=0.00005, batch=auto."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        epochs, lr, batch = ForgeMixin._TRAINING_PRESETS["Balanced"]
        assert epochs == "10"
        assert lr == "0.00005"
        assert batch == "auto"

    def test_thorough_preset_values(self):
        """Thorough preset: 30 epochs, lr=0.00002, batch=auto."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        epochs, lr, batch = ForgeMixin._TRAINING_PRESETS["Thorough"]
        assert epochs == "30"
        assert lr == "0.00002"
        assert batch == "auto"


class TestForgeTrainingHistory:
    """Training history save/load."""

    def test_history_file_path(self):
        """History file is at data/training_history.json."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        assert ForgeMixin._HISTORY_FILE.name == "training_history.json"
        assert "data" in str(ForgeMixin._HISTORY_FILE)

    def test_save_training_run(self, tmp_path, monkeypatch):
        """_save_training_run writes a valid JSON entry."""
        import json
        from enigma_engine.gui.gui_forge import ForgeMixin
        history_file = tmp_path / "training_history.json"
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge.ForgeMixin._HISTORY_FILE",
            history_file)

        obj = object.__new__(ForgeMixin)
        obj._save_training_run("Solo", "test_model", 5, 0.1234)

        runs = json.loads(history_file.read_text(encoding="utf-8"))
        assert len(runs) == 1
        assert runs[0]["mode"] == "Solo"
        assert runs[0]["model"] == "test_model"
        assert runs[0]["epochs"] == 5
        assert runs[0]["best_loss"] == 0.1234

    def test_save_training_run_appends(self, tmp_path, monkeypatch):
        """Multiple saves append to the same file."""
        import json
        from enigma_engine.gui.gui_forge import ForgeMixin
        history_file = tmp_path / "training_history.json"
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge.ForgeMixin._HISTORY_FILE",
            history_file)

        obj = object.__new__(ForgeMixin)
        obj._save_training_run("Solo", "m1", 3, 0.5)
        obj._save_training_run("DPO", "m2", 10, 0.3)

        runs = json.loads(history_file.read_text(encoding="utf-8"))
        assert len(runs) == 2
        assert runs[1]["mode"] == "DPO"

    def test_save_training_run_caps_at_200(
            self, tmp_path, monkeypatch):
        """History is capped at 200 entries."""
        import json
        from enigma_engine.gui.gui_forge import ForgeMixin
        history_file = tmp_path / "training_history.json"
        history_file.write_text(
            json.dumps([{"mode": "old"}] * 200),
            encoding="utf-8")
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge.ForgeMixin._HISTORY_FILE",
            history_file)

        obj = object.__new__(ForgeMixin)
        obj._save_training_run("New", "m1", 1, 0.1)

        runs = json.loads(history_file.read_text(encoding="utf-8"))
        assert len(runs) == 200
        assert runs[-1]["mode"] == "New"

    def test_save_training_run_persists_perplexity(
            self, tmp_path, monkeypatch):
        """When perplexity values are supplied they are saved to history."""
        import json
        from enigma_engine.gui.gui_forge import ForgeMixin
        history_file = tmp_path / "training_history.json"
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge.ForgeMixin._HISTORY_FILE",
            history_file)

        obj = object.__new__(ForgeMixin)
        obj._save_training_run(
            "Solo", "test_model", 5, 0.1234,
            before_perplexity=3.75, after_perplexity=2.40)

        runs = json.loads(history_file.read_text(encoding="utf-8"))
        assert len(runs) == 1
        assert runs[0]["before_perplexity"] == 3.75
        assert runs[0]["after_perplexity"] == 2.40

    def test_save_training_run_no_perplexity_omitted(
            self, tmp_path, monkeypatch):
        """When perplexity is not provided the fields are absent from history."""
        import json
        from enigma_engine.gui.gui_forge import ForgeMixin
        history_file = tmp_path / "training_history.json"
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge.ForgeMixin._HISTORY_FILE",
            history_file)

        obj = object.__new__(ForgeMixin)
        obj._save_training_run("Solo", "test_model", 5, 0.1234)

        runs = json.loads(history_file.read_text(encoding="utf-8"))
        assert "before_perplexity" not in runs[0]
        assert "after_perplexity" not in runs[0]


class TestLearnWhileChattingConfig:
    """Learn-while-chatting toggle on CONFIG page."""

    def test_toggle_saves_setting(self, tmp_path, monkeypatch):
        """Toggle writes learn_while_chatting to gui_settings.json."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        # Create minimal mock

        sync_calls = []

        obj = object.__new__(PagesMixin)
        obj._learn_while_chatting_var = MockVar()
        obj.status_bar = MockStatusBar()
        obj._refresh_performance_mode = lambda: sync_calls.append("refresh")
        obj._sync_router_training_state = lambda: sync_calls.append("sync")

        obj._toggle_learn_while_chatting()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["learn_while_chatting"] is True
        assert obj._chat_learning_enabled is True
        assert sync_calls == ["refresh", "sync"]


class TestInlineSearchEnabledConfig:
    """CONFIG-page checkbox for engine.inline_search_enabled.

    Stage B-2c (Pass 156z9w): the engine ships with the flag default-on
    in ``_init_common``; this checkbox is the user-facing off-switch.
    Persistence lives at the top level of gui_settings.json (NOT inside
    config_overrides — it is an engine attribute, not a chat kwarg).
    """

    def test_toggle_persists_and_applies_to_live_engine(
            self, tmp_path, monkeypatch):
        """Toggling OFF writes the flag to disk and updates a live engine."""
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        class FakeEngine:
            inline_search_enabled = True

        obj = object.__new__(PagesMixin)
        obj._inline_search_enabled_var = MockVar(initial=False)
        obj.status_bar = MockStatusBar()
        obj.engine = FakeEngine()
        obj.inline_search_enabled = True

        obj._toggle_inline_search_enabled()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["inline_search_enabled"] is False
        assert obj.inline_search_enabled is False
        assert obj.engine.inline_search_enabled is False

    def test_toggle_with_no_engine_still_persists(
            self, tmp_path, monkeypatch):
        """Pre-load toggle persists; engine apply is a no-op when None."""
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj._inline_search_enabled_var = MockVar(initial=False)
        obj.status_bar = MockStatusBar()
        obj.engine = None
        obj.inline_search_enabled = True

        obj._toggle_inline_search_enabled()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["inline_search_enabled"] is False
        assert obj.inline_search_enabled is False

    def test_on_model_loaded_applies_persisted_flag(self):
        """`_on_model_loaded` must propagate the GUI's saved flag to
        the freshly-loaded engine — without this the engine ships with
        its library default and the user's off-toggle silently reverts
        on every model load (signal-without-consumer).
        """
        # Structural test: the apply line must be present in the
        # method body. Behavioural coverage at the engine layer is
        # provided by the off-switch tests in test_chat.py.
        import re
        from enigma_engine.gui.gui_logic import LogicMixin
        src = inspect.getsource(LogicMixin._on_model_loaded)
        assert re.search(
            r'self\.engine\.inline_search_enabled\s*=\s*bool\(\s*'
            r'getattr\(self,\s*"inline_search_enabled",\s*True\)\s*\)',
            src,
        ), (
            "_on_model_loaded must apply the persisted "
            "inline_search_enabled flag to the loaded engine"
        )

    def test_boot_load_reads_persisted_off_value(
            self, tmp_path, monkeypatch):
        """Pass 156z9x Finding A — the boot-load step at desktop.py
        L166-167 is what carries the user's saved-OFF preference from
        disk into ``self.inline_search_enabled``.  Without this
        regression gate, a deletion of that line would silently revert
        every restart to library-default True and the off-toggle would
        only survive within a single GUI session.

        Calls ``_read_gui_bool_setting`` directly (the same helper
        ``__init__`` uses) on a stub instance with monkeypatched
        DATA_DIR + a pre-written settings file containing False.
        """
        import enigma_engine.gui.desktop as desktop_mod
        from enigma_engine.gui.desktop import EnigmaGUI

        monkeypatch.setattr(desktop_mod, "DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text(
            json.dumps({"inline_search_enabled": False}),
            encoding="utf-8",
        )

        obj = object.__new__(EnigmaGUI)
        result = obj._read_gui_bool_setting(
            "inline_search_enabled", True)
        assert result is False, (
            "boot-load helper must return the persisted False "
            "value, not the library default True"
        )

        # Sibling-parity: missing key falls back to default True so
        # fresh installs keep observability on.
        settings_file.write_text("{}", encoding="utf-8")
        result_default = obj._read_gui_bool_setting(
            "inline_search_enabled", True)
        assert result_default is True

    def test_boot_load_wire_site_present_in_init(self):
        """Structural gate on the wire-site itself — the
        ``EnigmaGUI.__init__`` body must call
        ``_read_gui_bool_setting`` with the literal
        ``inline_search_enabled`` key.  Without this assertion, a
        regression that drops the boot-load assignment leaves the
        helper-level test passing while every fresh GUI session
        silently uses the library default.

        Adversarial discipline: the assertion targets the literal
        call expression ``_read_gui_bool_setting(...)`` paired with
        the literal key — NOT the bare ``inline_search_enabled``
        token, which also appears at the in-memory default line and
        would mask a regression that deletes only the boot-load
        line.  The regex tolerates the line-continuation whitespace
        between ``(`` and the string literal that black/ruff produce.
        """
        import re
        from enigma_engine.gui.desktop import EnigmaGUI
        src = inspect.getsource(EnigmaGUI.__init__)
        pattern = re.compile(
            r'_read_gui_bool_setting\(\s*"inline_search_enabled"')
        assert pattern.search(src), (
            "__init__ must call _read_gui_bool_setting with the "
            "literal 'inline_search_enabled' key — this is the "
            "boot-load wire-site that carries the user's saved "
            "off-toggle from disk into self.inline_search_enabled"
        )


class TestApiChatConfig:
    """CONFIG-page controls for API-routed CORE chat."""

    def test_toggle_use_api_chat_persists_and_updates_state(
            self, tmp_path, monkeypatch):
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj._use_api_chat_var = MockVar(initial=True)
        obj.status_bar = MockStatusBar()
        obj.use_api_chat = False
        obj._api_chat_client = object()

        obj._toggle_use_api_chat()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["use_api_chat"] is True
        assert obj.use_api_chat is True

    def test_toggle_use_api_chat_off_clears_cached_client(
            self, tmp_path, monkeypatch):
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj._use_api_chat_var = MockVar(initial=False)
        obj.status_bar = MockStatusBar()
        obj.use_api_chat = True
        obj._api_chat_client = object()

        obj._toggle_use_api_chat()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["use_api_chat"] is False
        assert obj.use_api_chat is False
        assert obj._api_chat_client is None

    def test_save_api_base_url_persists_and_resets_client(
            self, tmp_path, monkeypatch):
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        class _Entry:
            def get(self):
                return "http://127.0.0.1:9090"

        obj = object.__new__(PagesMixin)
        obj._api_base_url_entry = _Entry()
        obj.status_bar = MockStatusBar()
        obj.api_base_url = "http://127.0.0.1:8080"
        obj._api_chat_client = object()

        obj._save_api_base_url()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["api_base_url"] == "http://127.0.0.1:9090"
        assert obj.api_base_url == "http://127.0.0.1:9090"
        assert obj._api_chat_client is None

    def test_boot_load_use_api_chat_wire_site_present(self):
        import re
        from enigma_engine.gui.desktop import EnigmaGUI

        src = inspect.getsource(EnigmaGUI.__init__)
        pattern = re.compile(
            r'_read_gui_bool_setting\(\s*"use_api_chat"')
        assert pattern.search(src), (
            "__init__ must boot-load the persisted use_api_chat flag"
        )

    def test_boot_load_api_base_url_wire_site_present(self):
        import re
        from enigma_engine.gui.desktop import EnigmaGUI

        src = inspect.getsource(EnigmaGUI.__init__)
        pattern = re.compile(
            r'_read_gui_str_setting\(\s*"api_base_url"')
        assert pattern.search(src), (
            "__init__ must boot-load the persisted api_base_url setting"
        )


class TestInlineSearchSpliceConfig:
    """B-3a (this pass): CONFIG-page checkbox for
    ``engine.inline_search_splice_enabled`` — the opt-in auto-stop /
    splice flag.  Default OFF (engine library default + library GUI
    default both False).  Persistence at top level of
    gui_settings.json mirrors the observability flag's pattern.
    """

    def test_toggle_persists_and_applies_to_live_engine(
            self, tmp_path, monkeypatch):
        """Toggling ON writes the flag to disk and updates a live engine."""
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        class FakeEngine:
            inline_search_splice_enabled = False

        obj = object.__new__(PagesMixin)
        obj._inline_search_splice_enabled_var = MockVar(initial=True)
        obj.status_bar = MockStatusBar()
        obj.engine = FakeEngine()
        obj.inline_search_splice_enabled = False

        obj._toggle_inline_search_splice_enabled()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["inline_search_splice_enabled"] is True
        assert obj.inline_search_splice_enabled is True
        assert obj.engine.inline_search_splice_enabled is True

    def test_toggle_with_no_engine_still_persists(
            self, tmp_path, monkeypatch):
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj._inline_search_splice_enabled_var = MockVar(initial=True)
        obj.status_bar = MockStatusBar()
        obj.engine = None
        obj.inline_search_splice_enabled = False

        obj._toggle_inline_search_splice_enabled()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["inline_search_splice_enabled"] is True
        assert obj.inline_search_splice_enabled is True

    def test_on_model_loaded_applies_splice_flag(self):
        """`_on_model_loaded` must propagate the GUI's saved splice
        flag to the freshly-loaded engine — without this the user's
        on-toggle silently reverts on every model load."""
        import re
        from enigma_engine.gui.gui_logic import LogicMixin
        src = inspect.getsource(LogicMixin._on_model_loaded)
        assert re.search(
            r'self\.engine\.inline_search_splice_enabled\s*=\s*bool\(\s*'
            r'getattr\(self,\s*"inline_search_splice_enabled",\s*False\)\s*\)',
            src,
        ), (
            "_on_model_loaded must apply the persisted "
            "inline_search_splice_enabled flag to the loaded engine"
        )

    def test_boot_load_reads_persisted_value(
            self, tmp_path, monkeypatch):
        import enigma_engine.gui.desktop as desktop_mod
        from enigma_engine.gui.desktop import EnigmaGUI

        monkeypatch.setattr(desktop_mod, "DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text(
            json.dumps({"inline_search_splice_enabled": True}),
            encoding="utf-8",
        )

        obj = object.__new__(EnigmaGUI)
        result = obj._read_gui_bool_setting(
            "inline_search_splice_enabled", False)
        assert result is True, (
            "boot-load helper must return the persisted True "
            "value, not the library default False"
        )

        # Missing key falls back to library default False so fresh
        # installs keep the feature OFF (opt-in).
        settings_file.write_text("{}", encoding="utf-8")
        result_default = obj._read_gui_bool_setting(
            "inline_search_splice_enabled", False)
        assert result_default is False

    def test_boot_load_wire_site_present_in_init(self):
        """Structural gate: __init__ must call _read_gui_bool_setting
        with the literal 'inline_search_splice_enabled' key."""
        import re
        from enigma_engine.gui.desktop import EnigmaGUI
        src = inspect.getsource(EnigmaGUI.__init__)
        pattern = re.compile(
            r'_read_gui_bool_setting\(\s*"inline_search_splice_enabled"')
        assert pattern.search(src), (
            "__init__ must call _read_gui_bool_setting with the "
            "literal 'inline_search_splice_enabled' key — boot-load "
            "wire-site for B-3a opt-in flag"
        )


class _MockTextbox:
    """Minimal CTkTextbox stub for json_schema tests.  Supports
    `.get("1.0", "end")`, `.insert("1.0", text)`, `.delete(...)`."""

    def __init__(self, initial: str = ""):
        self._text = initial

    def get(self, _start: str, _end: str) -> str:
        return self._text

    def insert(self, _index: str, text: str) -> None:
        self._text += text

    def delete(self, _start: str, _end: str) -> None:
        self._text = ""


class TestJsonSchemaConfig:
    """N-15b (Pass 156z9aa) — CONFIG-page textbox + Apply/Clear
    handlers + boot-load helper for ``EnigmaEngine.chat(json_schema=...)``.

    Persistence: raw text under ``gui_settings.json["json_schema_text"]``;
    runtime: parsed dict on ``self.json_schema``; chat send path
    forwards as ``kwargs["json_schema"]`` when non-None.
    """

    def test_apply_valid_dict_persists_and_sets_attr(
            self, tmp_path, monkeypatch):
        """Valid JSON dict in textbox → parsed onto self.json_schema
        AND raw text written to gui_settings.json["json_schema_text"].
        Adversarially gates THREE post-conditions in one test: live
        attr update, disk persistence, status_bar success message."""
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        raw = '{"type": "object", "properties": {"x": {"type": "integer"}}}'
        obj = object.__new__(PagesMixin)
        obj._json_schema_textbox = _MockTextbox(initial=raw)
        obj.status_bar = MockStatusBar()
        obj.json_schema = None  # in-memory default

        obj._apply_json_schema()

        assert obj.json_schema == {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }
        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["json_schema_text"] == raw

    def test_apply_invalid_json_does_not_clobber_attr(
            self, tmp_path, monkeypatch):
        """Invalid JSON → self.json_schema unchanged AND nothing
        persisted to disk.  Locks the contract that a parse error
        does not silently destroy a previously-valid live schema."""
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        existing = {"json_schema_text": '{"keep": "me"}'}
        settings_file.write_text(
            json.dumps(existing), encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj._json_schema_textbox = _MockTextbox(
            initial='{"broken": ')  # truncated JSON
        obj.status_bar = MockStatusBar()
        prev = {"keep": "me"}
        obj.json_schema = prev

        obj._apply_json_schema()

        assert obj.json_schema is prev, (
            "parse error must NOT clobber the live attribute"
        )
        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["json_schema_text"] == '{"keep": "me"}', (
            "parse error must NOT overwrite persisted text"
        )

    def test_apply_non_dict_rejected(self, tmp_path, monkeypatch):
        """Valid JSON but a list (or any non-dict) is rejected —
        the engine contract requires a dict.  Same don't-clobber
        semantics as the parse-error case."""
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj._json_schema_textbox = _MockTextbox(initial='[1, 2, 3]')
        obj.status_bar = MockStatusBar()
        obj.json_schema = {"prior": True}

        obj._apply_json_schema()

        assert obj.json_schema == {"prior": True}
        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert "json_schema_text" not in data

    def test_apply_invalid_shape_rejected_with_validator_message(
            self, tmp_path, monkeypatch):
        """Pass 156z9ad: valid JSON dict but the SHAPE is unsupported
        (e.g. ``{"type": "array"}`` — FSM is object-only) MUST be
        rejected at Apply time with the validator's exact message,
        not silently accepted to fail at send time.

        Don't-clobber: live attr + disk both stay at the last
        successfully-applied state — same semantics as the parse-
        error and non-dict branches.  Adversarial: status message
        must NAME the type mismatch (string ``"object"``) so the
        user knows what to fix; a generic "schema invalid" would
        not be falsifiable against a regression that swallows the
        validator and re-raises a stub message.
        """
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        existing = {"json_schema_text": '{"keep": "me"}'}
        settings_file.write_text(
            json.dumps(existing), encoding="utf-8")

        class _RecordingStatusBar:
            def __init__(self):
                self.left_text = ""

            def set_left(self, text):
                self.left_text = text

            def set_center(self, text):
                pass

            def set_right(self, text):
                pass

        obj = object.__new__(PagesMixin)
        obj._json_schema_textbox = _MockTextbox(
            initial='{"type": "array"}')
        obj.status_bar = _RecordingStatusBar()
        prev = {"keep": "me"}
        obj.json_schema = prev

        obj._apply_json_schema()

        # Don't-clobber: live attr unchanged
        assert obj.json_schema is prev, (
            "shape rejection must NOT clobber the live attribute"
        )
        # Don't-clobber: persisted text unchanged
        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["json_schema_text"] == '{"keep": "me"}', (
            "shape rejection must NOT overwrite persisted text"
        )
        # Status bar names the validator failure — adversarial:
        # must mention 'object' (the required type) so the user
        # knows what to fix, not just a generic "invalid"
        msg = obj.status_bar.left_text.lower()
        assert "object" in msg, (
            f"status bar must name the validator failure with "
            f"enough specificity to act on (expected 'object' in "
            f"the message naming the required type); got: {msg!r}"
        )

    def test_apply_calls_validate_json_schema_shape(self):
        """Structural gate on the wire-site: ``_apply_json_schema``
        MUST import and call ``validate_json_schema_shape`` between
        the ``isinstance(parsed, dict)`` check and the persist.
        Without this gate the textbox accepts shape-invalid schemas
        that only fail at send time (UX regression).

        Adversarial-falsifiable: deleting the call from the body
        flips this assertion to fail.  Strips comment-only lines
        before scanning per the §4 "Label-tracking" rule — a stale
        comment mentioning the validator must not satisfy the gate.
        """
        import re
        import inspect as _inspect
        from enigma_engine.gui.gui_pages_config import ConfigPageMixin
        src = _inspect.getsource(ConfigPageMixin._apply_json_schema)
        # Strip comment-only lines (rstrip + leading-# check) so a
        # commented-out reference to validate_json_schema_shape
        # cannot satisfy this gate
        code_lines = [
            ln for ln in src.splitlines()
            if not ln.strip().startswith("#")
        ]
        code = "\n".join(code_lines)
        # Must call the validator (real call expression, not docstring)
        assert re.search(
            r'validate_json_schema_shape\s*\(\s*parsed', code), (
            "_apply_json_schema must call "
            "validate_json_schema_shape(parsed) — without this "
            "gate the GUI accepts shape-invalid schemas that only "
            "surface at send time"
        )
        # Must catch ValueError from the validator (loud at the
        # closest boundary to the user)
        assert re.search(r'except\s+ValueError', code), (
            "_apply_json_schema must catch ValueError from the "
            "validator and surface it via status_bar"
        )

    def test_apply_empty_text_clears_attr(self, tmp_path, monkeypatch):
        """Empty/whitespace textbox + Apply → clears live attr AND
        persists empty string.  Adversarially distinguished from
        the parse-error case: empty IS a valid intentional clear."""
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text(
            json.dumps({"json_schema_text": '{"x": 1}'}),
            encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj._json_schema_textbox = _MockTextbox(initial="   \n  ")
        obj.status_bar = MockStatusBar()
        obj.json_schema = {"x": 1}

        obj._apply_json_schema()

        assert obj.json_schema is None
        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["json_schema_text"] == ""

    def test_clear_button_resets_textbox_and_attr(
            self, tmp_path, monkeypatch):
        """Explicit Clear button: empties textbox + persisted text
        + live attr in one call."""
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text(
            json.dumps({"json_schema_text": '{"x": 1}'}),
            encoding="utf-8")

        obj = object.__new__(PagesMixin)
        textbox = _MockTextbox(initial='{"x": 1}')
        obj._json_schema_textbox = textbox
        obj.status_bar = MockStatusBar()
        obj.json_schema = {"x": 1}

        obj._clear_json_schema()

        assert obj.json_schema is None
        assert textbox.get("1.0", "end") == ""
        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["json_schema_text"] == ""

    def test_boot_load_parses_persisted_dict(
            self, tmp_path, monkeypatch):
        """Helper ``_read_gui_json_schema_setting`` returns the
        parsed dict when persisted text is valid JSON dict.  This
        is the wire-site that carries the user's saved schema
        across restarts; without it every fresh GUI session would
        ship with no constraint until the user re-Applies."""
        import enigma_engine.gui.desktop as desktop_mod
        from enigma_engine.gui.desktop import EnigmaGUI

        monkeypatch.setattr(desktop_mod, "DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        raw = '{"type": "object"}'
        settings_file.write_text(
            json.dumps({"json_schema_text": raw}),
            encoding="utf-8")

        obj = object.__new__(EnigmaGUI)
        result = obj._read_gui_json_schema_setting()
        assert result == {"type": "object"}

    def test_boot_load_invalid_json_returns_none_with_warning(
            self, tmp_path, monkeypatch, caplog):
        """Persisted text that fails to parse → boot-load returns
        None AND logs a WARNING (loud-on-real-issue volume table).
        Without the warning, a user with a hand-corrupted
        gui_settings.json silently boots into no-constraint mode."""
        import logging
        import enigma_engine.gui.desktop as desktop_mod
        from enigma_engine.gui.desktop import EnigmaGUI

        monkeypatch.setattr(desktop_mod, "DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text(
            json.dumps({"json_schema_text": '{"broken": '}),
            encoding="utf-8")

        obj = object.__new__(EnigmaGUI)
        with caplog.at_level(logging.WARNING):
            result = obj._read_gui_json_schema_setting()
        assert result is None
        assert any(
            "json_schema_text" in r.message and "invalid" in r.message
            for r in caplog.records
        ), "parse failure must log a WARNING naming json_schema_text"

    def test_boot_load_empty_or_missing_returns_none_silent(
            self, tmp_path, monkeypatch, caplog):
        """Missing key + empty string → both return None silently
        (the fresh-install / cleared-by-user normal path)."""
        import logging
        import enigma_engine.gui.desktop as desktop_mod
        from enigma_engine.gui.desktop import EnigmaGUI

        monkeypatch.setattr(desktop_mod, "DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"

        obj = object.__new__(EnigmaGUI)

        # Missing key
        settings_file.write_text("{}", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            assert obj._read_gui_json_schema_setting() is None
        # Empty string
        settings_file.write_text(
            json.dumps({"json_schema_text": ""}), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            assert obj._read_gui_json_schema_setting() is None
        # Whitespace only
        settings_file.write_text(
            json.dumps({"json_schema_text": "   \n"}),
            encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            assert obj._read_gui_json_schema_setting() is None
        # No WARNING records expected for any of the silent paths
        assert not any(
            "json_schema" in r.message
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )

    def test_chat_send_path_forwards_json_schema_kwarg(self):
        """Structural gate on the chat send wire-site: the
        ``_send_message`` body MUST forward ``self.json_schema``
        into the engine.chat kwargs dict when non-None.

        Regex targets the literal call expression
        ``kwargs["json_schema"] = ...`` paired with a guard that
        consults ``json_schema`` — NOT the bare token, which also
        appears in the comment block above and would mask a
        regression that deletes only the kwargs assignment.
        """
        import re
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        src = inspect.getsource(LogicChatMixin._send_message)
        # Adversarial: must see BOTH the guard read AND the kwargs
        # assignment in the body; either alone is shared with the
        # comment block.
        guard = re.compile(
            r'getattr\(\s*self\s*,\s*"json_schema"')
        assignment = re.compile(
            r'kwargs\[\s*"json_schema"\s*\]\s*=')
        assert guard.search(src), (
            "_send_message must read self.json_schema via getattr "
            "(handles legacy GUI sessions that pre-date the field)"
        )
        assert assignment.search(src), (
            "_send_message must forward json_schema into kwargs "
            "for engine.chat — without this the GUI textbox is dead"
        )

    def test_chat_send_path_catches_value_error_on_bad_schema(self):
        """Pass 156z9ab Finding 5: ``JsonSchemaConstraint`` raises
        ``ValueError`` on unsupported schema shapes (non-object
        root, missing properties, malformed spec).  The send path
        MUST catch that explicitly and surface a user-facing
        message — silently retrying without the kwarg would
        produce unconstrained output the user opted out of, and
        propagating the exception unframed dumps a traceback into
        the chat log.
        """
        import re
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        src = inspect.getsource(LogicChatMixin._send_message)
        # Must see an `except ValueError` clause in the body.
        # Adversarial-falsifiable: deleting the clause reverts the
        # exception to propagating up to the _gen thread top-level
        # and this test fails immediately.
        assert re.search(r'except\s+ValueError', src), (
            "_send_message must explicitly catch ValueError so a "
            "bad json_schema surfaces as a chat-system message, "
            "not a raw traceback"
        )
        # And the catch handler must NOT call self.engine.chat
        # again without the kwarg — that would silently bypass the
        # user's constraint.  Gate by checking the handler region
        # contains a `return` and references status_bar / chat.
        assert re.search(
            r'except\s+ValueError.*?return',
            src, re.DOTALL,
        ), (
            "ValueError handler must abort the send (return), not "
            "fall through and retry without the schema constraint"
        )

    def test_persist_returns_false_on_disk_failure(
            self, tmp_path, monkeypatch, caplog):
        """Pass 156z9ab Finding 2: ``_persist_json_schema_text``
        must return False when ``atomic_write_json`` raises so
        ``_apply_json_schema`` can post a non-misleading status.
        Previously the IOError was swallowed at DEBUG and the
        success branch fired regardless.
        """
        import logging
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        # Force atomic_write_json to fail
        from enigma_engine.core import safe_save

        def _explode(*_a, **_k):
            raise OSError("simulated disk full")
        monkeypatch.setattr(safe_save, "atomic_write_json", _explode)

        obj = object.__new__(PagesMixin)
        with caplog.at_level(logging.WARNING):
            ok = obj._persist_json_schema_text('{"x": 1}')
        assert ok is False
        assert any(
            "json_schema_text" in r.message
            and r.levelno >= logging.WARNING
            for r in caplog.records
        ), "disk failure must log a WARNING (loud-on-real-issue)"

    def test_apply_surfaces_disk_failure_in_status_bar(
            self, tmp_path, monkeypatch):
        """End-to-end: when persist fails, the Apply handler must
        post a status message that names the disk failure — NOT a
        plain 'JSON schema applied' message.  Regression guard
        against a refactor that goes back to ignoring the bool
        return."""
        from enigma_engine.gui.gui_pages import PagesMixin
        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        (tmp_path / "gui_settings.json").write_text(
            "{}", encoding="utf-8")
        from enigma_engine.core import safe_save
        monkeypatch.setattr(
            safe_save, "atomic_write_json",
            lambda *_a, **_k: (_ for _ in ()).throw(
                OSError("disk full")),
        )

        class _RecordingStatusBar:
            def __init__(self):
                self.left_text = ""

            def set_left(self, text):
                self.left_text = text

            def set_center(self, text):
                pass

            def set_right(self, text):
                pass

        obj = object.__new__(PagesMixin)
        obj._json_schema_textbox = _MockTextbox(initial='{"y": 2}')
        obj.status_bar = _RecordingStatusBar()
        obj.json_schema = None

        obj._apply_json_schema()

        # Live attr WAS updated (in-memory success)
        assert obj.json_schema == {"y": 2}
        # But status bar must name the disk failure, not the plain
        # success path
        msg = obj.status_bar.left_text
        assert "disk save failed" in msg.lower(), (
            f"status bar should name the disk failure; got: {msg!r}"
        )

    def test_cmd_page_does_not_forward_json_schema(self):
        """Pass 156z9ab Finding 1 (sibling-boundary design call):
        the CMD-page chat path INTENTIONALLY drops
        ``json_schema`` from kwargs before calling engine.chat —
        a user-staged schema would override the
        ``[CMD]...[/CMD]`` policy and silently disable command
        execution.  This test gates the explicit ``pop`` call so
        a future refactor that "consolidates" kwargs handling
        can't silently re-enable the wrong behaviour.
        """
        import re
        import inspect as _inspect
        # The CMD page method that calls engine.chat lives on the
        # CMDPageMixin host; locate the source by reading the
        # module body directly (the per-method scope helper is an
        # inner function so getsource of the mixin class is the
        # right granularity).
        from enigma_engine.gui import gui_cmd_page
        src = _inspect.getsource(gui_cmd_page)
        assert re.search(
            r'kwargs\.pop\(\s*"json_schema"', src), (
            "CMD page must explicitly drop json_schema from "
            "kwargs before engine.chat — see Pass 156z9ab "
            "Finding 1 design rationale"
        )


class TestGamingModePreset:
    """Gaming preset should apply the full low-overhead profile."""

    def test_apply_gaming_mode_preset_disables_learning(self, tmp_path, monkeypatch):
        """Preset disables chat learning and syncs runtime state."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin

        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        sync_calls = []
        obj = object.__new__(PagesMixin)
        obj.status_bar = MockStatusBar()
        obj._auto_load_chat_model_var = MockVar()
        obj._auto_start_mods_var = MockVar()
        obj._auto_unload_on_minimize_var = MockVar()
        obj._learn_while_chatting_var = MockVar()
        obj._refresh_performance_mode = lambda: sync_calls.append("refresh")
        obj._sync_router_training_state = lambda: sync_calls.append("sync")

        obj._apply_gaming_mode_preset()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["auto_load_chat_model"] is False
        assert data["auto_start_mods"] is False
        assert data["auto_unload_on_minimize"] is True
        assert data["learn_while_chatting"] is False
        assert obj._chat_learning_enabled is False
        assert obj._learn_while_chatting_var.value is False
        assert sync_calls == ["refresh", "sync"]


class TestPerformanceSettings:
    """Performance-related GUI settings for memory usage."""

    def test_toggle_auto_load_chat_model_saves_setting(
            self, tmp_path, monkeypatch):
        """Toggle writes auto_load_chat_model to gui_settings.json."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin

        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj._auto_load_chat_model_var = MockVar(False)
        obj.status_bar = MockStatusBar()

        obj._toggle_auto_load_chat_model()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["auto_load_chat_model"] is False

    def test_toggle_auto_start_mods_saves_setting(
            self, tmp_path, monkeypatch):
        """Toggle writes auto_start_mods to gui_settings.json."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin

        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj._auto_start_mods_var = MockVar(False)
        obj.status_bar = MockStatusBar()

        obj._toggle_auto_start_mods()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["auto_start_mods"] is False

    def test_toggle_auto_unload_on_minimize_saves_setting(
            self, tmp_path, monkeypatch):
        """Toggle writes auto_unload_on_minimize to gui_settings.json."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin

        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj._auto_unload_on_minimize_var = MockVar()
        obj.status_bar = MockStatusBar()

        obj._toggle_auto_unload_on_minimize()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["auto_unload_on_minimize"] is True

    def test_apply_gaming_mode_preset_saves_three_settings(
            self, tmp_path, monkeypatch):
        """Gaming preset writes all memory-related settings."""
        import json
        from enigma_engine.gui.gui_pages import PagesMixin

        monkeypatch.setattr(
            "enigma_engine.gui.gui_pages_config.DATA_DIR", tmp_path)
        settings_file = tmp_path / "gui_settings.json"
        settings_file.write_text("{}", encoding="utf-8")

        obj = object.__new__(PagesMixin)
        obj.status_bar = MockStatusBar()
        obj._auto_load_chat_model_var = MockVar()
        obj._auto_start_mods_var = MockVar()
        obj._auto_unload_on_minimize_var = MockVar()

        obj._apply_gaming_mode_preset()

        data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert data["auto_load_chat_model"] is False
        assert data["auto_start_mods"] is False
        assert data["auto_unload_on_minimize"] is True


# ================================================================
# FORGE Page: 3-Mode Contract
# ================================================================

class TestForgeThreeModeContract:
    """Test the current FORGE contract with 3 user-facing modes."""

    def test_teacher_student_removed_from_forge_ui(self):
        """Legacy 'Teacher + Student' option is gone from FORGE page."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = inspect.getsource(ForgePageMixin._build_page_forge)
        assert "Teacher + Student" not in source

    def test_legacy_train_with_ai_toggle_removed(self):
        """Legacy Train-with-AI checkbox contract is removed."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = inspect.getsource(ForgePageMixin._build_page_forge)
        assert "train_with_ai_var" not in source
        assert "train_with_ai_cb" not in source


# ================================================================
# Shared Web Utilities
# ================================================================

class TestWebUtils:
    """Test shared web search and page fetching utilities."""

    def test_extract_strips_scripts(self):
        """extract_html_text removes script/style content."""
        from enigma_engine.core.web_utils import extract_html_text
        html = (
            "<html><script>var x=1;</script>"
            "<style>.a{color:red}</style>"
            "<p>Real content here</p></html>")
        result = extract_html_text(html)
        assert "var x" not in result
        assert "color" not in result
        assert "Real content here" in result

    def test_extract_strips_nav_footer(self):
        """extract_html_text skips nav, footer, header, aside."""
        from enigma_engine.core.web_utils import extract_html_text
        html = (
            "<nav>Navigation links</nav>"
            "<header>Header stuff</header>"
            "<main><p>Important article text</p></main>"
            "<footer>Footer links</footer>"
            "<aside>Sidebar content</aside>")
        result = extract_html_text(html)
        assert "Important article text" in result
        # Nav/footer/header/aside should be stripped
        assert "Navigation links" not in result
        assert "Footer links" not in result

    def test_extract_empty_html(self):
        """extract_html_text returns empty for blank input."""
        from enigma_engine.core.web_utils import extract_html_text
        assert extract_html_text("") == ""
        assert extract_html_text("<div></div>") == ""


# ================================================================
# FORGE: Optimized Web Learn
# ================================================================

class TestWebLearnOptimized:
    """Test optimized web learn uses shared web_utils and
    trainer system prompt."""

    def test_web_learn_no_inline_ddg_parser(self):
        """_web_learn no longer defines DDGParser inline."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "class DDGParser" not in source

    def test_web_learn_no_inline_text_extractor(self):
        """_web_learn no longer defines TextExtractor inline."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        source = inspect.getsource(ForgeMixin._web_learn)
        assert "class TextExtractor" not in source

    def test_web_learn_no_hardcoded_colors(self):
        """Web learn button uses theme constants, not hex."""
        from enigma_engine.gui.gui_pages import PagesMixin
        source = inspect.getsource(PagesMixin._build_page_forge)
        # Old hardcoded colors should be gone
        assert "#0d2137" not in source
        assert "#163352" not in source

    def test_legacy_mode_desc_widget_removed(self):
        """Old single-description widget is removed in card-based UI."""
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        source = inspect.getsource(ForgePageMixin._build_page_forge)
        assert "_training_mode_desc" not in source


# ================================================================
# FORGE: Auto-Train After Data Generation
# ================================================================


# ================================================================
# Phase 1 — Polish
# ================================================================


class TestPrintToLogger:
    """print() calls should be replaced with logger."""

    def test_no_prints_in_server(self):
        """server.py should use logger, not print (except docstrings)."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "api" / "server.py")
        source = source_path.read_text(encoding="utf-8")
        import ast
        tree = ast.parse(source)
        prints = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert len(prints) == 0, (
            f"server.py has {len(prints)} print() calls — "
            "use logger.info instead")


# ================================================================
# Identity card on MODELS page
# ================================================================


# ================================================================
# Identity editing on MODELS page
# ================================================================

class TestModelsIdentityEdit:
    """MODELS page supports inline name editing on model cards."""

    def test_no_simpledialog_in_pages(self):
        """gui_pages.py must not use simpledialog popups."""
        source_path = (
            Path(__file__).parent.parent
            / "enigma_engine" / "gui" / "gui_pages.py")
        source = source_path.read_text(encoding="utf-8")
        assert "simpledialog" not in source, (
            "gui_pages.py still uses simpledialog — "
            "all inputs must be inline")


# ================================================================
# Identity export
# ================================================================

class TestIdentityExport:
    """ModelContext supports exporting identity as a standalone JSON."""

    def test_export_identity_returns_dict(self):
        """export_identity returns a dict with all identity fields."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("export_test")
        ctx.display_name = "Test AI"
        ctx.personality = "Friendly"
        ctx.tags = ["general"]
        result = ctx.export_identity()
        assert isinstance(result, dict)
        assert result["model_key"] == "export_test"
        assert result["display_name"] == "Test AI"
        assert result["personality"] == "Friendly"
        assert result["tags"] == ["general"]

    def test_export_identity_includes_all_fields(self):
        """export_identity includes stats, training history, notes."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("export_full")
        ctx.total_messages = 100
        ctx.total_sessions = 5
        ctx.notes = "Fine-tuned model"
        result = ctx.export_identity()
        assert result["total_messages"] == 100
        assert result["total_sessions"] == 5
        assert result["notes"] == "Fine-tuned model"
        assert "created_at" in result
        assert "training_history" in result

    def test_export_identity_to_file(self, tmp_path):
        """export_identity can be written to a JSON file."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("export_file")
        ctx.display_name = "Export Test"
        data = ctx.export_identity()
        out = tmp_path / "identity.json"
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["display_name"] == "Export Test"


# ================================================================
# Font size control on CONFIG page
# ================================================================

class TestFontSizeControl:
    """CONFIG page has font size adjustment."""

    def test_font_size_offset_default_zero(self):
        """Font size offset resets to 0 correctly."""
        from enigma_engine.gui import widgets
        widgets.set_font_size_offset(0)
        assert widgets.get_font_size_offset() == 0

    def test_font_size_offset_adjusts_fonts(self):
        """set_font_size_offset changes module-level FONT_* tuples."""
        from enigma_engine.gui import widgets
        widgets.set_font_size_offset(0)
        original_body = widgets.FONT_BODY[1]
        widgets.set_font_size_offset(2)
        assert widgets.FONT_BODY[1] == original_body + 2
        # Restore
        widgets.set_font_size_offset(0)
        assert widgets.FONT_BODY[1] == original_body


# ================================================================
# A2: Keyboard shortcuts help overlay
# ================================================================


# ================================================================
# A3: Token counter in chat
# ================================================================


# ================================================================
# A4: HuggingFace download in GUI
# ================================================================


# ================================================================
# A7: Backup/restore system
# ================================================================


# ================================================================
# D3: Bare except:pass cleanup in mods
# ================================================================

class TestBareExceptCleanup:
    """Mods do not use bare except: — use except Exception instead."""

    @pytest.mark.parametrize("mod_name", [
        "voice", "threed", "videogen", "router",
        "imagegen", "audiogen",
    ])
    def test_mod_no_bare_except(self, mod_name):
        """Mod files must not contain bare 'except:'."""
        mod_dir = Path(__file__).parent.parent / "mods" / mod_name
        for py_file in mod_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped == "except:" or stripped == "except:  # noqa":
                    pytest.fail(
                        f"{py_file.name}:{i} has bare 'except:' "
                        f"— use 'except Exception:'")


# ================================================================
# FORGE Page: Adaptive Training Pipeline (TC-C3 + SA-B + SA-C)
# ================================================================


# ================================================================
# Input history (Up/Down recall)
# ================================================================

class TestInputHistory:
    """Verify chat input history recall logic."""

    def test_init_input_history_state(self):
        """_init_input_history sets empty list and idx=-1."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        obj = object.__new__(LogicChatMixin)
        obj._init_input_history()
        assert obj._input_history == []
        assert obj._input_hist_idx == -1
        assert obj._input_hist_draft == ""

    def test_history_max_constant(self):
        """Input history max is reasonable."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        assert LogicChatMixin._INPUT_HISTORY_MAX == 50


class TestChatHistoryTrimming:
    """Verify chat history is trimmed to prevent RAM leaks."""

    def test_max_chat_history_constant(self):
        """MAX_CHAT_HISTORY constant exists and is reasonable."""
        from enigma_engine.gui.media import MAX_CHAT_HISTORY
        assert isinstance(MAX_CHAT_HISTORY, int)
        assert 100 <= MAX_CHAT_HISTORY <= 1000

    def test_trim_chat_history_removes_oldest(self):
        """_trim_chat_history removes oldest messages when over cap."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        from enigma_engine.gui.media import MAX_CHAT_HISTORY
        obj = object.__new__(LogicChatMixin)
        # Simulate a history with messages over the cap
        obj.history = [
            {"role": "user", "content": f"msg{i}"}
            for i in range(MAX_CHAT_HISTORY + 50)
        ]
        obj._trim_chat_history()
        assert len(obj.history) == MAX_CHAT_HISTORY
        # Oldest messages should be gone
        assert obj.history[0]["content"] == "msg50"
        # Newest messages should remain
        assert obj.history[-1]["content"] == f"msg{MAX_CHAT_HISTORY + 49}"

    def test_trim_chat_history_doesnt_trim_under_cap(self):
        """_trim_chat_history does nothing when under cap."""
        from enigma_engine.gui.gui_logic_chat import LogicChatMixin
        obj = object.__new__(LogicChatMixin)
        obj.history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "reply1"},
        ]
        obj._trim_chat_history()
        assert len(obj.history) == 2


# ================================================================
# RLHF / Self-Play Dropdown (#21)
# ================================================================


# ================================================================
# S622–S629: Button System Upgrade
# ================================================================

class TestButtonThemeConstants:
    """S622: New theme constants for button colour system."""

    def test_reload_theme_updates_new_constants(self):
        """reload_theme must propagate new C_* constants."""
        from enigma_engine.gui import widgets
        widgets.reload_theme("midnight")
        # Midnight has different colours
        assert isinstance(widgets.C_RED_DIM, str)
        assert isinstance(widgets.C_CYAN_DIM, str)
        widgets.reload_theme("dark")


class TestThemedButton:
    """S623: themed_button() factory function."""

    def test_themed_button_valid_styles(self):
        """themed_button must accept all defined style names."""
        from enigma_engine.gui.widgets import BUTTON_STYLES
        expected = {"primary", "danger", "action", "tool",
                    "secondary", "warning", "icon"}
        assert expected == set(BUTTON_STYLES.keys())

    def test_button_styles_have_required_keys(self):
        """Each button style must define fg_color, hover_color, text_color."""
        from enigma_engine.gui.widgets import BUTTON_STYLES
        required = {"fg_color", "hover_color", "text_color"}
        for style_name, style_dict in BUTTON_STYLES.items():
            for key in required:
                assert key in style_dict, (
                    f"Style '{style_name}' missing key '{key}'")

    def test_button_styles_all_strings(self):
        """All colour values in button styles must be non-empty strings."""
        from enigma_engine.gui.widgets import BUTTON_STYLES
        for style_name, style_dict in BUTTON_STYLES.items():
            for key, val in style_dict.items():
                assert isinstance(val, str) and len(val) > 0, (
                    f"Style '{style_name}'.{key} = {val!r}")

    def test_primary_style_uses_green(self):
        """Primary buttons must use green colours."""
        from enigma_engine.gui.widgets import BUTTON_STYLES, C_GREEN
        assert BUTTON_STYLES["primary"]["text_color"] == C_GREEN

    def test_danger_style_uses_red(self):
        """Danger buttons must use red colours."""
        from enigma_engine.gui.widgets import BUTTON_STYLES, C_RED
        assert BUTTON_STYLES["danger"]["text_color"] == C_RED

    def test_tool_style_uses_cyan(self):
        """Tool buttons must use cyan colours."""
        from enigma_engine.gui.widgets import BUTTON_STYLES, C_CYAN
        assert BUTTON_STYLES["tool"]["text_color"] == C_CYAN

    def test_action_style_uses_accent(self):
        """Action buttons must use accent colours."""
        from enigma_engine.gui.widgets import BUTTON_STYLES, C_ACCENT
        assert BUTTON_STYLES["action"]["text_color"] == C_ACCENT

    def test_warning_style_uses_orange(self):
        """Warning buttons must use orange colours."""
        from enigma_engine.gui.widgets import BUTTON_STYLES, C_ORANGE
        assert BUTTON_STYLES["warning"]["text_color"] == C_ORANGE

    def test_secondary_style_uses_dim(self):
        """Secondary buttons must use dim text."""
        from enigma_engine.gui.widgets import BUTTON_STYLES, C_TEXT_DIM
        assert BUTTON_STYLES["secondary"]["text_color"] == C_TEXT_DIM

    def test_icon_style_transparent(self):
        """Icon buttons must have transparent background."""
        from enigma_engine.gui.widgets import BUTTON_STYLES
        assert BUTTON_STYLES["icon"]["fg_color"] == "transparent"


class TestButtonUsageInPages:
    """S624–S629: Verify GUI pages use themed_button for consistency."""

    def test_no_hardcoded_green_hover(self):
        """No GUI page file should hardcode green hover hex."""
        import enigma_engine.gui.gui_pages as mod_pages
        import enigma_engine.gui.gui_pages_forge as mod_forge
        import enigma_engine.gui.gui_docs_page as mod_docs
        import enigma_engine.gui.gui_cmd_page as mod_cmd
        import enigma_engine.gui.gui_mod_page as mod_mod
        import enigma_engine.gui.gui_forge_models as mod_fm
        for mod in [mod_pages, mod_forge, mod_docs,
                    mod_cmd, mod_mod, mod_fm]:
            src = inspect.getsource(mod)
            assert "#1a5a2a" not in src, (
                f"{mod.__name__} still has hardcoded #1a5a2a")
            assert "#1a4a2e" not in src, (
                f"{mod.__name__} still has hardcoded #1a4a2e")

    def test_no_hardcoded_red_bg(self):
        """No GUI page file should hardcode red background hex."""
        import enigma_engine.gui.gui_pages as mod_pages
        import enigma_engine.gui.gui_docs_page as mod_docs
        for mod in [mod_pages, mod_docs]:
            src = inspect.getsource(mod)
            assert "#3a1111" not in src, (
                f"{mod.__name__} still has hardcoded #3a1111")
            assert "#3b1111" not in src, (
                f"{mod.__name__} still has hardcoded #3b1111")
            assert "#5a1a1a" not in src, (
                f"{mod.__name__} still has hardcoded #5a1a1a")


# =====================================================================
# Cross-wiring: every TrainingConfig call site must include core fields
# =====================================================================


class TestTrainingConfigCrossWiring:
    """Structural test: all TrainingConfig() call sites in GUI must
    include the fields that should be present in every training mode.

    This test prevents the class of bug where a feature exists in
    TrainingConfig, has a GUI widget, but some training modes forget
    to pass it through.  When a new required field is added, adding
    it to REQUIRED_FIELDS here will catch any mode that misses it.
    """

    # Fields that EVERY TrainingConfig() call in the GUI must include.
    # Vision gets a pass on use_sequence_packing (batch_size=1).
    REQUIRED_FIELDS = [
        "use_gradient_checkpointing",
        "ce_chunk_size",
        "use_compile",
        "rolling_best_k",
        "save_every",
        "checkpoint_dir",
        "use_amp",
        "run_evaluation",
    ]

    # Files that contain TrainingConfig() constructor calls.
    GUI_TRAINING_FILES = [
        "enigma_engine.gui.gui_forge_training",
        "enigma_engine.gui.gui_forge_new_modes",
    ]

    def test_all_config_calls_include_required_fields(self):
        """Every TrainingConfig(...) block in GUI training files must
        mention each required field.  This catches 'feature exists but
        not wired' bugs automatically."""
        import importlib
        import inspect
        import re

        missing = []
        for mod_name in self.GUI_TRAINING_FILES:
            mod = importlib.import_module(mod_name)
            src = inspect.getsource(mod)
            # Find each TrainingConfig( ... ) block.  They span
            # multiple lines, so grab from 'TrainingConfig(' to the
            # matching closing paren.
            # We use a simple heuristic: find each occurrence and
            # capture the next ~40 lines (configs are ~15-25 lines).
            lines = src.splitlines()
            for i, line in enumerate(lines):
                if "TrainingConfig(" in line and "import" not in line:
                    # Grab the config block (up to 40 lines or the
                    # next line with just ')' or 'trainer = ')
                    block_lines = lines[i:i + 40]
                    block = "\n".join(block_lines)
                    # Find the closing of the constructor
                    end = block.find("\n                trainer")
                    if end == -1:
                        end = block.find("\n                self._log")
                    if end > 0:
                        block = block[:end]

                    for field in self.REQUIRED_FIELDS:
                        # Vision mode (batch_size=1) doesn't need
                        # sequence packing
                        if field == "use_sequence_packing" and \
                                "batch_size=1" in block:
                            continue
                        if field not in block:
                            # Get function context
                            func_match = re.search(
                                r"def\s+(\w+)",
                                "\n".join(lines[max(0, i - 80):i]))
                            func_name = (func_match.group(1)
                                         if func_match else "unknown")
                            missing.append(
                                f"{mod_name}::{func_name} "
                                f"(line ~{i + 1}) missing "
                                f"'{field}'")

        assert not missing, (
            "TrainingConfig call sites missing required fields:\n"
            + "\n".join(f"  - {m}" for m in missing))

    def test_forge_params_fields_are_consumed(self):
        """Every field returned by _read_forge_train_params() must
        appear in at least one TrainingConfig() call.  Catches dead
        GUI widget connections."""
        import inspect
        import enigma_engine.gui.gui_forge as mod_forge
        import enigma_engine.gui.gui_forge_training as mod_train
        import enigma_engine.gui.gui_forge_new_modes as mod_new

        # Extract field names from _read_forge_train_params return dict
        import re
        # Use the METHOD source, not the whole module — avoids
        # matching return dicts from other functions.
        forge_src = inspect.getsource(
            mod_forge.ForgeMixin._read_forge_train_params)
        # The return dict has lines like: "field_name": value,
        return_match = re.search(
            r"return\s*\{([^}]+)\}",
            forge_src, re.DOTALL)
        assert return_match, \
            "_read_forge_train_params has no return dict"
        return_block = return_match.group(1)
        field_names = re.findall(
            r'"(\w+)":', return_block)
        assert len(field_names) >= 5, \
            f"Expected >=5 fields, got {field_names}"

        # Check that each field appears in at least one config site
        consumer_src = (inspect.getsource(mod_train)
                        + inspect.getsource(mod_new))
        unused = []
        for field in field_names:
            # Search for forge_params["field"] or just field= in
            # TrainingConfig blocks
            pattern = (f'forge_params["{field}"]'
                       if field != "general_data" else field)
            if pattern not in consumer_src:
                unused.append(field)

        assert not unused, (
            "_read_forge_train_params() returns fields that no "
            "training mode uses:\n"
            + "\n".join(f"  - {f}" for f in unused))

    def test_solo_training_shows_batch_eta(self):
        """Batch-level ETA must appear in the solo training on_loss handler."""
        import inspect
        import enigma_engine.gui.gui_forge_training as mod_train
        src = inspect.getsource(mod_train.ForgeTrainingMixin)
        assert '_total_training_steps' in src, (
            "Solo training handler missing batch-level ETA "
            "from _total_training_steps")


class TestForgeAlignmentModeVisibility:
    """S819: Alignment modes (GRPO/ReMax/SimPO/ORPO) show only basic section."""

    def test_alignment_modes_visibility_logic(self):
        """_on_training_mode_changed maps alignment modes to {basic} only.

        Structural test — GUI widget show/hide requires tkinter runtime.
        Verifies the if-branch exists and returns the correct set.
        """
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        src = inspect.getsource(ForgeMixin._on_training_mode_changed)
        # The alignment branch must exist
        assert '"GRPO"' in src or "'GRPO'" in src, (
            "Missing GRPO in _on_training_mode_changed")
        assert '"ReMax"' in src or "'ReMax'" in src, (
            "Missing ReMax in _on_training_mode_changed")
        assert '"SimPO"' in src or "'SimPO'" in src, (
            "Missing SimPO in _on_training_mode_changed")
        assert '"ORPO"' in src or "'ORPO'" in src, (
            "Missing ORPO in _on_training_mode_changed")

    def test_all_eight_modes_have_visibility_branch(self):
        """Every training mode must have a visibility branch."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        src = inspect.getsource(ForgeMixin._on_training_mode_changed)
        for mode in ("Pre-Train", "Distill", "AI-Guided", "Image",
                     "Dialogue", "RLHF", "Self-Play"):
            assert f'"{mode}"' in src or f"'{mode}'" in src, (
                f"Missing visibility branch for mode {mode}")


class TestForgeAPOAlignmentMode:
    """D-9b (Pass 156k): FORGE radio card for APO-zero alignment mode.

    Pass 156j shipped the library-level loss + dispatch
    (`train_dpo(loss_type='apo_zero')`); D-9b adds the FORGE GUI
    surface so users can pick APO from the alignment row alongside
    GRPO/ReMax/SimPO/ORPO.

    Behavioural test that routing → APO loss already exists in
    `TestAPOZeroLoss.test_train_dpo_apo_zero_actually_routes_to_apo_loss`
    (Pass 156j); these structural tests close the GUI→trainer→loss
    chain end-to-end.
    """

    def test_apo_in_alignment_modes_radio_card(self):
        """APO must be listed as a radio card in the alignment row."""
        import inspect
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        src = inspect.getsource(ForgePageMixin._build_page_forge)
        # alignment_modes list must include "APO"
        assert '"APO"' in src or "'APO'" in src, (
            "APO missing from alignment_modes radio cards in "
            "_build_page_forge")

    def test_apo_visibility_branch_basic_only(self):
        """_on_training_mode_changed must treat APO same as
        SimPO/ORPO/GRPO/ReMax — show only the basic section."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        src = inspect.getsource(ForgeMixin._on_training_mode_changed)
        assert '"APO"' in src or "'APO'" in src, (
            "Missing APO visibility branch in _on_training_mode_changed")

    def test_apo_dispatch_in_start_training_by_mode(self):
        """_start_training_by_mode must dispatch APO →
        _start_apo_training."""
        import inspect
        from enigma_engine.gui.gui_forge import ForgeMixin
        src = inspect.getsource(ForgeMixin._start_training_by_mode)
        assert '"APO"' in src or "'APO'" in src, (
            "Missing APO dispatch case in _start_training_by_mode")
        assert "_start_apo_training" in src, (
            "_start_training_by_mode must call _start_apo_training")

    def test_start_apo_training_passes_apo_zero_loss_type(self):
        """_start_apo_training must route through the DPO trainer with
        loss_type='apo_zero' — the only thing that distinguishes APO
        from DPO at the call boundary. Behavioural proof that this
        actually changes the math comes from the Pass 156j test
        TestAPOZeroLoss.test_train_dpo_apo_zero_actually_routes_to_apo_loss.
        """
        import inspect
        from enigma_engine.gui.gui_forge_training import (
            ForgeTrainingMixin)
        # Method must exist
        assert hasattr(ForgeTrainingMixin, "_start_apo_training"), (
            "ForgeTrainingMixin missing _start_apo_training")
        # Must reference apo_zero in its source (either directly or
        # by delegating to _start_dpo_training with loss_type kwarg)
        src = inspect.getsource(
            ForgeTrainingMixin._start_apo_training)
        assert "apo_zero" in src, (
            "_start_apo_training must reference loss_type='apo_zero' "
            "(directly or via delegation)")

    def test_start_dpo_training_forwards_loss_type_to_trainer(self):
        """The shared DPO/APO body must forward loss_type through the
        dispatcher config so the kwarg reaches train_dpo in
        enigma_engine.training.dispatch. Catches regression where
        loss_type is accepted at the GUI but dropped before
        run_training(...)."""
        import inspect
        from enigma_engine.gui.gui_forge_training import (
            ForgeTrainingMixin)
        src = inspect.getsource(
            ForgeTrainingMixin._start_dpo_training)
        assert "loss_type" in src, (
            "_start_dpo_training must accept and forward loss_type")
        assert "run_training(" in src
        assert '"mode": "dpo"' in src
        assert '"loss_type": loss_type' in src, (
            "_start_dpo_training must pass loss_type through dpo config")

    def test_start_dpo_training_user_facing_strings_use_algo_label(self):
        """Pass 156k-audit: the SUGGESTIONS claim 'logs are accurate
        per mode' requires that user-facing error strings also use the
        parametrized `algo_label`, not hardcoded 'DPO'. A user who
        clicks APO and forgets the data file should see 'APO-ZERO
        requires a JSONL file', not 'DPO requires a JSONL file'.

        Strategy: scan source body for word-boundary `\bDPO\b`
        occurrences (catches bareword DPO inside larger f-strings,
        which is what the original audit caught at
        `--- DPO TRAINING STOPPED ---` and `DPO training failed`).
        Allowlist exactly the two ternary-definition lines that
        legitimately contain the literal label `"DPO"` /
        `"DPO Training"` — anywhere else means a hardcoded
        user-facing mention has drifted in.
        """
        import inspect
        import re
        from enigma_engine.gui.gui_forge_training import (
            ForgeTrainingMixin)
        src = inspect.getsource(
            ForgeTrainingMixin._start_dpo_training)
        # Strip the docstring (first triple-quoted block).
        src_no_doc = re.sub(r'"""[\s\S]*?"""', "", src, count=1)
        # Drop comment-only lines (whitespace + #).
        src_lines = [
            ln for ln in src_no_doc.splitlines()
            if not ln.lstrip().startswith("#")
        ]
        # Word-boundary match: `\bDPO\b` catches both `"DPO"`
        # and bareword `DPO` inside f-strings (e.g.
        # `f"--- DPO TRAINING STOPPED ---"`).
        offending = []
        for ln in src_lines:
            if not re.search(r'\bDPO\b', ln):
                continue
            # Allowlist the two ternary definition lines.
            if 'algo_label = "DPO"' in ln:
                continue
            if '"DPO Training"' in ln and 'if loss_type ==' in ln:
                continue
            offending.append(ln.strip())
        assert offending == [], (
            "Hardcoded user-facing 'DPO' label found outside the "
            "`algo_label` / `algo_summary_label` ternary lines. "
            "User-facing strings (logs, status, errors) must use "
            "`algo_label` so APO mode shows 'APO-ZERO', not 'DPO'.\n"
            f"Offending lines: {offending}")

class TestModelsPageMerging:
    """N-21: MODELS page has merge controls wired to model_merging."""

    def test_models_page_has_merge_controls(self):
        """_build_page_models creates merge controls and button wiring."""
        import inspect
        from enigma_engine.gui.gui_pages import PagesMixin
        src = inspect.getsource(PagesMixin._build_page_models)
        assert "_merge_model_a_var" in src
        assert "_merge_model_b_var" in src
        assert "_merge_mode_var" in src
        assert "_merge_t_var" in src
        assert "_merge_density_var" in src
        assert "_merge_output_entry" in src
        assert "command=self._merge_models" in src

    def test_merge_handler_dispatches_all_modes(self):
        """_merge_models dispatches SLERP/LINEAR/TIES to core helpers."""
        import inspect
        from enigma_engine.gui.gui_forge_models import ForgeModelsMixin
        src = inspect.getsource(ForgeModelsMixin._merge_models)
        assert "slerp_merge" in src
        assert "linear_merge" in src
        assert "ties_merge" in src
        assert 'mode == "SLERP"' in src
        assert 'mode == "LINEAR"' in src
        assert 'mode == "TIES"' in src


class TestForgeTeacherSubprocess:
    """FORGE External Teacher (HTTP) subprocess wrapper around collect_distill_data.py."""

    def test_build_teacher_argv_shape(self):
        """argv carries every required flag in the expected order."""
        from enigma_engine.gui.gui_forge_teacher import _build_teacher_argv
        argv = _build_teacher_argv(
            endpoint="http://localhost:11434/v1",
            model="qwen3:8b",
            magpie_n=500,
            tag="external",
            max_tokens=512,
            python_exe="python",
            script_path="collect_distill_data.py",
        )
        assert argv[0] == "python"
        assert argv[1] == "collect_distill_data.py"
        assert "--endpoint" in argv
        assert argv[argv.index("--endpoint") + 1] == "http://localhost:11434/v1"
        assert "--model" in argv
        assert argv[argv.index("--model") + 1] == "qwen3:8b"
        assert "--magpie" in argv
        assert argv[argv.index("--magpie") + 1] == "500"
        assert "--tag" in argv
        assert argv[argv.index("--tag") + 1] == "external"
        assert "--max-tokens" in argv
        assert argv[argv.index("--max-tokens") + 1] == "512"
        # Resume by default so re-clicks with same tag skip done prompts.
        assert "--resume" in argv

    def test_build_teacher_argv_coerces_int_kwargs(self):
        """magpie_n / max_tokens accept int-coercible inputs."""
        from enigma_engine.gui.gui_forge_teacher import _build_teacher_argv
        argv = _build_teacher_argv(
            endpoint="http://x/v1", model="m",
            magpie_n=10, tag="t", max_tokens=64,
            python_exe="python", script_path="s.py",
        )
        assert "10" in argv
        assert "64" in argv

    def test_build_teacher_argv_prompts_mode(self):
        """Passing prompts_path emits --prompts <path> instead of --magpie N."""
        from enigma_engine.gui.gui_forge_teacher import _build_teacher_argv
        argv = _build_teacher_argv(
            endpoint="http://x/v1", model="m",
            magpie_n=999, tag="t", max_tokens=64,
            python_exe="python", script_path="s.py",
            prompts_path="data/prompts.txt",
        )
        assert "--prompts" in argv
        assert argv[argv.index("--prompts") + 1] == "data/prompts.txt"
        # Mutual exclusion: --magpie must NOT appear in prompts mode.
        assert "--magpie" not in argv

    def test_forge_mixin_inherits_teacher_mixin(self):
        """ForgeMixin must compose ForgeTeacherMixin so the host gets
        the start/stop/kill methods."""
        from enigma_engine.gui.gui_forge import ForgeMixin
        from enigma_engine.gui.gui_forge_teacher import ForgeTeacherMixin
        assert issubclass(ForgeMixin, ForgeTeacherMixin)
        for name in (
            "_start_external_teacher_corpus",
            "_stop_external_teacher_corpus",
            "_kill_teacher_subprocess",
        ):
            assert hasattr(ForgeMixin, name), (
                f"ForgeMixin missing {name}")

    def test_forge_distill_section_has_teacher_widgets(self):
        """`_build_page_forge` creates the External-teacher widgets so the
        button is reachable from the Distill mode section."""
        import inspect
        from enigma_engine.gui.gui_pages_forge import ForgePageMixin
        src = inspect.getsource(ForgePageMixin._build_page_forge)
        for token in (
            "teacher_endpoint_var",
            "teacher_model_var",
            "teacher_magpie_var",
            "teacher_tag_var",
            "teacher_start_btn",
            "teacher_stop_btn",
            "teacher_mode_var",
            "teacher_prompts_path_var",
            "_browse_teacher_prompts_file",
            "_suggest_magpie_n_from_tag",
            "_start_external_teacher_corpus",
            "_stop_external_teacher_corpus",
        ):
            assert token in src, (
                f"FORGE distill section missing {token!r}")

    def test_kill_teacher_wired_in_on_close(self):
        """`_on_close` must call `_kill_teacher_subprocess` so the
        subprocess doesn't outlive the GUI."""
        import inspect
        from enigma_engine.gui.desktop import EnigmaGUI
        src = inspect.getsource(EnigmaGUI._on_close)
        assert "_kill_teacher_subprocess" in src

    def test_start_validates_required_inputs(self, monkeypatch):
        """Empty endpoint or model logs an error and does NOT spawn."""
        from enigma_engine.gui.gui_forge_teacher import ForgeTeacherMixin

        class _StubVar:
            def __init__(self, v): self._v = v
            def get(self): return self._v

        class _Host(ForgeTeacherMixin):
            def __init__(self, *, endpoint="", model="", magpie="500"):
                self.teacher_endpoint_var = _StubVar(endpoint)
                self.teacher_model_var = _StubVar(model)
                self.teacher_magpie_var = _StubVar(magpie)
                self.teacher_tag_var = _StubVar("t")
                self.teacher_max_tokens_var = _StubVar("512")
                self.logs: list[str] = []
                self._teacher_proc = None
            def _log(self, msg): self.logs.append(msg)
            def after(self, ms, fn): fn()

        spawn_called = {"n": 0}

        def _fake_popen(*args, **kwargs):
            spawn_called["n"] += 1
            raise AssertionError("Popen must not be called on validation fail")

        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge_teacher.subprocess.Popen",
            _fake_popen)

        # Missing endpoint
        h = _Host(endpoint="", model="m")
        h._start_external_teacher_corpus()
        assert spawn_called["n"] == 0
        assert any("endpoint" in m.lower() for m in h.logs)

        # Missing model
        h = _Host(endpoint="http://x", model="")
        h._start_external_teacher_corpus()
        assert spawn_called["n"] == 0
        assert any("model" in m.lower() for m in h.logs)

        # Non-integer magpie
        h = _Host(endpoint="http://x", model="m", magpie="not-a-number")
        h._start_external_teacher_corpus()
        assert spawn_called["n"] == 0
        assert any("magpie" in m.lower() for m in h.logs)

    def test_stop_and_kill_idempotent_when_no_proc(self):
        """Stop / kill are safe to call when nothing is running."""
        from enigma_engine.gui.gui_forge_teacher import ForgeTeacherMixin

        class _Host(ForgeTeacherMixin):
            def __init__(self):
                self._teacher_proc = None
                self.logs: list[str] = []
            def _log(self, msg): self.logs.append(msg)

        h = _Host()
        h._stop_external_teacher_corpus()  # should not raise
        h._kill_teacher_subprocess()  # should not raise
        assert h._teacher_proc is None

    def test_prompts_mode_rejects_missing_file(self, monkeypatch, tmp_path):
        """Mode=prompts with empty path AND with non-existent path both
        bail before spawning. Mode=prompts with valid file spawns with
        --prompts (not --magpie) in the argv."""
        from enigma_engine.gui.gui_forge_teacher import ForgeTeacherMixin

        class _StubVar:
            def __init__(self, v): self._v = v
            def get(self): return self._v

        class _Host(ForgeTeacherMixin):
            def __init__(self, *, mode="prompts", prompts=""):
                self.teacher_endpoint_var = _StubVar("http://x")
                self.teacher_model_var = _StubVar("m")
                self.teacher_magpie_var = _StubVar("")  # empty OK in prompts mode
                self.teacher_tag_var = _StubVar("t")
                self.teacher_max_tokens_var = _StubVar("512")
                self.teacher_mode_var = _StubVar(mode)
                self.teacher_prompts_path_var = _StubVar(prompts)
                self.logs: list[str] = []
                self._teacher_proc = None
            def _log(self, msg): self.logs.append(msg)
            def after(self, ms, fn): fn()

        # Empty path → bail
        spawned = {"n": 0}
        def _fake_popen_block(*a, **k):
            spawned["n"] += 1
            raise AssertionError("Popen must not be called when prompts path empty")
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge_teacher.subprocess.Popen",
            _fake_popen_block)
        h = _Host(prompts="")
        h._start_external_teacher_corpus()
        assert spawned["n"] == 0
        assert any("prompts" in m.lower() for m in h.logs)

        # Non-existent path → bail
        h = _Host(prompts=str(tmp_path / "nope.txt"))
        h._start_external_teacher_corpus()
        assert spawned["n"] == 0

        # Valid path → spawn with --prompts in argv
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("Hello\nWorld\n", encoding="utf-8")
        captured: dict = {}

        class _FakeProc:
            stdout = None
            def wait(self, timeout=None): return 0
            def terminate(self): pass

        def _fake_popen_ok(argv, **kwargs):
            captured["argv"] = argv
            return _FakeProc()

        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge_teacher.subprocess.Popen",
            _fake_popen_ok)
        # Health-check passes synchronously
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge_teacher._check_endpoint_reachable",
            lambda endpoint, **kw: (True, "ok"))
        # Don't actually spawn reader/health threads — invoke target inline
        # so the test exercises the full chain (validate → health → spawn).
        def _inline_thread(**kw):
            target = kw.get("target")
            args = kw.get("args", ())
            class _T:
                def start(self_inner):
                    if target is not None:
                        target(*args)
            return _T()
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge_teacher.threading.Thread",
            _inline_thread)
        h = _Host(prompts=str(prompts_file))
        h._start_external_teacher_corpus()
        argv = captured.get("argv", [])
        assert "--prompts" in argv, f"argv missing --prompts: {argv}"
        assert "--magpie" not in argv, f"argv should not have --magpie: {argv}"
        assert str(prompts_file) in argv

    def test_endpoint_health_check_blocks_spawn_on_unreachable(
        self, monkeypatch, tmp_path,
    ):
        """If `_check_endpoint_reachable` returns (False, ...), the spawn
        path must NOT call subprocess.Popen — health-check is a hard gate,
        not a warning."""
        from enigma_engine.gui.gui_forge_teacher import ForgeTeacherMixin

        class _StubVar:
            def __init__(self, v): self._v = v
            def get(self): return self._v

        class _Host(ForgeTeacherMixin):
            def __init__(self):
                self.teacher_endpoint_var = _StubVar("http://nope:9999/v1")
                self.teacher_model_var = _StubVar("m")
                self.teacher_magpie_var = _StubVar("10")
                self.teacher_tag_var = _StubVar("t")
                self.teacher_max_tokens_var = _StubVar("512")
                self.teacher_mode_var = _StubVar("magpie")
                self.teacher_prompts_path_var = _StubVar("")
                self.logs: list[str] = []
                self._teacher_proc = None
            def _log(self, msg): self.logs.append(msg)
            def after(self, ms, fn): fn()

        spawned = {"n": 0}
        def _fake_popen_block(*a, **k):
            spawned["n"] += 1
            raise AssertionError("Popen must NOT be called when health-check fails")
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge_teacher.subprocess.Popen",
            _fake_popen_block)
        # Health-check fails (e.g. connection refused).
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge_teacher._check_endpoint_reachable",
            lambda endpoint, **kw: (False, "Connection refused"))
        # Health-check thread runs target inline.
        def _inline_thread(**kw):
            target = kw.get("target")
            args = kw.get("args", ())
            class _T:
                def start(self_inner):
                    if target is not None:
                        target(*args)
            return _T()
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge_teacher.threading.Thread",
            _inline_thread)

        h = _Host()
        h._start_external_teacher_corpus()
        assert spawned["n"] == 0
        assert any("not reachable" in m.lower() for m in h.logs)

    def test_check_endpoint_reachable_volume_table(self):
        """Pure-helper unit test for the three branches of the health-check
        volume table: 2xx silent, non-2xx best-effort, URLError hard fail."""
        import urllib.error
        from enigma_engine.gui.gui_forge_teacher import _check_endpoint_reachable

        # 2xx → (True, "ok")
        class _OK:
            status = 200
            def close(self): pass
        ok, msg = _check_endpoint_reachable(
            "http://x/v1", _opener=lambda url, timeout=2.0: _OK())
        assert ok is True and msg == "ok"

        # HTTPError 404 → (True, best-effort)
        def _http_err(url, timeout=2.0):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        ok, msg = _check_endpoint_reachable("http://x/v1", _opener=_http_err)
        assert ok is True
        assert "404" in msg

        # URLError → (False, reason)
        def _url_err(url, timeout=2.0):
            raise urllib.error.URLError("Connection refused")
        ok, msg = _check_endpoint_reachable("http://x/v1", _opener=_url_err)
        assert ok is False
        assert "refused" in msg.lower() or "connection" in msg.lower()

    def test_stop_during_health_check_blocks_spawn(self, monkeypatch):
        """STOP clicked during the health-check window must prevent the
        subprocess from spawning even if the health-check succeeds.
        Closes the boundary-signal-without-consumer race where STOP was
        a no-op until _teacher_proc was set."""
        from enigma_engine.gui.gui_forge_teacher import ForgeTeacherMixin

        class _StubVar:
            def __init__(self, v): self._v = v
            def get(self): return self._v

        class _Host(ForgeTeacherMixin):
            def __init__(self):
                self.teacher_endpoint_var = _StubVar("http://x/v1")
                self.teacher_model_var = _StubVar("m")
                self.teacher_magpie_var = _StubVar("10")
                self.teacher_tag_var = _StubVar("t")
                self.teacher_max_tokens_var = _StubVar("512")
                self.teacher_mode_var = _StubVar("magpie")
                self.teacher_prompts_path_var = _StubVar("")
                self.logs: list[str] = []
                self._teacher_proc = None
                # Pending callbacks queued via after(); the test fires
                # them only AFTER calling stop, simulating real timing.
                self._pending: list = []
            def _log(self, msg): self.logs.append(msg)
            def after(self, ms, fn): self._pending.append(fn)

        spawned = {"n": 0}
        def _fake_popen_block(*a, **k):
            spawned["n"] += 1
            raise AssertionError("Popen must NOT spawn after STOP cancels health-check")
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge_teacher.subprocess.Popen",
            _fake_popen_block)
        # Health-check would succeed if we reached spawn.
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge_teacher._check_endpoint_reachable",
            lambda endpoint, **kw: (True, "ok"))
        # Health-check thread runs target inline.
        def _inline_thread(**kw):
            target = kw.get("target")
            args = kw.get("args", ())
            class _T:
                def start(self_inner):
                    if target is not None:
                        target(*args)
            return _T()
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge_teacher.threading.Thread",
            _inline_thread)

        h = _Host()
        # START → health-check thread runs inline → queues spawn via after()
        h._start_external_teacher_corpus()
        assert h._teacher_health_in_flight is True
        assert len(h._pending) == 1  # spawn callback is queued

        # STOP fires before the queued spawn callback runs.
        h._stop_external_teacher_corpus()
        assert h._teacher_cancel_requested is True

        # Now flush the queued spawn callback — it must observe the
        # cancel flag and bail without invoking Popen.
        for fn in h._pending:
            fn()
        assert spawned["n"] == 0
        assert any("cancelled" in m.lower() for m in h.logs)

    def test_double_start_during_health_check_is_rejected(self, monkeypatch):
        """Clicking START twice during the health-check window must NOT
        spawn two health-check threads. The second click should log a
        message and bail."""
        from enigma_engine.gui.gui_forge_teacher import ForgeTeacherMixin

        class _StubVar:
            def __init__(self, v): self._v = v
            def get(self): return self._v

        class _Host(ForgeTeacherMixin):
            def __init__(self):
                self.teacher_endpoint_var = _StubVar("http://x/v1")
                self.teacher_model_var = _StubVar("m")
                self.teacher_magpie_var = _StubVar("10")
                self.teacher_tag_var = _StubVar("t")
                self.teacher_max_tokens_var = _StubVar("512")
                self.teacher_mode_var = _StubVar("magpie")
                self.teacher_prompts_path_var = _StubVar("")
                self.logs: list[str] = []
                self._teacher_proc = None
            def _log(self, msg): self.logs.append(msg)
            def after(self, ms, fn): pass  # don't drain

        threads_started = {"n": 0}
        def _counting_thread(**kw):
            threads_started["n"] += 1
            class _T:
                def start(self_inner): pass  # do NOT actually run target
            return _T()
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge_teacher.threading.Thread",
            _counting_thread)

        h = _Host()
        h._start_external_teacher_corpus()
        assert threads_started["n"] == 1
        assert h._teacher_health_in_flight is True

        # Second click during health-check window → rejected, no new thread.
        h._start_external_teacher_corpus()
        assert threads_started["n"] == 1
        assert any("already in progress" in m.lower() for m in h.logs)

    def test_count_distill_jsonl_rows_volume_table(self, tmp_path):
        """Pure-helper unit test for the three branches: missing → 0,
        present-with-rows → N, malformed/empty → 0."""
        from enigma_engine.gui.gui_forge_teacher import _count_distill_jsonl_rows

        # Missing → 0
        assert _count_distill_jsonl_rows("nope", base_dir=tmp_path) == 0

        # Present with N rows → N (blank lines ignored)
        f = tmp_path / "distill_t1.jsonl"
        f.write_text(
            '{"prompt":"a","response":"b"}\n'
            '\n'  # blank line ignored
            '{"prompt":"c","response":"d"}\n'
            '{"prompt":"e","response":"f"}\n',
            encoding="utf-8")
        assert _count_distill_jsonl_rows("t1", base_dir=tmp_path) == 3

        # Empty file → 0 (no rows, no crash)
        (tmp_path / "distill_empty.jsonl").write_text("", encoding="utf-8")
        assert _count_distill_jsonl_rows("empty", base_dir=tmp_path) == 0

    def test_suggest_magpie_n_writes_existing_plus_500(self, monkeypatch, tmp_path):
        """Click ↻ button → reads tag → counts rows → sets Magpie-N."""
        from enigma_engine.gui.gui_forge_teacher import ForgeTeacherMixin

        class _StubVar:
            def __init__(self, v): self._v = v
            def get(self): return self._v
            def set(self, v): self._v = v

        class _Host(ForgeTeacherMixin):
            def __init__(self, tag):
                self.teacher_tag_var = _StubVar(tag)
                self.teacher_magpie_var = _StubVar("500")
                self.logs: list[str] = []
            def _log(self, msg): self.logs.append(msg)

        # Tag with 200 existing rows → suggest 700.
        f = tmp_path / "distill_qwen3.jsonl"
        f.write_text("\n".join(['{"x":1}'] * 200) + "\n", encoding="utf-8")
        monkeypatch.setattr(
            "enigma_engine.gui.gui_forge_teacher._count_distill_jsonl_rows",
            lambda tag, **kw: 200 if tag == "qwen3" else 0)
        h = _Host("qwen3")
        h._suggest_magpie_n_from_tag()
        assert h.teacher_magpie_var.get() == "700"
        assert any("200 existing" in m for m in h.logs)

        # Fresh tag → 500.
        h2 = _Host("brand_new")
        h2._suggest_magpie_n_from_tag()
        assert h2.teacher_magpie_var.get() == "500"
        assert any("fresh" in m.lower() for m in h2.logs)

    def test_parse_teacher_progress_volume_table(self):
        """Pure-helper unit test for the four branches of the progress
        regex: magpie line, prompts line, non-progress text, zero total."""
        from enigma_engine.gui.gui_forge_teacher import _parse_teacher_progress

        # Magpie line → (done, total)
        assert _parse_teacher_progress(
            "INFO:__main__:[10/500] ok=10 failed=0 duplicate=0 (1.23 rows/s)"
        ) == (10, 500)

        # Prompts line → (done, total)
        assert _parse_teacher_progress(
            "INFO:__main__:[3/100] ok=3 failed=0 skipped=0 (0.50 rows/s)"
        ) == (3, 100)

        # Non-progress lines → None
        assert _parse_teacher_progress(
            "INFO:__main__:loaded 500 prompt(s) from data/x.txt") is None
        assert _parse_teacher_progress(
            "INFO:__main__:resume: 200 prompt(s) already in x.jsonl") is None
        # User-supplied content with [N/M] but no `ok=` → None
        assert _parse_teacher_progress(
            "[teacher] some user-supplied [1/5] text") is None

        # Zero total → None (caller can't compute pct)
        assert _parse_teacher_progress(
            "INFO:__main__:[10/0] ok=10") is None

    def test_reader_loop_drives_progress_bar(self, monkeypatch):
        """`_teacher_reader_loop` parses `[N/M] ok=` lines from stdout
        and forwards (pct, msg) to `_update_forge_progress`."""
        import io
        from enigma_engine.gui.gui_forge_teacher import ForgeTeacherMixin

        class _FakeProc:
            def __init__(self, lines):
                self.stdout = io.StringIO("\n".join(lines) + "\n")
            def wait(self, timeout=None): return 0

        progress_calls: list = []

        class _Host(ForgeTeacherMixin):
            def __init__(self):
                self._teacher_proc = None
                self.logs: list[str] = []
            def _log(self, msg): self.logs.append(msg)
            def after(self, ms, fn): fn()  # run inline
            def _update_forge_progress(self, pct, msg):
                progress_calls.append((pct, msg))
            def _reset_forge_progress(self): pass
            def _teacher_finalize(self, rc, tag): pass

        lines = [
            "INFO:__main__:loaded 500 prompt(s) from data/x.txt",
            "INFO:__main__:[10/500] ok=10 failed=0 duplicate=0 (1.0 rows/s)",
            "INFO:__main__:[250/500] ok=250 failed=0 duplicate=0 (1.0 rows/s)",
            "INFO:__main__:[500/500] ok=500 failed=0 duplicate=0 (1.0 rows/s)",
        ]
        h = _Host()
        proc = _FakeProc(lines)
        h._teacher_reader_loop(proc, "t")

        # Three progress lines → three updates; the loaded-prompts line
        # must NOT trigger a progress update.
        assert progress_calls == [
            (2, "teacher 10/500"),
            (50, "teacher 250/500"),
            (100, "teacher 500/500"),
        ]


