"""Tests for ``enigma_engine.core.style_preferences`` (PERSONA-2 Slice 1).

Black-box invariant: writing style preferences must NOT touch the model
artifact. The model file under ``models/`` stays byte-identical when
style preferences are written. Style preferences are user-side runtime
config that follows the user, not the model.
"""
from __future__ import annotations

import json

import pytest


# =============================================================================
# Defaults — must preserve current behavior (no regression for existing users)
# =============================================================================


class TestStylePreferencesDefaults:
    """Defaults must match the pre-PERSONA-2 behavior exactly."""

    def test_default_verbosity_is_normal(self):
        from enigma_engine.core.style_preferences import StylePreferences
        assert StylePreferences().verbosity == "normal"

    def test_default_formality_is_neutral(self):
        from enigma_engine.core.style_preferences import StylePreferences
        assert StylePreferences().formality == "neutral"

    def test_default_length_is_medium(self):
        from enigma_engine.core.style_preferences import StylePreferences
        assert StylePreferences().default_response_length == "medium"

    def test_default_code_examples_off(self):
        from enigma_engine.core.style_preferences import StylePreferences
        assert StylePreferences().prefer_code_examples is False

    def test_default_bullet_points_off(self):
        from enigma_engine.core.style_preferences import StylePreferences
        assert StylePreferences().prefer_bullet_points is False

    def test_is_default_true_for_defaults(self):
        """``is_default()`` returns True for an unmodified instance — used by
        the injection layer to skip the prompt block entirely when the user
        hasn't customized anything."""
        from enigma_engine.core.style_preferences import StylePreferences
        assert StylePreferences().is_default() is True

    def test_is_default_false_for_any_change(self):
        from enigma_engine.core.style_preferences import StylePreferences
        assert StylePreferences(verbosity="terse").is_default() is False
        assert StylePreferences(formality="formal").is_default() is False
        assert StylePreferences(default_response_length="short").is_default() is False
        assert StylePreferences(prefer_code_examples=True).is_default() is False
        assert StylePreferences(prefer_bullet_points=True).is_default() is False


# =============================================================================
# Validation — enum fields must fail loud on invalid values (no silent coerce)
# =============================================================================


class TestStylePreferencesValidation:
    """Invalid enum values must raise ValueError at construction.

    Style overrides like ``/style "be a pirate"`` are REFUSED because the
    chat command will only accept known enum values — that's identity,
    not style. Validation here enforces it at the type boundary.
    """

    def test_invalid_verbosity_raises(self):
        from enigma_engine.core.style_preferences import StylePreferences
        with pytest.raises(ValueError, match="verbosity"):
            StylePreferences(verbosity="pirate")

    def test_invalid_formality_raises(self):
        from enigma_engine.core.style_preferences import StylePreferences
        with pytest.raises(ValueError, match="formality"):
            StylePreferences(formality="rude")

    def test_invalid_length_raises(self):
        from enigma_engine.core.style_preferences import StylePreferences
        with pytest.raises(ValueError, match="length"):
            StylePreferences(default_response_length="enormous")


class TestStylePreferencesValidEnumValues:
    """Every value in VERBOSITY_VALUES / FORMALITY_VALUES / LENGTH_VALUES must
    be acceptable by the constructor. Gates against a typo in the constants
    that would silently break a valid user choice (F3 audit follow-up).
    """

    @pytest.mark.parametrize(
        "value",
        ["terse", "normal", "verbose"],
    )
    def test_all_verbosity_values_accepted(self, value):
        from enigma_engine.core.style_preferences import (
            VERBOSITY_VALUES,
            StylePreferences,
        )
        assert value in VERBOSITY_VALUES
        prefs = StylePreferences(verbosity=value)
        assert prefs.verbosity == value

    @pytest.mark.parametrize(
        "value",
        ["casual", "neutral", "formal"],
    )
    def test_all_formality_values_accepted(self, value):
        from enigma_engine.core.style_preferences import (
            FORMALITY_VALUES,
            StylePreferences,
        )
        assert value in FORMALITY_VALUES
        prefs = StylePreferences(formality=value)
        assert prefs.formality == value

    @pytest.mark.parametrize(
        "value",
        ["short", "medium", "long"],
    )
    def test_all_length_values_accepted(self, value):
        from enigma_engine.core.style_preferences import (
            LENGTH_VALUES,
            StylePreferences,
        )
        assert value in LENGTH_VALUES
        prefs = StylePreferences(default_response_length=value)
        assert prefs.default_response_length == value

    def test_verbosity_values_tuple_has_three(self):
        """Pin the tuple length so adding/removing a value is explicit."""
        from enigma_engine.core.style_preferences import VERBOSITY_VALUES
        assert len(VERBOSITY_VALUES) == 3

    def test_formality_values_tuple_has_three(self):
        from enigma_engine.core.style_preferences import FORMALITY_VALUES
        assert len(FORMALITY_VALUES) == 3

    def test_length_values_tuple_has_three(self):
        from enigma_engine.core.style_preferences import LENGTH_VALUES
        assert len(LENGTH_VALUES) == 3


# =============================================================================
# Round-trip — load/save via JSON
# =============================================================================


class TestStylePreferencesRoundTrip:
    """Save then load must return identical preferences."""

    def test_roundtrip_all_fields_non_default(self, tmp_path):
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            load_style_preferences,
            save_style_preferences,
        )
        path = tmp_path / "style.json"
        prefs = StylePreferences(
            verbosity="terse",
            formality="formal",
            default_response_length="short",
            prefer_code_examples=True,
            prefer_bullet_points=True,
        )
        save_style_preferences(prefs, path)
        loaded = load_style_preferences(path)
        assert loaded == prefs

    def test_roundtrip_defaults(self, tmp_path):
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            load_style_preferences,
            save_style_preferences,
        )
        path = tmp_path / "style.json"
        prefs = StylePreferences()
        save_style_preferences(prefs, path)
        loaded = load_style_preferences(path)
        assert loaded == prefs
        assert loaded.is_default()

    def test_load_missing_file_returns_defaults(self, tmp_path):
        """Missing file falls back to defaults — no crash, no error."""
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            load_style_preferences,
        )
        path = tmp_path / "does_not_exist.json"
        loaded = load_style_preferences(path)
        assert loaded == StylePreferences()

    def test_load_corrupt_file_returns_defaults(self, tmp_path, caplog):
        """Corrupt JSON falls back to defaults with a WARNING — loud on real
        issue, never silent."""
        import logging
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            load_style_preferences,
        )
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json", encoding="utf-8")
        caplog.set_level(logging.WARNING, logger="enigma_engine.core.style_preferences")
        loaded = load_style_preferences(path)
        assert loaded == StylePreferences()
        assert any("style preferences" in r.message.lower() for r in caplog.records)

    def test_load_non_object_json_returns_defaults(self, tmp_path):
        """JSON that's not an object (e.g. a list) falls back to defaults."""
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            load_style_preferences,
        )
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        loaded = load_style_preferences(path)
        assert loaded == StylePreferences()

    def test_load_unknown_fields_ignored(self, tmp_path):
        """Future-added fields don't crash older loaders — they're silently
        dropped. Same forward-compat discipline as AIProfile.from_dict."""
        from enigma_engine.core.style_preferences import (
            load_style_preferences,
        )
        path = tmp_path / "future.json"
        path.write_text(
            '{"verbosity": "terse", "unknown_future_field": "xyz"}',
            encoding="utf-8",
        )
        loaded = load_style_preferences(path)
        assert loaded.verbosity == "terse"

    def test_load_invalid_enum_returns_defaults(self, tmp_path, caplog):
        """A file with invalid enum value falls back to defaults loudly."""
        import logging
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            load_style_preferences,
        )
        path = tmp_path / "bad_enum.json"
        path.write_text('{"verbosity": "pirate"}', encoding="utf-8")
        caplog.set_level(logging.WARNING, logger="enigma_engine.core.style_preferences")
        loaded = load_style_preferences(path)
        assert loaded == StylePreferences()
        assert any("invalid" in r.message.lower() for r in caplog.records)


# =============================================================================
# Black-box invariant — the feature must NOT touch any model file
# =============================================================================


class TestStylePreferencesBlackBoxInvariant:
    """PERSONA-2 black-box rule: style preferences live entirely in
    user-side config. Saving them must never modify any file under
    ``models/`` or any model artifact. This test gates the rule at the
    file-system level."""

    def test_save_does_not_touch_model_files(self, tmp_path):
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            save_style_preferences,
        )
        prefs_path = tmp_path / "style.json"
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        fake_model = models_dir / "test.pth"
        fake_model.write_bytes(b"fake-model-bytes")
        before_bytes = fake_model.read_bytes()
        before_mtime = fake_model.stat().st_mtime

        save_style_preferences(
            StylePreferences(verbosity="terse", prefer_bullet_points=True),
            prefs_path,
        )

        # Model file is byte-identical
        assert fake_model.read_bytes() == before_bytes
        # Model file mtime did not change
        assert fake_model.stat().st_mtime == before_mtime
        # Style preferences file was written
        assert prefs_path.exists()

    def test_save_writes_only_the_target_path(self, tmp_path):
        """No collateral writes — only the target JSON file appears."""
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            save_style_preferences,
        )
        prefs_path = tmp_path / "style.json"
        save_style_preferences(StylePreferences(formality="formal"), prefs_path)
        # Only the target file (plus possible atomic-save backup) should exist
        files = list(tmp_path.iterdir())
        # atomic_write_json may leave a `.bak` rotation — that's OK
        non_target = [
            f for f in files
            if f != prefs_path and f.suffix not in (".bak", ".tmp")
        ]
        assert not non_target, f"unexpected files written: {non_target}"


# =============================================================================
# JSON shape — explicit so the schema is observable
# =============================================================================


class TestStylePreferencesJsonShape:
    """The serialized JSON shape is part of the user-visible contract.
    Pin the field names so future schema changes are explicit."""

    def test_save_writes_all_five_fields(self, tmp_path):
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            save_style_preferences,
        )
        path = tmp_path / "style.json"
        save_style_preferences(StylePreferences(), path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data.keys()) == {
            "verbosity",
            "formality",
            "default_response_length",
            "prefer_code_examples",
            "prefer_bullet_points",
        }


# =============================================================================
# Slice 2 — render_style_preferences_block
# =============================================================================


class TestRenderStylePreferencesBlock:
    """The render function must:
    - Return ``""`` for defaults (zero-overhead skip)
    - Wrap directives in ``[USER STYLE PREFERENCES]`` / ``[/USER STYLE PREFERENCES]``
    - Only include directives for non-default fields
    - Produce a stable, human-readable + LLM-readable shape
    """

    def test_defaults_render_empty(self):
        """is_default() short-circuits — defaults produce no block."""
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            render_style_preferences_block,
        )
        assert render_style_preferences_block(StylePreferences()) == ""

    def test_block_has_open_and_close_markers(self):
        """Bracketed markers separate instructions from user input."""
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            render_style_preferences_block,
        )
        block = render_style_preferences_block(StylePreferences(verbosity="terse"))
        assert block.startswith("[USER STYLE PREFERENCES]")
        assert block.rstrip().endswith("[/USER STYLE PREFERENCES]")

    def test_terse_verbosity_directive(self):
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            render_style_preferences_block,
        )
        block = render_style_preferences_block(StylePreferences(verbosity="terse"))
        assert "terse" in block.lower()

    def test_verbose_verbosity_directive(self):
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            render_style_preferences_block,
        )
        block = render_style_preferences_block(StylePreferences(verbosity="verbose"))
        assert "thorough" in block.lower() or "verbose" in block.lower()

    def test_casual_formality_directive(self):
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            render_style_preferences_block,
        )
        block = render_style_preferences_block(StylePreferences(formality="casual"))
        assert "casual" in block.lower()

    def test_formal_formality_directive(self):
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            render_style_preferences_block,
        )
        block = render_style_preferences_block(StylePreferences(formality="formal"))
        assert "formal" in block.lower() or "professional" in block.lower()

    def test_short_length_directive(self):
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            render_style_preferences_block,
        )
        block = render_style_preferences_block(
            StylePreferences(default_response_length="short")
        )
        assert "short" in block.lower()

    def test_long_length_directive(self):
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            render_style_preferences_block,
        )
        block = render_style_preferences_block(
            StylePreferences(default_response_length="long")
        )
        assert "thorough" in block.lower() or "long" in block.lower()

    def test_code_examples_directive(self):
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            render_style_preferences_block,
        )
        block = render_style_preferences_block(
            StylePreferences(prefer_code_examples=True)
        )
        assert "code example" in block.lower() or "code examples" in block.lower()

    def test_bullet_points_directive(self):
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            render_style_preferences_block,
        )
        block = render_style_preferences_block(
            StylePreferences(prefer_bullet_points=True)
        )
        assert "bullet" in block.lower()

    def test_default_fields_produce_no_directive(self):
        """A non-default field that happens to match a default value shouldn't
        leak a directive. e.g. verbosity='normal' is default, so no verbosity
        directive should appear even when other fields are non-default."""
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            render_style_preferences_block,
        )
        # formality non-default, verbosity left at default
        block = render_style_preferences_block(
            StylePreferences(formality="formal")
        )
        # verbosity directives (terse / thorough) should NOT appear
        assert "be terse" not in block.lower()
        assert "be thorough" not in block.lower()

    def test_multiple_non_default_fields_produce_multiple_directives(self):
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            render_style_preferences_block,
        )
        block = render_style_preferences_block(
            StylePreferences(
                verbosity="terse",
                formality="formal",
                prefer_bullet_points=True,
            )
        )
        # Three non-default fields → three directive lines
        directive_lines = [
            l for l in block.split("\n") if l.startswith("- ")
        ]
        assert len(directive_lines) == 3


# =============================================================================
# Slice 2 — get_style_preferences_block_for_prompt (load + render integration)
# =============================================================================


class TestGetStylePreferencesBlockForPrompt:
    """Integration: load from disk + render in one call."""

    def test_missing_file_returns_empty(self, tmp_path):
        """No file → defaults → empty block (zero overhead for first-run)."""
        from enigma_engine.core.style_preferences import (
            get_style_preferences_block_for_prompt,
        )
        missing = tmp_path / "no_such_file.json"
        assert get_style_preferences_block_for_prompt(missing) == ""

    def test_default_file_returns_empty(self, tmp_path):
        """File exists but contains defaults → still empty block."""
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            get_style_preferences_block_for_prompt,
            save_style_preferences,
        )
        path = tmp_path / "style.json"
        save_style_preferences(StylePreferences(), path)
        assert get_style_preferences_block_for_prompt(path) == ""

    def test_customized_file_returns_block(self, tmp_path):
        """Non-default file → non-empty block with markers."""
        from enigma_engine.core.style_preferences import (
            StylePreferences,
            get_style_preferences_block_for_prompt,
            save_style_preferences,
        )
        path = tmp_path / "style.json"
        save_style_preferences(
            StylePreferences(verbosity="terse", prefer_bullet_points=True),
            path,
        )
        block = get_style_preferences_block_for_prompt(path)
        assert "[USER STYLE PREFERENCES]" in block
        assert "[/USER STYLE PREFERENCES]" in block
        assert "terse" in block.lower()
        assert "bullet" in block.lower()


# =============================================================================
# Slice 2 — injection into _prepare_chat
# =============================================================================


class TestPrepareChatStyleInjection:
    """The system prompt produced by ``_prepare_chat`` must include the
    style block when preferences are non-default, and must NOT include
    any style block when preferences are at defaults (or missing).

    This is the production seam — both GUI and API chat paths go
    through ``_prepare_chat``, so getting it right here covers both.
    """

    def _make_mixin(self):
        """Bare _ChatMixin matching tests/test_chat.py pattern."""
        from unittest.mock import MagicMock
        from enigma_engine.core.engine_chat import _ChatMixin
        obj = object.__new__(_ChatMixin)
        obj._is_gguf = False
        obj.model = MagicMock()
        obj.get_max_context_length = MagicMock(return_value=4096)
        obj.count_tokens = MagicMock(return_value=5)
        obj._history_summary = ""
        return obj

    def test_prepare_chat_injects_block_when_non_default(
            self, tmp_path, monkeypatch):
        """A non-default preferences file produces an injected block in the
        ChatContext prompt. Use monkeypatch to point at a tmp prefs file
        so the test doesn't depend on real ``data/style_preferences.json``.
        """
        from enigma_engine.core import style_preferences as sp_module
        prefs_path = tmp_path / "style.json"
        sp_module.save_style_preferences(
            sp_module.StylePreferences(verbosity="terse"),
            prefs_path,
        )
        monkeypatch.setattr(sp_module, "STYLE_PREFERENCES_PATH", prefs_path)

        mixin = self._make_mixin()
        ctx = mixin._prepare_chat("hello", system_prompt="You are helpful.")

        assert "[USER STYLE PREFERENCES]" in ctx.prompt
        assert "terse" in ctx.prompt.lower()

    def test_prepare_chat_omits_block_when_defaults(self, tmp_path, monkeypatch):
        """A default preferences file (or no file) leaves the prompt clean."""
        from enigma_engine.core import style_preferences as sp_module
        # Point at a path that does NOT exist — load falls back to defaults
        monkeypatch.setattr(
            sp_module,
            "STYLE_PREFERENCES_PATH",
            tmp_path / "no_such_file.json",
        )

        mixin = self._make_mixin()
        ctx = mixin._prepare_chat("hello", system_prompt="You are helpful.")

        assert "[USER STYLE PREFERENCES]" not in ctx.prompt

    def test_prepare_chat_omits_block_when_file_has_defaults(
            self, tmp_path, monkeypatch):
        """File exists with all defaults → still no injection."""
        from enigma_engine.core import style_preferences as sp_module
        prefs_path = tmp_path / "style.json"
        sp_module.save_style_preferences(
            sp_module.StylePreferences(),
            prefs_path,
        )
        monkeypatch.setattr(sp_module, "STYLE_PREFERENCES_PATH", prefs_path)

        mixin = self._make_mixin()
        ctx = mixin._prepare_chat("hello", system_prompt="You are helpful.")

        assert "[USER STYLE PREFERENCES]" not in ctx.prompt

    def test_prepare_chat_appends_block_after_base_system_prompt(
            self, tmp_path, monkeypatch):
        """The base system_prompt is preserved; the style block is appended
        to it, not replacing it."""
        from enigma_engine.core import style_preferences as sp_module
        prefs_path = tmp_path / "style.json"
        sp_module.save_style_preferences(
            sp_module.StylePreferences(formality="formal"),
            prefs_path,
        )
        monkeypatch.setattr(sp_module, "STYLE_PREFERENCES_PATH", prefs_path)

        mixin = self._make_mixin()
        base = "You are Enigma, a helpful AI assistant."
        ctx = mixin._prepare_chat("hello", system_prompt=base)

        # Both the base AND the style block must appear; base must come first
        assert base in ctx.prompt
        assert "[USER STYLE PREFERENCES]" in ctx.prompt
        assert ctx.prompt.index(base) < ctx.prompt.index("[USER STYLE PREFERENCES]")
