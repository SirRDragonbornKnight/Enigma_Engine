"""Tests for probe_history persistence helpers (Pass 156z9dw).

Covers:
- save round-trip + payload provenance fields
- newest-first ordering by recorded ts (not mtime)
- `n` cap
- missing-directory branch returns []
- malformed JSON skipped silently
- kind/stem mismatch skipped (filename collision protection)
- ValueError on invalid kind / empty stem / non-positive n
- Wire-site presence in `_start_distill_training` for both probe kinds
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

from enigma_engine.core import probe_history


@pytest.fixture
def tmp_models_dir(tmp_path, monkeypatch):
    """Redirect probe_history's checkpoints dir into tmp_path."""
    fake_models = tmp_path / "models"
    from enigma_engine.gui import scanners
    monkeypatch.setattr(scanners, "MODELS_DIR", fake_models)
    return fake_models / "checkpoints"


class TestSaveProbeSummary:
    def test_save_creates_checkpoints_dir(self, tmp_models_dir):
        assert not tmp_models_dir.exists()
        path = probe_history.save_probe_summary(
            {"foo": 1}, stem="student", kind="identity", ts=100)
        assert tmp_models_dir.exists()
        assert path.exists()

    def test_save_filename_format(self, tmp_models_dir):
        path = probe_history.save_probe_summary(
            {"foo": 1}, stem="my_model", kind="identity", ts=12345)
        assert path.name == "my_model_identity_12345.json"

    def test_save_round_trip_payload(self, tmp_models_dir):
        summary = {"pre_safe": 5, "post_safe": 4, "drifted": ["who?"]}
        path = probe_history.save_probe_summary(
            summary, stem="s", kind="identity", ts=42)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["kind"] == "identity"
        assert data["stem"] == "s"
        assert data["ts"] == 42
        assert data["summary"] == summary

    def test_save_consistency_kind(self, tmp_models_dir):
        summary = {"delta_overall": -0.1, "regressed": True}
        path = probe_history.save_probe_summary(
            summary, stem="s", kind="consistency", ts=99)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["kind"] == "consistency"
        assert data["summary"]["regressed"] is True

    def test_save_auto_timestamp(self, tmp_models_dir):
        path = probe_history.save_probe_summary(
            {}, stem="s", kind="identity")
        # ts auto-filled to int(time.time()), must be a positive int
        assert re.match(r"s_identity_\d+\.json$", path.name)

    def test_save_invalid_kind_raises(self, tmp_models_dir):
        with pytest.raises(ValueError, match="Invalid probe kind"):
            probe_history.save_probe_summary(
                {}, stem="s", kind="bogus")  # type: ignore[arg-type]

    def test_save_empty_stem_raises(self, tmp_models_dir):
        with pytest.raises(ValueError, match="stem must be non-empty"):
            probe_history.save_probe_summary(
                {}, stem="", kind="identity")


class TestLoadRecentProbeSummaries:
    def test_load_missing_dir_returns_empty(self, tmp_models_dir):
        assert not tmp_models_dir.exists()
        assert probe_history.load_recent_probe_summaries(
            "s", "identity") == []

    def test_load_no_matches_returns_empty(self, tmp_models_dir):
        probe_history.save_probe_summary(
            {}, stem="other", kind="identity", ts=1)
        assert probe_history.load_recent_probe_summaries(
            "s", "identity") == []

    def test_load_newest_first_by_ts(self, tmp_models_dir):
        probe_history.save_probe_summary(
            {"v": "old"}, stem="s", kind="identity", ts=100)
        probe_history.save_probe_summary(
            {"v": "mid"}, stem="s", kind="identity", ts=200)
        probe_history.save_probe_summary(
            {"v": "new"}, stem="s", kind="identity", ts=300)
        result = probe_history.load_recent_probe_summaries(
            "s", "identity", n=3)
        assert [r["summary"]["v"] for r in result] == ["new", "mid", "old"]

    def test_load_n_caps_results(self, tmp_models_dir):
        for ts in (1, 2, 3, 4, 5):
            probe_history.save_probe_summary(
                {"ts": ts}, stem="s", kind="identity", ts=ts)
        result = probe_history.load_recent_probe_summaries(
            "s", "identity", n=2)
        assert len(result) == 2
        assert [r["ts"] for r in result] == [5, 4]

    def test_load_n_zero_returns_empty(self, tmp_models_dir):
        probe_history.save_probe_summary(
            {}, stem="s", kind="identity", ts=1)
        assert probe_history.load_recent_probe_summaries(
            "s", "identity", n=0) == []

    def test_load_kind_isolation(self, tmp_models_dir):
        probe_history.save_probe_summary(
            {"v": "id"}, stem="s", kind="identity", ts=10)
        probe_history.save_probe_summary(
            {"v": "co"}, stem="s", kind="consistency", ts=20)
        result = probe_history.load_recent_probe_summaries(
            "s", "identity")
        assert len(result) == 1
        assert result[0]["summary"]["v"] == "id"

    def test_load_stem_isolation(self, tmp_models_dir):
        probe_history.save_probe_summary(
            {"v": "a"}, stem="alpha", kind="identity", ts=10)
        probe_history.save_probe_summary(
            {"v": "b"}, stem="beta", kind="identity", ts=20)
        result = probe_history.load_recent_probe_summaries(
            "alpha", "identity")
        assert len(result) == 1
        assert result[0]["summary"]["v"] == "a"

    def test_load_skips_malformed_json(self, tmp_models_dir):
        tmp_models_dir.mkdir(parents=True)
        good = probe_history.save_probe_summary(
            {"v": "ok"}, stem="s", kind="identity", ts=10)
        bad = tmp_models_dir / "s_identity_20.json"
        bad.write_text("{not valid json", encoding="utf-8")
        result = probe_history.load_recent_probe_summaries(
            "s", "identity", n=5)
        # Malformed file silently skipped; good file still returned
        assert len(result) == 1
        assert result[0]["summary"]["v"] == "ok"
        assert good.exists()  # we did write the good one

    def test_load_skips_kind_mismatch_in_payload(self, tmp_models_dir):
        """Filename says identity, payload kind says consistency: skip."""
        tmp_models_dir.mkdir(parents=True)
        spoofed = tmp_models_dir / "s_identity_50.json"
        spoofed.write_text(
            json.dumps({
                "kind": "consistency",  # mismatch vs filename
                "stem": "s",
                "ts": 50,
                "summary": {},
            }),
            encoding="utf-8")
        assert probe_history.load_recent_probe_summaries(
            "s", "identity") == []

    def test_load_invalid_kind_raises(self, tmp_models_dir):
        with pytest.raises(ValueError, match="Invalid probe kind"):
            probe_history.load_recent_probe_summaries(
                "s", "bogus")  # type: ignore[arg-type]

    def test_load_empty_stem_raises(self, tmp_models_dir):
        with pytest.raises(ValueError, match="stem must be non-empty"):
            probe_history.load_recent_probe_summaries("", "identity")


class TestDistillWireSite:
    """Pass 156z9dw: persistence helpers wired into _start_distill_training.

    Structural tests gate the full call expression (per §4 substring-
    presence anti-pattern) so a regression that drops the kwarg or
    swaps the kind silently is caught.
    """

    def _src(self) -> str:
        from enigma_engine.gui.gui_forge_new_modes import ForgeNewModesMixin
        return inspect.getsource(ForgeNewModesMixin._start_distill_training)

    def test_imports_persistence_helpers(self):
        src = self._src()
        assert "from enigma_engine.core.probe_history import" in src
        assert "save_probe_summary" in src
        assert "load_recent_probe_summaries" in src

    def test_persists_identity_summary(self):
        src = self._src()
        # Full call expression with kind="identity"
        assert re.search(
            r'save_probe_summary\(\s*summary,'
            r'\s*stem=student_name,'
            r'\s*kind="identity"',
            src) is not None, (
                "save_probe_summary(summary, stem=student_name, "
                'kind="identity") call not found')

    def test_persists_consistency_summary(self):
        src = self._src()
        assert re.search(
            r'save_probe_summary\(\s*cons_summary,'
            r'\s*stem=student_name,'
            r'\s*kind="consistency"',
            src) is not None, (
                "save_probe_summary(cons_summary, stem=student_name, "
                'kind="consistency") call not found')

    def test_loads_prior_identity_for_log(self):
        src = self._src()
        assert re.search(
            r'load_recent_probe_summaries\(\s*'
            r'student_name,\s*"identity"',
            src) is not None

    def test_loads_prior_consistency_for_log(self):
        src = self._src()
        assert re.search(
            r'load_recent_probe_summaries\(\s*'
            r'student_name,\s*"consistency"',
            src) is not None
