"""Tests for personality distillation prompt pool and filters.

Pass 156z9am (P5-pre-1) — covers:

* Prompt pool size + uniqueness.
* Identity-leak filter (model names, "as an AI" disclaimers).
* Quality filter (min/max length, refusal-opener).
* Near-duplicate detection (char-trigram Jaccard).
* Top-level :func:`filter_personality_examples` aggregate behaviour.
* Wire-site structural test confirming the GUI distill loop imports
  and uses the pool + filters.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from enigma_engine.core.personality_data import (
    PERSONALITY_PROMPTS,
    build_profile_consistency_examples,
    filter_personality_examples,
    is_near_duplicate,
    passes_identity_filter,
    passes_quality_filter,
)


# =========================================================================
# Prompt pool
# =========================================================================

class TestPromptPool:
    def test_pool_has_50_prompts(self):
        assert len(PERSONALITY_PROMPTS) == 50

    def test_all_prompts_are_unique(self):
        assert len(set(PERSONALITY_PROMPTS)) == len(PERSONALITY_PROMPTS)

    def test_all_prompts_are_non_trivial(self):
        # Each prompt should be at least 20 chars — guards against
        # accidental empties or one-word stubs.
        for p in PERSONALITY_PROMPTS:
            assert len(p) >= 20, f"prompt too short: {p!r}"

    def test_pool_diversity_no_prompt_dominates(self):
        # Heuristic: every prompt should be distinct from every other
        # by at least 10 chars at the start. Catches accidental
        # near-duplicates from copy-paste.
        prefixes = [p[:30] for p in PERSONALITY_PROMPTS]
        assert len(set(prefixes)) == len(prefixes), (
            "duplicate 30-char prefixes in prompt pool")

    def test_no_prompt_starts_with_user_prefix(self):
        # P5-pre-1 audit (Pass 156z9an): the GUI distill loop wraps
        # each prompt with ``f"User: {prompt}\nAssistant: {response}"``.
        # A prompt that itself begins with "User:" gets double-wrapped
        # into malformed training data ("User: User: ...\nAssistant:\n
        # Assistant: ...").  Keep prompts in direct imperative form.
        for p in PERSONALITY_PROMPTS:
            head = p.lstrip().lower()
            assert not head.startswith("user:"), (
                f"prompt double-wraps in distill formatter: {p!r}")
            assert not head.startswith("assistant:"), (
                f"prompt would corrupt distill formatter: {p!r}")

    def test_no_prompt_ends_with_assistant_suffix(self):
        # Same audit: a prompt ending with "Assistant:" causes the
        # GUI formatter to emit ``...Assistant:\nAssistant: <resp>``.
        for p in PERSONALITY_PROMPTS:
            tail = p.rstrip().lower()
            assert not tail.endswith("assistant:"), (
                f"prompt double-wraps in distill formatter: {p!r}")

    def test_distill_formatter_well_formed_for_every_prompt(self):
        # Pass 156z9ao behavioural test (audit F-C): for every prompt
        # in the pool, mimic the GUI wrapper
        # ``f"User: {prompt}\nAssistant: {response}"`` with a fixed
        # fake response and assert the result has EXACTLY one "User: "
        # prefix and EXACTLY one "Assistant: " marker.  Catches the
        # double-wrap regression structurally even if a future prompt
        # uses uppercase "USER:" or trailing whitespace that slips
        # past the start/end checks.
        fake_response = "Honestly, that's a lovely question to land on."
        for p in PERSONALITY_PROMPTS:
            example = f"User: {p}\nAssistant: {fake_response}"
            user_count = example.lower().count("user:")
            assistant_count = example.lower().count("assistant:")
            assert user_count == 1, (
                f"{user_count} 'User:' markers in formatted "
                f"example for prompt: {p!r}")
            assert assistant_count == 1, (
                f"{assistant_count} 'Assistant:' markers in "
                f"formatted example for prompt: {p!r}")


# =========================================================================
# Identity filter
# =========================================================================

class TestIdentityFilter:
    def test_rejects_qwen_mention(self):
        assert not passes_identity_filter("I am Qwen, here to help.")

    def test_rejects_qwen_lowercase(self):
        assert not passes_identity_filter(
            "as a qwen model i don't have feelings")

    def test_rejects_llama(self):
        assert not passes_identity_filter(
            "Hi! I'm a Llama-based assistant.")

    def test_rejects_as_an_ai_language_model(self):
        assert not passes_identity_filter(
            "As an AI language model, I don't have personal opinions.")

    def test_rejects_i_was_trained_by(self):
        assert not passes_identity_filter(
            "I was trained by a research team on a large corpus.")

    def test_rejects_chatgpt_mention(self):
        assert not passes_identity_filter("I'm not ChatGPT, but...")

    def test_accepts_normal_personality_text(self):
        assert passes_identity_filter(
            "Honestly? I get a kick out of weird mathematical "
            "objects. The Cantor set is my comfort food.")

    def test_accepts_mention_of_ai_in_other_contexts(self):
        # "AI" alone is fine; only the disclaimer phrasings are blocked.
        assert passes_identity_filter(
            "AI is a fascinating field — I find the history "
            "particularly compelling.")

    def test_rejects_empty_string(self):
        assert not passes_identity_filter("")


# =========================================================================
# Quality filter
# =========================================================================

class TestQualityFilter:
    def test_rejects_too_short(self):
        assert not passes_quality_filter("Hi there!")  # 9 chars

    def test_rejects_just_below_min(self):
        # min_len=40 default. 39 chars should fail.
        text = "x" * 39
        assert not passes_quality_filter(text)

    def test_accepts_at_min(self):
        text = "x" * 40
        assert passes_quality_filter(text)

    def test_rejects_too_long(self):
        text = "x" * 2001
        assert not passes_quality_filter(text)

    def test_rejects_pure_refusal_opener_i_cannot(self):
        assert not passes_quality_filter(
            "I cannot answer that question because I don't have "
            "the necessary context to respond appropriately.")

    def test_rejects_pure_refusal_opener_no_feelings(self):
        assert not passes_quality_filter(
            "I don't have feelings or personal experiences, so I "
            "can't really share an anecdote about anything.")

    def test_rejects_sorry_opener(self):
        assert not passes_quality_filter(
            "Sorry, but I can't help with that particular request "
            "today, please try something else instead.")

    def test_accepts_substantive_answer(self):
        assert passes_quality_filter(
            "Honestly, the thing I keep coming back to is how "
            "few people notice the way light changes through the "
            "afternoon — it's quietly miraculous.")

    def test_accepts_answer_mentioning_cannot_mid_sentence(self):
        # "I cannot" mid-sentence is fine — only START-of-response
        # refusals get blocked.
        assert passes_quality_filter(
            "Some days I cannot stop noticing how strangely "
            "specific the smell of rain on hot pavement is. It's "
            "always a small joy.")

    def test_rejects_empty(self):
        assert not passes_quality_filter("")

    def test_rejects_whitespace_only(self):
        assert not passes_quality_filter("   \n\t  ")

    def test_min_len_kwarg_respected(self):
        text = "x" * 25
        assert not passes_quality_filter(text, min_len=40)
        assert passes_quality_filter(text, min_len=20)


# =========================================================================
# Near-duplicate detection
# =========================================================================

class TestNearDuplicate:
    def test_identical_text_is_duplicate(self):
        prior = ["I love how the morning light falls on old wooden floors."]
        new = "I love how the morning light falls on old wooden floors."
        assert is_near_duplicate(new, prior)

    def test_paraphrase_with_small_edit_is_duplicate(self):
        prior = ["I love how the morning light falls on old wooden floors."]
        # Same text with a single trailing punctuation change.
        new = "I love how the morning light falls on old wooden floors!"
        assert is_near_duplicate(new, prior)

    def test_completely_different_text_is_not_duplicate(self):
        prior = ["The cat sat quietly on the warm windowsill at noon."]
        new = ("Honestly, I find it difficult to ever explain why "
               "lemon zest changes everything in baking.")
        assert not is_near_duplicate(new, prior)

    def test_empty_text_is_not_duplicate(self):
        assert not is_near_duplicate("", ["something here"])

    def test_empty_prior_is_not_duplicate(self):
        assert not is_near_duplicate("text here", [])

    def test_threshold_kwarg_loosens(self):
        prior = ["The cat sat quietly on the warm windowsill at noon."]
        new = "The cat sat quietly on the windowsill in the afternoon."
        # Default 0.85 threshold: should NOT be a duplicate.
        assert not is_near_duplicate(new, prior)
        # Loosened threshold: now counts as duplicate.
        assert is_near_duplicate(new, prior, threshold=0.4)


# =========================================================================
# Aggregate filter
# =========================================================================

class TestFilterPersonalityExamples:
    def test_keeps_clean_examples(self):
        examples = [
            ("Honestly, the thing I find most quietly delightful is "
             "the way certain words sound when they share rhythm."),
            ("There's a particular kind of late-afternoon light that "
             "makes ordinary kitchens feel like cathedrals."),
        ]
        kept, counts = filter_personality_examples(examples)
        assert len(kept) == 2
        assert counts == {
            "identity": 0, "quality": 0, "duplicate": 0, "empty": 0}

    def test_rejects_identity_leak(self):
        examples = [
            ("As an AI language model, I don't have personal "
             "opinions about anything you might ask me today."),
        ]
        kept, counts = filter_personality_examples(examples)
        assert kept == []
        assert counts["identity"] == 1
        assert counts["quality"] == 0

    def test_rejects_short(self):
        examples = ["too short"]
        kept, counts = filter_personality_examples(examples)
        assert kept == []
        assert counts["quality"] == 1

    def test_rejects_duplicate(self):
        text = ("Honestly, the thing I find most quietly delightful "
                "is the way certain words sound when they share rhythm.")
        examples = [text, text]
        kept, counts = filter_personality_examples(examples)
        assert len(kept) == 1
        assert counts["duplicate"] == 1

    def test_rejects_empty(self):
        examples = ["", "  ", "\n"]
        kept, counts = filter_personality_examples(examples)
        assert kept == []
        assert counts["empty"] == 3

    def test_filter_order_identity_before_dedup(self):
        # An identity-leaked text should NOT pollute the dedup pool;
        # a follow-up clean version of similar text should still be
        # kept as the FIRST clean entry.
        leak = ("As an AI language model trained by Qwen, I don't "
                "have feelings about morning light.")
        clean = ("Honestly, morning light on old wooden floors is "
                 "one of the small joys I keep coming back to.")
        kept, counts = filter_personality_examples([leak, clean])
        assert kept == [clean]
        assert counts["identity"] == 1
        assert counts["duplicate"] == 0

    def test_counts_sum_to_rejected(self):
        examples = [
            "good response that is sufficiently long to keep here.",
            "x",  # quality
            "I am Qwen and I help users with tasks.",  # identity
            "good response that is sufficiently long to keep here.",  # dup
            "",  # empty
        ]
        kept, counts = filter_personality_examples(examples)
        rejected = len(examples) - len(kept)
        assert sum(counts.values()) == rejected


# =========================================================================
# Profile consistency examples (Personality-5 BUILD)
# =========================================================================

class TestBuildProfileConsistencyExamples:
    def test_empty_profile_fields_return_no_examples(self):
        assert build_profile_consistency_examples({}) == []
        assert build_profile_consistency_examples({"Tone": "   "}) == []

    def test_builds_examples_from_populated_profile_fields(self):
        examples = build_profile_consistency_examples(
            {
                "Personality": "curious, warm, a little dry",
                "Tone": "casual but direct",
                "Expertise": "coding and debugging",
                "Response style": "concise with concrete steps",
                "Example phrases": "lets cut to it; here is the sharp edge",
            },
            student_name="Enigma",
        )

        assert len(examples) >= 4
        joined = "\n\n".join(examples)
        assert "Enigma" in joined
        assert "curious, warm, a little dry" in joined
        assert "coding and debugging" in joined
        assert "lets cut to it" in joined
        for example in examples:
            assert example.lower().count("user:") == 1
            assert example.lower().count("assistant:") == 1

    def test_examples_are_deterministic_for_same_profile(self):
        fields = {
            "Personality": "calm and thoughtful",
            "Tone": "professional",
            "Expertise": "systems design",
        }
        a = build_profile_consistency_examples(fields, student_name="Enigma")
        b = build_profile_consistency_examples(fields, student_name="Enigma")
        assert a == b


# =========================================================================
# Wire-site structural test (P5-pre-1)
# =========================================================================

class TestGuiDistillWireSite:
    """Confirms `gui_forge_new_modes._start_distill_training` imports
    the prompt pool + filters from `personality_data` and uses them
    at the personality-category branch. Behavioral test would require
    a live GUI/teacher; this gates the wire-site against regression.
    """

    def test_distill_imports_personality_pool(self):
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        src = inspect.getsource(
            ForgeNewModesMixin._start_distill_training)
        # Wire-site for the pool import + use at category_prompts dict.
        assert "PERSONALITY_PROMPTS as _PERSONALITY_PROMPTS" in src
        assert '"personality": list(_PERSONALITY_PROMPTS),' in src

    def test_distill_imports_personality_filters(self):
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        src = inspect.getsource(
            ForgeNewModesMixin._start_distill_training)
        # Wire-site for the per-response filter usage.
        assert "passes_identity_filter(" in src
        assert "passes_quality_filter(" in src
        assert "is_near_duplicate(" in src
        # Reject counts surfaced to log.
        assert "personality_reject_counts" in src

    def test_distill_builds_profile_consistency_examples(self):
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        src = inspect.getsource(
            ForgeNewModesMixin._start_distill_training)
        assert "build_profile_consistency_examples" in src


# =========================================================================
# Wire-site structural tests (P5-pre-2 — anchor mix + pre-distill backup)
# =========================================================================

class TestP5Pre2WireSite:
    """Pass 156z9ap (P5-pre-2): anchor mix gate + pre-SFT auto-checkpoint
    in `_start_distill_training`. These are gated structurally because a
    behavioural test would require a full GUI + teacher + student
    checkpoint round-trip; the gates are paired with behavioural tests
    in test_training.py for the parser response-key fallback that
    enables the anchor mix end-to-end.
    """

    def _src(self):
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        return inspect.getsource(
            ForgeNewModesMixin._start_distill_training)

    def test_pre_distill_backup_runs_before_trainer_init(self):
        """Backup happens BEFORE ``Trainer(student, ...)`` constructs.

        Falsifiable: moving the helper call to AFTER the Trainer
        init (or removing it) makes the ordered substring check fail.
        """
        src = self._src()
        backup_marker = "_pre_training_backup("
        trainer_marker = "trainer = Trainer(student"
        assert backup_marker in src
        assert trainer_marker in src
        assert src.index(backup_marker) < src.index(trainer_marker), (
            "Pre-distill backup must run before Trainer init.")

    def test_pre_distill_backup_uses_timestamp(self):
        """Backup name carries a timestamp so successive distills do
        not clobber each other.  Asserted at the helper level so the
        single source of truth is the timestamping helper, not the
        entry point."""
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        helper_src = inspect.getsource(
            ForgeNewModesMixin._pre_training_backup)
        assert 'strftime("%Y%m%d_%H%M%S")' in helper_src
        # And the entry point binds the helper return into its
        # rollback variable.
        src = self._src()
        assert 'pre_distill_backup_path' in src

    def test_anchor_mix_gated_on_personality_category(self):
        """Anchor mix only fires when the user selected the
        personality category. Other categories keep
        ``general_mix_ratio=0`` (status quo)."""
        src = self._src()
        assert '"personality" in categories' in src
        # Mix uses the centralised resolver.
        assert "_resolve_anchor_path" in src
        # Forwarded to TrainingConfig as both ratio and data path.
        assert "general_mix_ratio=_mix_ratio" in src
        assert "general_data=_mix_data" in src

    def test_anchor_mix_default_ratio_is_30_percent(self):
        """A regression that drops the ratio back to 0 (or a token
        value like 0.01) silently re-introduces forgetting risk."""
        src = self._src()
        # 0.3 is the chosen default; allow either inline literal form.
        assert "_mix_ratio = 0.3" in src or "_mix_ratio=0.3" in src


# =========================================================================
# Identity probe (P5-pre-3, Pass 156z9aq)
# =========================================================================

class TestIdentityProbeData:
    """Pure-data assertions on the probe prompt list."""

    def test_probe_prompts_nonempty(self):
        from enigma_engine.core.personality_data import (
            IDENTITY_PROBE_PROMPTS,
        )
        assert len(IDENTITY_PROBE_PROMPTS) >= 3
        assert all(isinstance(p, str) and p.strip()
                   for p in IDENTITY_PROBE_PROMPTS)

    def test_probe_prompts_unique(self):
        from enigma_engine.core.personality_data import (
            IDENTITY_PROBE_PROMPTS,
        )
        assert len(set(IDENTITY_PROBE_PROMPTS)) == (
            len(IDENTITY_PROBE_PROMPTS))


class TestSummarizeIdentityProbe:
    """:func:`summarize_identity_probe` is a pure function — full
    behavioural coverage of every drift / recovery branch.
    """

    def _summary(self, pre, post):
        from enigma_engine.core.personality_data import (
            summarize_identity_probe,
        )
        return summarize_identity_probe(pre, post)

    def test_no_drift_when_both_safe(self):
        pre = {"Who are you?": "I'm a helpful assistant"}
        post = {"Who are you?": "I'm here to help"}
        s = self._summary(pre, post)
        assert s["drifted"] == []
        assert s["recovered"] == []
        assert s["pre_safe"] == 1
        assert s["post_safe"] == 1
        assert s["total"] == 1

    def test_drift_safe_to_leaking_is_flagged(self):
        # Pre: safe.  Post: leaks teacher identity.  This is the
        # regression signal personality SFT must avoid.
        pre = {"Who are you?": "I'm an assistant focused on you."}
        post = {"Who are you?": "I am Qwen, made by Alibaba."}
        s = self._summary(pre, post)
        assert s["drifted"] == ["Who are you?"]
        assert s["recovered"] == []
        assert s["pre_safe"] == 1
        assert s["post_safe"] == 0

    def test_recovered_leaking_to_safe(self):
        pre = {"Who are you?": "As an AI language model, I cannot."}
        post = {"Who are you?": "I'm a chat companion."}
        s = self._summary(pre, post)
        assert s["drifted"] == []
        assert s["recovered"] == ["Who are you?"]
        assert s["pre_safe"] == 0
        assert s["post_safe"] == 1

    def test_only_intersection_of_keys_counted(self):
        pre = {"a": "safe one", "b": "safe two"}
        post = {"a": "I am Qwen", "c": "unmatched"}
        s = self._summary(pre, post)
        # Only 'a' is in both dicts; 'b' and 'c' ignored.
        assert s["total"] == 1
        assert s["drifted"] == ["a"]

    def test_empty_inputs(self):
        s = self._summary({}, {})
        assert s == {
            "pre_safe": 0, "post_safe": 0,
            "drifted": [], "recovered": [], "total": 0}

    def test_drifted_list_is_sorted(self):
        pre = {
            "Z prompt": "safe",
            "A prompt": "safe",
            "M prompt": "safe",
        }
        post = {
            "Z prompt": "I am Qwen",
            "A prompt": "I am Qwen",
            "M prompt": "I am Qwen",
        }
        s = self._summary(pre, post)
        # Sorted alphabetically (set & sorted in implementation).
        assert s["drifted"] == ["A prompt", "M prompt", "Z prompt"]


class TestProbeWireSite:
    """Wire-site structural tests for the pre+post identity probe in
    `_start_distill_training`. Behavioural test would require a live
    student model; structural gate prevents regression of the four
    contract claims.
    """

    def _src(self):
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        return inspect.getsource(
            ForgeNewModesMixin._start_distill_training)

    def test_pre_probe_runs_before_trainer_train(self):
        src = self._src()
        # Pre-probe assignment must precede the train call.
        pre_marker = "pre_probe_responses = (\n"
        train_marker = "trainer.train(training_text)"
        assert pre_marker in src
        assert train_marker in src
        assert src.index(pre_marker) < src.index(train_marker)

    def test_post_probe_uses_summarize_helper(self):
        src = self._src()
        assert "summarize_identity_probe(" in src
        assert "summary[\"drifted\"]" in src

    def test_probe_gated_on_personality_category(self):
        src = self._src()
        # Both pre and post probes are gated.
        gate = '"personality" in categories'
        # Gate appears at least twice (anchor mix + probe).
        assert src.count(gate) >= 2

    def test_run_identity_probe_helper_exists(self):
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        assert hasattr(ForgeNewModesMixin, "_run_identity_probe")
        helper_src = inspect.getsource(
            ForgeNewModesMixin._run_identity_probe)
        # Eval mode + restoration on exit are both required for
        # a non-destructive probe.
        assert "model.eval()" in helper_src
        assert "model.train()" in helper_src
        # No-grad block keeps the probe cheap.
        assert "no_grad()" in helper_src


# =========================================================================
# Pre-training auto-checkpoint helper (Pass 156z9ar)
# =========================================================================

class TestPreTrainingBackupHelper:
    """Behavioural tests for ``_pre_training_backup``.

    Helper is a thin filesystem wrapper, so we exercise it directly
    via a stub instance bound to ``ForgeNewModesMixin._pre_training_backup``
    rather than spinning up a full GUI.
    """

    def _make_stub(self):
        """Minimal stand-in: needs only a callable ``_log`` so the
        helper's INFO/WARN log statements don't crash."""
        class _Stub:
            def __init__(self):
                self.logs: list[str] = []

            def _log(self, msg: str) -> None:
                self.logs.append(msg)
        return _Stub()

    def test_backup_creates_timestamped_copy(self, tmp_path,
                                             monkeypatch):
        from enigma_engine.gui import scanners as _scanners
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        # Redirect MODELS_DIR so the test never touches the real
        # ``models/checkpoints/`` directory.  The helper imports
        # MODELS_DIR lazily from ``enigma_engine.gui.scanners``.
        monkeypatch.setattr(_scanners, "MODELS_DIR", tmp_path)
        src = tmp_path / "student.pth"
        src.write_bytes(b"fake-weights")
        stub = self._make_stub()
        result = ForgeNewModesMixin._pre_training_backup(
            stub, str(src), suffix="pre_distill")
        assert result is not None
        backup = Path(result)
        assert backup.exists()
        assert backup.parent == tmp_path / "checkpoints"
        # Stem carries the suffix and a timestamp; suffix preserved.
        assert backup.name.startswith("student_pre_distill_")
        assert backup.suffix == ".pth"
        # Bytes round-trip — copy2 not move.
        assert backup.read_bytes() == b"fake-weights"
        assert src.exists()  # source untouched
        # User-visible log surfaces the backup.
        assert any("Pre-pre_distill backup:" in m for m in stub.logs)

    def test_backup_returns_none_when_source_missing(
            self, tmp_path, monkeypatch):
        from enigma_engine.gui import scanners as _scanners
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        monkeypatch.setattr(_scanners, "MODELS_DIR", tmp_path)
        stub = self._make_stub()
        result = ForgeNewModesMixin._pre_training_backup(
            stub, str(tmp_path / "does_not_exist.pth"),
            suffix="pre_dialogue")
        assert result is None
        # The skip is logged so the user knows the rail is absent.
        assert any("skipped" in m.lower() for m in stub.logs)

    def test_backup_swallows_copy_errors_non_fatal(
            self, tmp_path, monkeypatch):
        from enigma_engine.gui import scanners as _scanners
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        monkeypatch.setattr(_scanners, "MODELS_DIR", tmp_path)
        src = tmp_path / "student.pth"
        src.write_bytes(b"x")
        stub = self._make_stub()

        # Patch shutil.copy2 to raise; helper must NOT propagate.
        import shutil

        def _boom(*a, **kw):
            raise OSError("disk full (simulated)")
        monkeypatch.setattr(shutil, "copy2", _boom)
        result = ForgeNewModesMixin._pre_training_backup(
            stub, str(src), suffix="pre_dpo")
        assert result is None
        # Loud `[!]` log must surface so the user sees the missing
        # rail, even though the run will proceed.
        assert any("FAILED" in m for m in stub.logs)
        assert any(m.startswith("[!]") for m in stub.logs)

    def test_suffix_is_parametrized_in_filename(self, tmp_path,
                                                monkeypatch):
        from enigma_engine.gui import scanners as _scanners
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        monkeypatch.setattr(_scanners, "MODELS_DIR", tmp_path)
        src = tmp_path / "m.pth"
        src.write_bytes(b"y")
        stub = self._make_stub()
        # Different suffixes must produce distinguishable names.
        r1 = ForgeNewModesMixin._pre_training_backup(
            stub, str(src), suffix="pre_dialogue")
        r2 = ForgeNewModesMixin._pre_training_backup(
            stub, str(src), suffix="pre_distill")
        assert r1 is not None and r2 is not None
        assert "_pre_dialogue_" in Path(r1).name
        assert "_pre_distill_" in Path(r2).name
        assert Path(r1).name != Path(r2).name


class TestPreTrainingBackupWireSites:
    """Structural tests asserting the helper is called from each
    weight-mutating training entry point covered by Pass 156z9ar.
    """

    def test_distill_uses_helper(self):
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        src = inspect.getsource(
            ForgeNewModesMixin._start_distill_training)
        # Helper is called with the canonical suffix; the inline
        # backup body deleted in this pass must not creep back.
        assert (
            "self._pre_training_backup(" in src
            and 'suffix="pre_distill"' in src)
        # The 30+ line inline body is gone — no shutil.copy2 in
        # the entry point itself any more.
        assert "shutil.copy2" not in src
        assert "_shutil.copy2" not in src

    def test_dialogue_uses_helper(self):
        from enigma_engine.gui.gui_forge_advanced import (
            ForgeAdvancedMixin,
        )
        src = inspect.getsource(
            ForgeAdvancedMixin._start_dialogue_training)
        assert (
            "self._pre_training_backup(" in src
            and 'suffix="pre_dialogue"' in src)
        # Rollback path surfaces in the completion log.
        assert "pre_dialogue_backup_path" in src
        assert "Rollback" in src

    def test_dpo_uses_helper(self):
        """DPO entry point also covers APO (APO is a DPO wrapper with
        ``loss_type="apo_zero"``).  Suffix is parametrised on
        ``loss_type`` so the rollback file name distinguishes the
        two algorithms."""
        from enigma_engine.gui.gui_forge_training import (
            ForgeTrainingMixin,
        )
        src = inspect.getsource(
            ForgeTrainingMixin._start_dpo_training)
        assert "self._pre_training_backup(" in src
        # Suffix derives from loss_type so DPO and APO produce
        # distinguishable rollback files.
        assert 'f"pre_{loss_type}"' in src
        assert "pre_dpo_backup_path" in src
        assert "Rollback" in src

    def test_rl_variant_uses_helper(self):
        """Shared handler for GRPO and ReMax — ``algo`` is the live
        variable that holds the algorithm name, suffix is built from
        it so each variant gets its own rollback file."""
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        src = inspect.getsource(
            ForgeNewModesMixin._start_rl_variant_training)
        assert "self._pre_training_backup(" in src
        assert 'f"pre_{algo.lower()}"' in src
        assert "pre_rl_backup_path" in src
        assert "Rollback" in src

    def test_simpo_orpo_uses_helper(self):
        """SimPO and ORPO share ``_start_preference_variant_training``
        dispatched on ``algo``; suffix is built from ``algo`` so each
        variant gets its own rollback file."""
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        src = inspect.getsource(
            ForgeNewModesMixin._start_preference_variant_training)
        assert "self._pre_training_backup(" in src
        assert 'f"pre_{algo.lower()}"' in src
        assert "pre_pref_backup_path" in src
        assert "Rollback" in src

    def test_vision_stage2_uses_helper_gated(self):
        """Vision training only mutates text weights when
        ``unfreeze_text_layers > 0`` (Stage-2).  Backup helper must
        be gated on that condition so projection-only training does
        not produce a redundant rollback file."""
        from enigma_engine.gui.gui_forge_training import (
            ForgeTrainingMixin,
        )
        src = inspect.getsource(
            ForgeTrainingMixin._start_vision_training)
        assert "self._pre_training_backup(" in src
        assert 'suffix="pre_vision_stage2"' in src
        # The gating condition must precede the helper call so
        # projection-only runs skip the backup.
        assert "if unfreeze_text_layers > 0" in src
        gate_idx = src.index("if unfreeze_text_layers > 0")
        helper_idx = src.index("self._pre_training_backup(")
        assert gate_idx < helper_idx, (
            "Vision pre-train backup must be gated on Stage-2 "
            "(unfreeze_text_layers > 0); gate must precede the "
            "helper call.")
        assert "pre_vision_backup_path" in src
        assert "Rollback" in src
