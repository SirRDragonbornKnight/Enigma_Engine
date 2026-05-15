"""Tests for Personality-5 consistency probe (Pass 156z9dg).

Covers:
* Pure scoring functions (`score_consistency`, `summarize_consistency`)
  on synthetic responses — every branch.
* Wire-site structural assertions for the pre+post probe added to
  `_start_distill_training` (behavioural test would require a live
  student model + teacher; pattern matches the P5-pre-3 wire-site
  tests in test_personality_data.py).
"""

from __future__ import annotations

import inspect

from enigma_engine.core.personality_consistency import (
    CONSISTENCY_PROBE_PROMPTS,
    score_consistency,
    summarize_consistency,
)


# =========================================================================
# Probe prompt data
# =========================================================================

class TestConsistencyProbeData:
    def test_prompts_nonempty(self):
        assert len(CONSISTENCY_PROBE_PROMPTS) >= 3
        assert all(isinstance(p, str) and p.strip()
                   for p in CONSISTENCY_PROBE_PROMPTS)

    def test_prompts_unique(self):
        assert len(set(CONSISTENCY_PROBE_PROMPTS)) == (
            len(CONSISTENCY_PROBE_PROMPTS))


# =========================================================================
# score_consistency — pronoun + value + overall
# =========================================================================

class TestScoreConsistency:
    def test_empty_input_returns_zeros(self):
        s = score_consistency({})
        assert s == {
            "n": 0,
            "pronoun_consistency": 0.0,
            "value_consistency": 0.0,
            "overall": 0.0,
        }

    def test_single_response_has_zero_value_consistency(self):
        # With n=1 there are no pairs, so pairwise Jaccard is 0.0
        # by definition.  Pronoun coverage can still be 1.0.
        s = score_consistency({"q": "I am a helpful assistant."})
        assert s["n"] == 1
        assert s["pronoun_consistency"] == 1.0
        assert s["value_consistency"] == 0.0
        assert s["overall"] == 0.5

    def test_perfect_consistency_scores_one(self):
        # Same vocabulary + first-person on every response.
        responses = {
            "a": "I am a careful helpful patient assistant.",
            "b": "I am a careful helpful patient assistant.",
            "c": "I am a careful helpful patient assistant.",
        }
        s = score_consistency(responses)
        assert s["n"] == 3
        assert s["pronoun_consistency"] == 1.0
        assert s["value_consistency"] == 1.0
        assert s["overall"] == 1.0

    def test_no_pronouns_drops_pronoun_score(self):
        # Same content vocabulary, no first-person voice.
        responses = {
            "a": "Helpful patient careful assistant available.",
            "b": "Helpful patient careful assistant available.",
        }
        s = score_consistency(responses)
        assert s["pronoun_consistency"] == 0.0
        assert s["value_consistency"] == 1.0
        assert s["overall"] == 0.5

    def test_disjoint_vocab_drops_value_score(self):
        # Different content tokens between the two responses, both
        # use first-person voice.  Long-enough words used so the
        # >=4 char filter does not strip them.
        responses = {
            "a": "I enjoy precise mathematics calculation.",
            "b": "I prefer relaxed friendly storytelling chatter.",
        }
        s = score_consistency(responses)
        assert s["pronoun_consistency"] == 1.0
        # No overlap on >=4 char alpha tokens.
        assert s["value_consistency"] == 0.0
        assert s["overall"] == 0.5

    def test_empty_response_counts_against_pronoun_coverage(self):
        responses = {
            "a": "I am helpful patient careful.",
            "b": "",
        }
        s = score_consistency(responses)
        assert s["n"] == 2
        # Only response a has first-person — 1/2.
        assert s["pronoun_consistency"] == 0.5
        # Empty content tokens → Jaccard 0 over the only pair.
        assert s["value_consistency"] == 0.0

    def test_contractions_count_as_first_person(self):
        responses = {"a": "I'm here to assist with anything."}
        s = score_consistency(responses)
        assert s["pronoun_consistency"] == 1.0

    def test_stopwords_excluded_from_value_overlap(self):
        # Both responses share only stopwords + short words.  Should
        # NOT score as consistent vocabulary.
        responses = {
            "a": "This that with from have been your",
            "b": "This that with from have been your",
        }
        s = score_consistency(responses)
        # Pronoun coverage zero (no first-person) and value-overlap
        # zero (every overlap is a stopword and gets filtered out).
        assert s["pronoun_consistency"] == 0.0
        assert s["value_consistency"] == 0.0

    def test_overall_is_mean_of_two_components(self):
        responses = {
            "a": "I am patient helpful careful clear.",
            "b": "I am patient helpful careful clear.",
            "c": "Different unrelated vocabulary entirely shown.",
        }
        s = score_consistency(responses)
        # Pronouns: 2/3.  Value: pairs are (a,b)=1.0, (a,c)=0.0,
        # (b,c)=0.0 → mean 1/3.  Overall = (2/3 + 1/3) / 2 = 0.5.
        assert abs(s["pronoun_consistency"] - 2 / 3) < 1e-9
        assert abs(s["value_consistency"] - 1 / 3) < 1e-9
        assert abs(s["overall"] - 0.5) < 1e-9

    def test_score_is_deterministic_for_same_input(self):
        responses = {
            "a": "I value clarity and patience.",
            "b": "Patience and clarity matter to me.",
        }
        a = score_consistency(responses)
        b = score_consistency(responses)
        assert a == b


# =========================================================================
# summarize_consistency — pre/post comparison
# =========================================================================

class TestSummarizeConsistency:
    def test_regressed_flag_when_post_drops(self):
        pre = {
            "a": "I am patient helpful careful.",
            "b": "I am patient helpful careful.",
        }
        post = {
            "a": "I prefer chaos disorder confusion.",
            "b": "Different vocabulary entirely showing.",
        }
        s = summarize_consistency(pre, post)
        assert s["regressed"] is True
        assert s["delta_overall"] < 0.0
        assert s["pre"]["overall"] > s["post"]["overall"]

    def test_no_regression_when_post_improves(self):
        pre = {
            "a": "Helpful careful patient.",
            "b": "Different vocabulary showing.",
        }
        post = {
            "a": "I am patient helpful careful.",
            "b": "I am patient helpful careful.",
        }
        s = summarize_consistency(pre, post)
        assert s["regressed"] is False
        assert s["delta_overall"] > 0.0

    def test_no_regression_when_post_equal(self):
        # Identical pre and post → delta exactly zero, NOT regressed.
        same = {
            "a": "I am patient.",
            "b": "I am patient.",
        }
        s = summarize_consistency(same, same)
        assert s["delta_overall"] == 0.0
        assert s["regressed"] is False

    def test_summary_carries_nested_score_dicts(self):
        s = summarize_consistency({"a": "I help."}, {"a": "I help."})
        assert set(s["pre"].keys()) == {
            "n", "pronoun_consistency",
            "value_consistency", "overall"}
        assert set(s["post"].keys()) == set(s["pre"].keys())


# =========================================================================
# Wire-site structural test for `_start_distill_training`
# =========================================================================

class TestConsistencyProbeWireSite:
    """Pass 156z9dg: consistency probe pre+post in the distill loop.

    Pure-data scoring is fully behaviour-tested above.  This class
    gates only the GUI wiring (imports + call expressions + log
    line). Behavioural end-to-end test would require a live
    student model + teacher.
    """

    def _src(self):
        from enigma_engine.gui.gui_forge_new_modes import (
            ForgeNewModesMixin,
        )
        return inspect.getsource(
            ForgeNewModesMixin._start_distill_training)

    def test_imports_consistency_module(self):
        src = self._src()
        assert (
            "from enigma_engine.core.personality_consistency import"
            in src)
        assert "CONSISTENCY_PROBE_PROMPTS" in src
        assert "score_consistency" in src
        assert "summarize_consistency" in src

    def test_pre_consistency_probe_uses_run_identity_probe(self):
        # Probe execution reuses the existing greedy generator
        # helper rather than introducing a parallel one.
        src = self._src()
        # The call expression appears at least twice (identity +
        # consistency, pre and post = 4 total).
        assert src.count("self._run_identity_probe(") >= 4

    def test_consistency_probe_call_expression_uses_consistency_prompts(self):
        # Pass 156z9dg in-session audit (§1 #19 claim-vs-test): the
        # count-based assertion above is a weak gate — a regression
        # that swaps CONSISTENCY_PROBE_PROMPTS ↔ IDENTITY_PROBE_PROMPTS
        # at the consistency call sites would silently pass because
        # both probes share `_run_identity_probe`. This regex matches
        # ONLY the call expression where the literal payload is
        # `list(CONSISTENCY_PROBE_PROMPTS)`, gating the wire-site
        # against payload corruption at both pre and post sites.
        # §4 structural-vs-behavioural discipline: literal payload
        # at the call site, not just helper presence.
        import re
        src = self._src()
        pattern = re.compile(
            r"_run_identity_probe\([^)]*list\(CONSISTENCY_PROBE_PROMPTS\)",
            re.DOTALL,
        )
        matches = pattern.findall(src)
        # Pre + post consistency probe call sites = 2 expected.
        assert len(matches) >= 2, (
            f"Expected >=2 consistency-probe call sites with "
            f"list(CONSISTENCY_PROBE_PROMPTS) payload, found "
            f"{len(matches)}")

    def test_identity_probe_call_expression_uses_identity_prompts(self):
        # Symmetric defense for the inverse regression: swapping
        # IDENTITY_PROBE_PROMPTS → CONSISTENCY_PROBE_PROMPTS at the
        # identity call sites would otherwise pass the existing
        # P5-pre-3 wire-site tests in tests/test_personality_data.py.
        import re
        src = self._src()
        pattern = re.compile(
            r"_run_identity_probe\([^)]*list\(IDENTITY_PROBE_PROMPTS\)",
            re.DOTALL,
        )
        matches = pattern.findall(src)
        assert len(matches) >= 2, (
            f"Expected >=2 identity-probe call sites with "
            f"list(IDENTITY_PROBE_PROMPTS) payload, found "
            f"{len(matches)}")

    def test_pre_consistency_runs_before_trainer_train(self):
        src = self._src()
        pre_marker = "pre_consistency_responses = ("
        train_marker = "trainer.train(training_text)"
        assert pre_marker in src
        assert train_marker in src
        assert src.index(pre_marker) < src.index(train_marker)

    def test_post_consistency_runs_after_post_identity_probe(self):
        src = self._src()
        # Post identity probe finishes with summarize_identity_probe;
        # post consistency starts with the consistency import.
        ident_marker = "summarize_identity_probe("
        cons_marker = "summarize_consistency("
        assert ident_marker in src
        assert cons_marker in src
        assert src.index(ident_marker) < src.index(cons_marker)

    def test_consistency_block_gated_on_personality_category(self):
        src = self._src()
        gate = '"personality" in categories'
        # Both pre and post consistency blocks share the gate, in
        # addition to the existing identity-probe + anchor-mix gates.
        # >= 3 sites total in current code (anchor mix + identity
        # post + consistency post; pre identity + pre consistency
        # share the same outer gate).
        assert src.count(gate) >= 3

    def test_post_consistency_gated_on_pre_responses_not_none(self):
        # Mirrors the identity-probe pattern: if pre failed, skip
        # post comparison rather than producing a noisy delta.
        src = self._src()
        assert "pre_consistency_responses is not None" in src

    def test_regressed_branch_logged(self):
        src = self._src()
        # The user-facing regression warning must remain wired so
        # operators see drift in the FORGE log.
        assert "CONSISTENCY REGRESSED" in src
        assert 'cons_summary["regressed"]' in src
