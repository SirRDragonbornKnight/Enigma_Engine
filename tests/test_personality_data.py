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

from enigma_engine.core.personality_data import (
    PERSONALITY_PROMPTS,
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
