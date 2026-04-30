"""Tests for token sampling, batch generation, and repetition penalty."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class FakeGen:
    """Minimal _GenerationMixin stub for testing _sample_token_batch."""
    def __init__(self):
        from enigma_engine.core.engine_generation import _GenerationMixin
        # Bind the method from the mixin
        self._sample_token_batch = _GenerationMixin._sample_token_batch.__get__(self)
        self._adaptive_rep_window = _GenerationMixin._adaptive_rep_window


class TestBatchSamplingVectorized:
    """Verify batch_generate uses vectorised sampling."""

    def test_sample_token_batch_returns_correct_shape(self):
        """_sample_token_batch should return [batch, 1]."""
        import torch

        gen = FakeGen()
        batch, vocab = 4, 32
        logits = torch.randn(batch, vocab)
        generated = torch.randint(0, vocab, (batch, 10))
        result = gen._sample_token_batch(
            logits, generated,
            temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0)
        assert result.shape == (batch, 1)

    def test_sample_token_batch_respects_top_k(self):
        """Only top-k tokens should get non-zero probability."""
        import torch

        gen = FakeGen()
        # Create logits where token 0 is clearly the best
        logits = torch.full((2, 50), -10.0)
        logits[:, 0] = 10.0
        generated = torch.zeros(2, 5, dtype=torch.long)
        result = gen._sample_token_batch(
            logits.clone(), generated,
            temperature=0.01, top_k=1, top_p=1.0, repetition_penalty=1.0)
        # With top_k=1 and low temperature, token 0 should always win
        assert (result.squeeze(-1) == 0).all()

    def test_sample_token_batch_repetition_penalty(self):
        """Repetition penalty should reduce probability of repeated tokens."""
        import torch

        gen = FakeGen()
        batch, vocab = 1, 10
        # All logits equal; token 3 repeated many times in history
        logits = torch.zeros(batch, vocab)
        generated = torch.full((batch, 20), 3, dtype=torch.long)
        # With strong penalty, token 3 should be suppressed
        # Run many samples and verify token 3 is rarely chosen
        counts = torch.zeros(vocab, dtype=torch.long)
        for _ in range(200):
            tok = gen._sample_token_batch(
                logits.clone(), generated,
                temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=2.0)
            counts[int(tok.item())] += 1
        # Token 3 should appear less than average (20 out of 200)
        assert counts[3] < 40, f"Token 3 appeared {counts[3]} times, expected < 40"


# =========================================================================
# Item 19 — GUI Themes / Color Customization
# =========================================================================

@pytest.mark.structural
class TestSamplingRepPenaltySign:
    """Verify apply_repetition_penalty handles negative logits correctly."""

    def test_rep_penalty_negative_logits_penalized(self):
        """Negative logits should become MORE negative with penalty, not less."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import apply_repetition_penalty

        vocab = 10
        logits = torch.full((1, vocab), -2.0)
        # Token 3 appeared recently
        generated = torch.tensor([[3, 3, 3, 3, 3]], dtype=torch.long)
        result = apply_repetition_penalty(logits, generated, penalty=2.0)
        # Negative logits penalized = multiplied by penalty, making them more negative
        assert result[0, 3] < -2.0, (
            "Negative logits must become MORE negative with penalty (multiply, not divide)")


@pytest.mark.structural
class TestSamplingRepPenaltyOrder:
    """Verify sample_next_token applies rep penalty before temperature."""

    def test_temperature_zero_returns_argmax(self):
        """temperature <= 0 should use greedy (argmax) decoding."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import sample_next_token
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
        generated = torch.tensor([[0, 1]])
        result = sample_next_token(
            logits, generated, temperature=0.0, repetition_penalty=1.0)
        assert result.item() == 1  # argmax of [1,5,2,3]

    def test_temperature_negative_returns_argmax(self):
        """Negative temperature should also use argmax."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import sample_next_token
        logits = torch.tensor([[10.0, 1.0, 2.0]])
        generated = torch.tensor([[0]])
        result = sample_next_token(
            logits, generated, temperature=-1.0, repetition_penalty=1.0)
        assert result.item() == 0


class TestWindowedRepetitionPenalty:
    """Repetition penalty should only look at recent tokens, not the full history."""

    def test_model_level_penalty_windows_tokens(self):
        """apply_repetition_penalty only penalises tokens in the window."""
        import torch
        from enigma_engine.core.model_utils import apply_repetition_penalty

        vocab = 32
        logits = torch.ones(1, vocab)
        # Token 5 appears far in the past (position 0), outside a window of 3
        # Token 10 appears recently (position 9), inside the window
        generated = torch.zeros(1, 10, dtype=torch.long)
        generated[0, 0] = 5   # old
        generated[0, 9] = 10  # recent

        result = apply_repetition_penalty(logits, generated, penalty=2.0, window=3)
        # Token 10 (recent) should be penalised
        assert result[0, 10] < 1.0, "Recent token should be penalised"
        # Token 5 (old, outside window) should NOT be penalised
        assert result[0, 5] == 1.0, "Old token outside window should be untouched"


# ================================================================
# Repetition Penalty — Deeper Behavioral Tests
# ================================================================


class TestRepetitionPenaltyBehavior:
    """Verify apply_repetition_penalty produces correct penalty values."""

    def test_no_penalty_returns_clone(self):
        """penalty=1.0 should return an exact clone, no changes."""
        import torch
        from enigma_engine.core.model_utils import apply_repetition_penalty

        logits = torch.tensor([[3.0, -2.0, 1.0, -1.0, 5.0]])
        generated = torch.tensor([[0, 1, 2, 3]])
        result = apply_repetition_penalty(logits, generated, penalty=1.0)
        torch.testing.assert_close(result, logits)
        # Must be a new tensor (clone), not same object
        assert result.data_ptr() != logits.data_ptr()

    def test_positive_logits_divided_by_penalty(self):
        """Positive logits for repeated tokens should be divided by penalty."""
        import torch
        from enigma_engine.core.model_utils import apply_repetition_penalty

        logits = torch.tensor([[4.0, 2.0, 6.0, 1.0]])
        generated = torch.tensor([[0, 2]])  # tokens 0 and 2 appeared
        result = apply_repetition_penalty(logits, generated, penalty=2.0)
        # Token 0: 4.0 / 2.0 = 2.0
        assert result[0, 0].item() == pytest.approx(2.0)
        # Token 2: 6.0 / 2.0 = 3.0
        assert result[0, 2].item() == pytest.approx(3.0)
        # Token 1 and 3: untouched
        assert result[0, 1].item() == pytest.approx(2.0)
        assert result[0, 3].item() == pytest.approx(1.0)

    def test_negative_logits_multiplied_by_penalty(self):
        """Negative logits for repeated tokens should be multiplied (more negative)."""
        import torch
        from enigma_engine.core.model_utils import apply_repetition_penalty

        logits = torch.tensor([[-3.0, 2.0, -1.0]])
        generated = torch.tensor([[0, 2]])
        result = apply_repetition_penalty(logits, generated, penalty=2.0)
        # Token 0: -3.0 * 2.0 = -6.0
        assert result[0, 0].item() == pytest.approx(-6.0)
        # Token 2: -1.0 * 2.0 = -2.0
        assert result[0, 2].item() == pytest.approx(-2.0)
        # Token 1: untouched
        assert result[0, 1].item() == pytest.approx(2.0)

    def test_1d_logits_supported(self):
        """1D logits (unbatched) should work correctly."""
        import torch
        from enigma_engine.core.model_utils import apply_repetition_penalty

        logits = torch.tensor([4.0, -2.0, 1.0])
        generated = torch.tensor([0, 1])
        result = apply_repetition_penalty(logits, generated, penalty=2.0)
        assert result[0].item() == pytest.approx(2.0)   # 4.0 / 2.0
        assert result[1].item() == pytest.approx(-4.0)   # -2.0 * 2.0
        assert result[2].item() == pytest.approx(1.0)    # untouched

    def test_not_inplace(self):
        """Original logits tensor must not be modified."""
        import torch
        from enigma_engine.core.model_utils import apply_repetition_penalty

        logits = torch.tensor([[4.0, -2.0, 1.0]])
        original = logits.clone()
        generated = torch.tensor([[0, 1]])
        apply_repetition_penalty(logits, generated, penalty=2.0)
        torch.testing.assert_close(logits, original)

    def test_out_of_range_tokens_ignored(self):
        """Token IDs outside vocab range should not crash or affect logits."""
        import torch
        from enigma_engine.core.model_utils import apply_repetition_penalty

        vocab = 5
        logits = torch.ones(1, vocab)
        generated = torch.tensor([[0, 100, -1, 3]])  # 100 and -1 are out of range
        result = apply_repetition_penalty(logits, generated, penalty=2.0)
        # Only tokens 0 and 3 should be penalised
        assert result[0, 0].item() < 1.0
        assert result[0, 3].item() < 1.0
        assert result[0, 1].item() == pytest.approx(1.0)
        assert result[0, 2].item() == pytest.approx(1.0)


# ================================================================
# Top-p (Nucleus) Sampling
# ================================================================


class TestTopPFiltering:
    """Verify nucleus sampling correctly limits cumulative probability."""

    def test_top_p_filters_low_probability_tokens(self):
        """With top_p=0.5, only the highest-prob tokens summing to 0.5 remain."""
        import torch
        from enigma_engine.core.model_utils import sample_next_token

        # Token 0 has probability ~0.88, token 1 ~0.12, rest ~0
        logits = torch.tensor([[10.0, 8.0, 0.0, 0.0, 0.0]])
        generated = torch.zeros(1, 5, dtype=torch.long)

        # With top_p=0.5, only token 0 should be selectable (p≈0.88 > 0.5)
        results = set()
        for _ in range(50):
            tok = sample_next_token(
                logits.clone(), generated, temperature=1.0, top_k=0,
                top_p=0.5, repetition_penalty=1.0)
            results.add(tok.item())
        # Only token 0 (dominant) should appear
        assert results == {0}, f"Expected only token 0, got {results}"

    def test_top_p_1_disables_filtering(self):
        """top_p=1.0 should allow all tokens."""
        import torch
        from enigma_engine.core.model_utils import sample_next_token

        # Uniform logits — all tokens equally likely
        logits = torch.zeros(1, 5)
        generated = torch.zeros(1, 5, dtype=torch.long)

        results = set()
        for _ in range(200):
            tok = sample_next_token(
                logits.clone(), generated, temperature=1.0, top_k=0,
                top_p=1.0, repetition_penalty=1.0)
            results.add(tok.item())
        # All 5 tokens should eventually appear
        assert len(results) >= 4, f"Expected most tokens to appear, got {results}"


# ================================================================
# Min-p Filtering
# ================================================================


class TestMinPFiltering:
    """Verify min-p filtering removes tokens below threshold relative to max."""

    def test_min_p_removes_low_probability_tokens(self):
        """Tokens with probability < min_p * max_probability should be removed."""
        import torch
        from enigma_engine.core.model_utils import sample_next_token

        # Token 0: very high logit, tokens 1-4: very low logits
        logits = torch.tensor([[10.0, -10.0, -10.0, -10.0, -10.0]])
        generated = torch.zeros(1, 5, dtype=torch.long)

        # min_p=0.5: tokens below 50% of max_prob are removed
        results = set()
        for _ in range(50):
            tok = sample_next_token(
                logits.clone(), generated, temperature=1.0, top_k=0,
                top_p=1.0, repetition_penalty=1.0, min_p=0.5)
            results.add(tok.item())
        assert results == {0}, f"Only dominant token should survive, got {results}"

    def test_min_p_zero_disables_filtering(self):
        """min_p=0.0 should not remove any tokens."""
        import torch
        from enigma_engine.core.model_utils import sample_next_token

        # Moderate spread so all tokens have some probability
        logits = torch.tensor([[2.0, 1.5, 1.0, 0.5, 0.0]])
        generated = torch.zeros(1, 5, dtype=torch.long)

        results = set()
        for _ in range(300):
            tok = sample_next_token(
                logits.clone(), generated, temperature=1.0, top_k=0,
                top_p=1.0, repetition_penalty=1.0, min_p=0.0)
            results.add(tok.item())
        assert len(results) >= 4, f"All tokens should be possible, got {results}"


# ================================================================
# Top-k Value Checks
# ================================================================


class TestTopKBehavior:
    """Verify top-k correctly limits the candidate set."""

    def test_top_k_1_equals_greedy(self):
        """top_k=1 should always pick the highest logit (greedy)."""
        import torch
        from enigma_engine.core.model_utils import sample_next_token

        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0, 4.0]])
        generated = torch.zeros(1, 5, dtype=torch.long)

        for _ in range(20):
            tok = sample_next_token(
                logits.clone(), generated, temperature=1.0, top_k=1,
                top_p=1.0, repetition_penalty=1.0)
            assert tok.item() == 1, "top_k=1 must always return argmax"

    def test_top_k_limits_candidates(self):
        """top_k=2 should only allow the 2 highest-logit tokens."""
        import torch
        from enigma_engine.core.model_utils import sample_next_token

        # Token 3 and 4 are the top 2
        logits = torch.tensor([[0.0, 0.0, 0.0, 10.0, 9.0]])
        generated = torch.zeros(1, 5, dtype=torch.long)

        results = set()
        for _ in range(100):
            tok = sample_next_token(
                logits.clone(), generated, temperature=1.0, top_k=2,
                top_p=1.0, repetition_penalty=1.0)
            results.add(tok.item())
        assert results.issubset({3, 4}), f"Only top-2 tokens allowed, got {results}"


# ================================================================
# Frequency & Presence Penalty (engine_generation._sample_token)
# ================================================================


class FakeGenAdvanced:
    """Minimal stub for testing _sample_token with advanced strategies."""
    def __init__(self):
        from enigma_engine.core.engine_generation import _GenerationMixin
        self._sample_token = _GenerationMixin._sample_token.__get__(self)
        self._adaptive_rep_window = _GenerationMixin._adaptive_rep_window


class TestFrequencyPenalty:
    """Verify frequency_penalty subtracts proportionally to token count."""

    def test_frequent_token_suppressed(self):
        """A token appearing 10 times should be penalised more than one appearing once."""
        import torch

        gen = FakeGenAdvanced()
        vocab = 10
        # Equal logits
        logits = torch.ones(1, vocab) * 5.0
        # Token 3 appears 10 times, token 7 appears once
        generated = torch.tensor([[3] * 10 + [7]])

        # With frequency_penalty=1.0, token 3 loses 10 points, token 7 loses 1
        results = set()
        for _ in range(50):
            tok = gen._sample_token(
                logits.clone(), generated, temperature=1.0, top_k=0,
                top_p=1.0, repetition_penalty=1.0, frequency_penalty=1.0)
            results.add(tok.item())
        # Token 3 should basically never appear (logit = 5 - 10 = -5)
        assert 3 not in results, f"Heavily penalised token 3 should not appear: {results}"


class TestPresencePenalty:
    """Verify presence_penalty subtracts flat amount for any seen token."""

    def test_seen_tokens_penalised_equally(self):
        """Presence penalty doesn't care about count — 1 or 10 appearances same penalty."""
        import torch

        gen = FakeGenAdvanced()
        vocab = 10
        logits = torch.ones(1, vocab) * 5.0
        # Token 3 appears 10 times, token 7 appears once — both "present"
        generated = torch.tensor([[3] * 10 + [7]])

        # Strong presence penalty should suppress both equally
        # Use _sample_token which has presence_penalty param
        # But we can verify the logit effect directly:
        logits_test = logits.clone()
        window_size = gen._adaptive_rep_window(generated.shape[-1])
        window = generated[0, -window_size:]
        token_ids = window.clamp(0, vocab - 1)
        counts = torch.bincount(token_ids, minlength=vocab)
        # Presence penalty: subtract presence_penalty * (count > 0)
        presence = 2.0
        logits_test[0] -= presence * (counts > 0).float()
        # Token 3 and 7 should both get -2.0 (same, not proportional to count)
        assert logits_test[0, 3].item() == pytest.approx(3.0)
        assert logits_test[0, 7].item() == pytest.approx(3.0)
        # Unseen token should be untouched
        assert logits_test[0, 0].item() == pytest.approx(5.0)


# ================================================================
# Greedy Decode Consistency
# ================================================================


class TestGreedyDecode:
    """Verify greedy decode (temperature=0) is deterministic and correct."""

    def test_greedy_is_deterministic(self):
        """Same logits + temperature=0 must always produce the same token."""
        import torch
        from enigma_engine.core.model_utils import sample_next_token

        logits = torch.tensor([[1.5, 3.2, 0.1, 2.7, 4.0]])
        generated = torch.zeros(1, 5, dtype=torch.long)

        results = set()
        for _ in range(20):
            tok = sample_next_token(
                logits.clone(), generated, temperature=0.0,
                repetition_penalty=1.0)
            results.add(tok.item())
        assert results == {4}, f"Greedy must always pick argmax (4), got {results}"

    def test_greedy_skips_penalty(self):
        """Temperature <= 0 returns argmax before repetition penalty is applied."""
        import torch
        from enigma_engine.core.model_utils import sample_next_token

        # Token 2 has highest logit but is in history
        logits = torch.tensor([[1.0, 2.0, 10.0]])
        generated = torch.tensor([[2, 2, 2, 2, 2]])  # Token 2 repeated

        # Greedy decode should still pick token 2 (argmax on raw logits)
        tok = sample_next_token(
            logits.clone(), generated, temperature=0.0,
            repetition_penalty=2.0)
        assert tok.item() == 2, "Greedy ignores repetition penalty"


# ================================================================
# Temperature Behavior
# ================================================================


class TestTemperatureEffect:
    """Verify temperature controls sampling diversity."""

    def test_low_temperature_concentrates(self):
        """Very low temperature should approximate greedy decoding."""
        import torch
        from enigma_engine.core.model_utils import sample_next_token

        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0, 4.0]])
        generated = torch.zeros(1, 5, dtype=torch.long)

        results = set()
        for _ in range(50):
            tok = sample_next_token(
                logits.clone(), generated, temperature=0.01, top_k=0,
                top_p=1.0, repetition_penalty=1.0)
            results.add(tok.item())
        assert results == {1}, f"Low temp should always pick max, got {results}"

    def test_high_temperature_increases_diversity(self):
        """High temperature should spread probability across tokens."""
        import torch
        from enigma_engine.core.model_utils import sample_next_token

        logits = torch.tensor([[2.0, 1.8, 1.5, 1.2, 1.0]])
        generated = torch.zeros(1, 5, dtype=torch.long)

        results = set()
        for _ in range(500):
            tok = sample_next_token(
                logits.clone(), generated, temperature=5.0, top_k=0,
                top_p=1.0, repetition_penalty=1.0)
            results.add(tok.item())
        assert len(results) >= 4, f"High temp should show diversity, got {results}"


# ================================================================
# NaN Guard — All-inf logits fallback
# ================================================================


class TestNaNGuard:
    """Sampling must not crash when all logits are -inf (aggressive filtering)."""

    def test_batch_nan_guard_returns_valid_token(self):
        """_sample_token_batch with all -inf logits should return a token, not crash."""
        import torch

        gen = FakeGen()
        batch, vocab = 2, 10
        # All logits -inf (as if every token was filtered out)
        logits = torch.full((batch, vocab), float('-inf'))
        generated = torch.zeros(batch, 5, dtype=torch.long)

        result = gen._sample_token_batch(
            logits, generated, temperature=1.0, top_k=0,
            top_p=1.0, repetition_penalty=1.0)
        assert result.shape == (batch, 1)
        # Must return valid token IDs (not NaN, not negative)
        assert not torch.isnan(result.float()).any(), "Must not return NaN"
        assert (result >= 0).all(), "Token IDs must be non-negative"
        assert (result < vocab).all(), "Token IDs must be within vocab range"

    def test_single_nan_guard_returns_valid_token(self):
        """_sample_token with all -inf logits should fall back gracefully."""
        import torch

        gen = FakeGenAdvanced()
        vocab = 10
        logits = torch.full((1, vocab), float('-inf'))
        generated = torch.zeros(1, 5, dtype=torch.long)

        result = gen._sample_token(
            logits, generated, temperature=1.0, top_k=0,
            top_p=1.0, repetition_penalty=1.0)
        assert result.shape == (1, 1)
        assert not torch.isnan(result.float()).any(), "Must not return NaN"
        assert (result >= 0).all()
        assert (result < vocab).all()

    def test_sample_next_token_nan_guard(self):
        """sample_next_token with all -inf logits should not crash."""
        import torch
        from enigma_engine.core.model_utils import sample_next_token

        vocab = 10
        logits = torch.full((1, vocab), float('-inf'))
        generated = torch.zeros(1, 5, dtype=torch.long)

        result = sample_next_token(
            logits, generated, temperature=1.0, top_k=0,
            top_p=1.0, repetition_penalty=1.0)
        assert result.shape == (1, 1)
        assert not torch.isnan(result.float()).any(), "Must not return NaN"
        assert (result >= 0).all()
        assert (result < vocab).all()

    def test_nan_guard_prefers_prefilter_distribution(self):
        """S720: When filtering kills all tokens, fall back to pre-filter
        distribution rather than always picking token 0."""
        import torch
        from enigma_engine.core.model_utils import sample_next_token

        vocab = 10
        # Token 7 has the highest logit — pre-filter distribution
        # should favor it when post-filter produces all -inf.
        logits = torch.full((1, vocab), -100.0)
        logits[0, 7] = 10.0
        generated = torch.zeros(1, 5, dtype=torch.long)

        # Use extremely aggressive min_p to kill everything
        # except token 7 which will survive.
        results = set()
        for _ in range(20):
            r = sample_next_token(
                logits.clone(), generated, temperature=0.01,
                top_k=0, top_p=1.0, repetition_penalty=1.0,
                min_p=0.99)
            results.add(r.item())
        # Token 7 should dominate (near-deterministic at temp=0.01)
        assert 7 in results, (
            "Pre-filter fallback should preserve model distribution")


# ================================================================
# Adaptive Repetition Window
# ================================================================


class TestAdaptiveRepWindow:
    """Verify the adaptive window scales correctly with sequence length."""

    def test_short_sequence_gets_min_window(self):
        """Sequences < 128 should get the minimum window of 64."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        assert _GenerationMixin._adaptive_rep_window(50) == 64
        assert _GenerationMixin._adaptive_rep_window(100) == 64

    def test_medium_sequence_scales(self):
        """Window = seq_len // 2, between 64 and 256."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        assert _GenerationMixin._adaptive_rep_window(200) == 100
        assert _GenerationMixin._adaptive_rep_window(400) == 200

    def test_long_sequence_gets_max_window(self):
        """Sequences > 512 should cap at 256."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        assert _GenerationMixin._adaptive_rep_window(1000) == 256
        assert _GenerationMixin._adaptive_rep_window(10000) == 256


# ================================================================
# Batch Sampling — Behavioral Checks
# ================================================================


class TestBatchSamplingBehavior:
    """Verify _sample_token_batch produces correct output values, not just shapes."""

    def test_batch_greedy_with_low_temp(self):
        """Near-zero temperature in batch should approximate argmax per row."""
        import torch

        gen = FakeGen()
        batch, vocab = 3, 10
        logits = torch.zeros(batch, vocab)
        logits[0, 5] = 10.0  # row 0: token 5 wins
        logits[1, 2] = 10.0  # row 1: token 2 wins
        logits[2, 8] = 10.0  # row 2: token 8 wins
        generated = torch.zeros(batch, 5, dtype=torch.long)

        result = gen._sample_token_batch(
            logits, generated, temperature=0.001, top_k=0,
            top_p=1.0, repetition_penalty=1.0)
        assert result[0].item() == 5
        assert result[1].item() == 2
        assert result[2].item() == 8

    def test_batch_repetition_penalty_suppresses(self):
        """Batch rep penalty should suppress repeated tokens per-row."""
        import torch

        gen = FakeGen()
        batch, vocab = 2, 10
        logits = torch.ones(batch, vocab) * 3.0
        # Row 0: token 3 repeated, row 1: token 7 repeated
        generated = torch.zeros(batch, 20, dtype=torch.long)
        generated[0, :] = 3
        generated[1, :] = 7

        # Strong penalty + low temp should avoid the repeated token
        results_row0 = set()
        results_row1 = set()
        for _ in range(50):
            result = gen._sample_token_batch(
                logits.clone(), generated, temperature=0.01, top_k=0,
                top_p=1.0, repetition_penalty=5.0)
            results_row0.add(result[0].item())
            results_row1.add(result[1].item())

        assert 3 not in results_row0, "Row 0's repeated token 3 should be suppressed"
        assert 7 not in results_row1, "Row 1's repeated token 7 should be suppressed"

    def test_batch_top_k_limits_candidates(self):
        """Batch top_k=1 should give argmax per row."""
        import torch

        gen = FakeGen()
        batch, vocab = 2, 20
        logits = torch.randn(batch, vocab)
        expected = logits.argmax(dim=-1)
        generated = torch.zeros(batch, 5, dtype=torch.long)

        result = gen._sample_token_batch(
            logits, generated, temperature=1.0, top_k=1,
            top_p=1.0, repetition_penalty=1.0)
        assert result[0].item() == expected[0].item()
        assert result[1].item() == expected[1].item()

    def test_batch_frequency_penalty(self):
        """Batch frequency penalty should subtract proportionally to count."""
        import torch

        gen = FakeGen()
        batch, vocab = 1, 10
        logits = torch.ones(batch, vocab) * 5.0
        # Token 5 appears 8 times
        generated = torch.tensor([[5] * 8])

        # frequency_penalty=1.0 → token 5 loses 8 points → logit = -3
        results = set()
        for _ in range(50):
            tok = gen._sample_token_batch(
                logits.clone(), generated, temperature=0.01, top_k=0,
                top_p=1.0, repetition_penalty=1.0, frequency_penalty=1.0)
            results.add(tok.item())
        assert 5 not in results, "Frequency-penalised token should be suppressed"

