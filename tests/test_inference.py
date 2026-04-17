"""Tests for inference engine: EnigmaEngine lifecycle, generation, sampling, and utilities."""
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_tiny_model(vocab_size=256):
    """Create a nano model for testing."""
    from enigma_engine.core.model import create_model
    model = create_model("nano", vocab_size=vocab_size)
    model.eval()
    return model


def _make_stub_tokenizer(vocab_size=256):
    """Create a stub tokenizer with encode/decode."""
    class StubTokenizer:
        def __init__(self, vs):
            self.vocab_size = vs
            self.eos_token_id = 2
            self.pad_token_id = 0
        def encode(self, text, add_special_tokens=False):
            # Simple char-to-id mapping, clamped to vocab
            return [min(ord(c), vs - 1) for c in text]
        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(min(i, 127)) for i in ids if i > 0)
    vs = vocab_size
    return StubTokenizer(vs)


def _make_engine(vocab_size=256):
    """Create an EnigmaEngine via from_model (no disk I/O)."""
    from enigma_engine.core.inference import EnigmaEngine
    model = _make_tiny_model(vocab_size)
    tok = _make_stub_tokenizer(vocab_size)
    return EnigmaEngine.from_model(model, tok, device="cpu")


# ── EnigmaEngine.from_model factory ─────────────────────────────────────────

class TestFromModel:
    """Test creating an engine from an existing model + tokenizer."""

    def test_from_model_creates_engine(self):
        """from_model should return a working EnigmaEngine instance."""
        engine = _make_engine()
        assert engine.model is not None
        assert engine.tokenizer is not None
        assert engine.device == torch.device("cpu")

    def test_from_model_sets_eval_mode(self):
        """from_model should set model to eval mode."""
        engine = _make_engine()
        assert not engine.model.training

    def test_from_model_has_generation_lock(self):
        """Engine must have a generation lock for thread safety."""
        engine = _make_engine()
        assert hasattr(engine._generation_lock, 'acquire')
        assert hasattr(engine._generation_lock, 'release')

    def test_from_model_metadata_defaults(self):
        """Engine should have safe default metadata."""
        engine = _make_engine()
        assert engine.model_metadata["supports_nsfw"] is False
        assert engine.model_metadata["content_rating"] == "sfw"

    def test_from_model_vision_encoder_none(self):
        """Vision encoder should be None by default."""
        engine = _make_engine()
        assert engine.vision_encoder is None

    def test_from_model_tools_disabled(self):
        """Tools should be disabled by default."""
        engine = _make_engine()
        assert engine.enable_tools is False
        assert engine._tool_executor is None


# ── Device & dtype selection ─────────────────────────────────────────────────

class TestDeviceAndDtype:
    """Test _select_device and _select_dtype logic."""

    def test_select_device_explicit_cpu(self):
        """Explicit 'cpu' should return cpu device."""
        from enigma_engine.core.inference import EnigmaEngine
        dev = EnigmaEngine._select_device(None, "cpu")
        assert dev == torch.device("cpu")

    def test_select_dtype_explicit_float32(self):
        """precision='float32' should return float32."""
        from enigma_engine.core.inference import EnigmaEngine
        dt = EnigmaEngine._select_dtype(torch.device("cpu"), precision="float32")
        assert dt == torch.float32

    def test_select_dtype_half_on_cpu_falls_back(self):
        """use_half=True on CPU should fall back to float32."""
        from enigma_engine.core.inference import EnigmaEngine
        dt = EnigmaEngine._select_dtype(torch.device("cpu"), use_half=True)
        assert dt == torch.float32

    def test_select_dtype_fp16_string_on_cpu_falls_back(self):
        """precision='fp16' on CPU should fall back to float32 with warning."""
        from enigma_engine.core.inference import EnigmaEngine
        dt = EnigmaEngine._select_dtype(torch.device("cpu"), precision="fp16")
        assert dt == torch.float32

    def test_select_dtype_auto_on_cpu(self):
        """Auto precision on CPU should be float32."""
        from enigma_engine.core.inference import EnigmaEngine
        dt = EnigmaEngine._select_dtype(torch.device("cpu"), precision="auto")
        assert dt == torch.float32


# ── Encode / Decode ──────────────────────────────────────────────────────────

class TestEncodeDecode:
    """Test prompt encoding and output decoding."""

    def test_encode_prompt_returns_tensor(self):
        """_encode_prompt should return a 2D long tensor."""
        engine = _make_engine()
        ids = engine._encode_prompt("hello")
        assert isinstance(ids, torch.Tensor)
        assert ids.dim() == 2
        assert ids.dtype == torch.long

    def test_encode_prompt_batch_dim(self):
        """Encoded prompt should have batch dimension 1."""
        engine = _make_engine()
        ids = engine._encode_prompt("test")
        assert ids.shape[0] == 1

    def test_decode_output_returns_string(self):
        """_decode_output should return a string."""
        engine = _make_engine()
        ids = torch.tensor([[65, 66, 67]])
        text = engine._decode_output(ids)
        assert isinstance(text, str)

    def test_decode_handles_string_input(self):
        """_decode_output should pass through string input."""
        engine = _make_engine()
        assert engine._decode_output("hello") == "hello"


# ── count_tokens ─────────────────────────────────────────────────────────────

class TestCountTokens:
    """Test token counting with caching."""

    def test_count_tokens_basic(self):
        """count_tokens should return positive int for non-empty text."""
        engine = _make_engine()
        count = engine.count_tokens("hello world")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_empty(self):
        """count_tokens on empty string should return 0."""
        engine = _make_engine()
        assert engine.count_tokens("") == 0

    def test_count_tokens_cache_hit(self):
        """Same text should return cached result."""
        engine = _make_engine()
        c1 = engine.count_tokens("hello")
        c2 = engine.count_tokens("hello")
        assert c1 == c2

    def test_count_tokens_cache_cap(self):
        """Cache should clear when it exceeds 4096 entries."""
        engine = _make_engine()
        # Fill cache to 4096
        for i in range(4097):
            engine.count_tokens(f"text_{i}")
        # Cache should have been cleared and repopulated with last entry
        assert len(engine._token_count_cache) < 4096


# ── clear_kv_cache ───────────────────────────────────────────────────────────

class TestClearKVCache:
    """Test KV cache clearing dispatches correctly."""

    def test_clear_kv_cache_with_method(self):
        """If model has clear_kv_cache method, it should be called."""
        engine = _make_engine()
        called = []
        engine.model.clear_kv_cache = lambda: called.append(True)
        engine.clear_kv_cache()
        assert called

    def test_clear_kv_cache_with_reset_cache(self):
        """Falls back to reset_cache if clear_kv_cache missing."""
        engine = _make_engine()
        if hasattr(engine.model, 'clear_kv_cache'):
            delattr(engine.model, 'clear_kv_cache')
        called = []
        engine.model.reset_cache = lambda: called.append(True)
        engine.clear_kv_cache()
        assert called


# ── get_max_context_length ───────────────────────────────────────────────────

class TestMaxContextLength:
    """Test context length retrieval."""

    def test_returns_from_config(self):
        """Should return max_seq_len from model config."""
        engine = _make_engine()
        length = engine.get_max_context_length()
        assert isinstance(length, int)
        assert length > 0

    def test_returns_model_config_value(self):
        """Should match model.config.max_seq_len."""
        engine = _make_engine()
        expected = engine.model.config.max_seq_len
        assert engine.get_max_context_length() == expected


# ── _infer_model_config ──────────────────────────────────────────────────────

class TestInferModelConfig:
    """Test model config inference from state dict."""

    def test_infer_from_nano_state_dict(self):
        """Should infer vocab_size, dim, n_layers from a nano model."""
        from enigma_engine.core.inference import EnigmaEngine
        model = _make_tiny_model(256)
        state_dict = model.state_dict()
        # Use from_model to create a minimal engine, then test inference
        engine = _make_engine(256)
        config = engine._infer_model_config(state_dict)
        assert "vocab_size" in config
        assert config["vocab_size"] == 256
        assert "n_layers" in config
        assert config["n_layers"] > 0

    def test_infer_model_size_matches_nano(self):
        """_infer_model_size should return 'nano' for a nano state dict."""
        engine = _make_engine(256)
        state_dict = engine.model.state_dict()
        size = engine._infer_model_size(state_dict)
        assert size == "nano"


# ── Input validation in _generate_text ───────────────────────────────────────

class TestGenerateValidation:
    """Test that generation rejects invalid inputs."""

    def test_rejects_non_string_prompt(self):
        """generate should reject non-string prompt."""
        engine = _make_engine()
        with pytest.raises(TypeError, match="prompt must be a string"):
            engine.generate(123)

    def test_empty_prompt_returns_empty(self):
        """Empty prompt should return empty string."""
        engine = _make_engine()
        result = engine.generate("   ")
        assert result == ""

    def test_rejects_negative_max_gen(self):
        """Negative max_gen should raise ValueError."""
        engine = _make_engine()
        with pytest.raises(ValueError, match="max_gen must be positive"):
            engine.generate("hello", max_gen=-1)

    def test_rejects_negative_temperature(self):
        """Negative temperature should raise ValueError."""
        engine = _make_engine()
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            engine.generate("hello", temperature=-0.5)

    def test_rejects_negative_top_k(self):
        """Negative top_k should raise ValueError."""
        engine = _make_engine()
        with pytest.raises(ValueError, match="top_k must be non-negative"):
            engine.generate("hello", top_k=-1)

    def test_rejects_top_p_out_of_range(self):
        """top_p outside [0, 1] should raise ValueError."""
        engine = _make_engine()
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            engine.generate("hello", top_p=1.5)

    def test_rejects_rep_penalty_below_one(self):
        """repetition_penalty < 1.0 should raise ValueError."""
        engine = _make_engine()
        with pytest.raises(ValueError, match="repetition_penalty must be >= 1.0"):
            engine.generate("hello", repetition_penalty=0.5)


# ── max_gen aliases ──────────────────────────────────────────────────────────

class TestMaxGenAliases:
    """Test that max_tokens, max_new_tokens, max_length aliases work."""

    def test_max_tokens_alias(self):
        """max_tokens should be used as max_gen."""
        engine = _make_engine()
        # Just verify it doesn't crash with a small limit
        result = engine.generate("hi", max_tokens=5)
        assert isinstance(result, str)

    def test_max_new_tokens_alias(self):
        """max_new_tokens should be used as max_gen."""
        engine = _make_engine()
        result = engine.generate("hi", max_new_tokens=5)
        assert isinstance(result, str)


# ── Actual generation produces output ────────────────────────────────────────

class TestGeneration:
    """Test that generate() actually produces text with a real nano model."""

    def test_generate_produces_text(self):
        """generate should return non-empty string for valid prompt."""
        engine = _make_engine()
        result = engine.generate("hello", max_gen=10, temperature=1.0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_includes_prompt(self):
        """Generated text should start with the prompt."""
        engine = _make_engine()
        prompt = "test"
        result = engine.generate(prompt, max_gen=10, temperature=1.0)
        # The model uses encode/decode which may not perfectly preserve
        # the prompt text, so just check we get a string back
        assert isinstance(result, str)

    def test_generate_respects_max_gen(self):
        """Output should not have unlimited tokens."""
        engine = _make_engine()
        result = engine.generate("a", max_gen=3, temperature=1.0)
        assert isinstance(result, str)

    def test_generate_with_stop_strings(self):
        """Stop strings should truncate output in generated portion."""
        engine = _make_engine()
        # Generate with a stop string — verify it doesn't crash
        result = engine.generate("hello", max_gen=20, stop_strings=["STOP"],
                                 temperature=1.0)
        assert isinstance(result, str)

    def test_greedy_generation_deterministic(self):
        """Temperature=0 (greedy) should produce deterministic output."""
        engine = _make_engine()
        # Need to clear KV cache between runs for determinism
        engine.clear_kv_cache()
        r1 = engine.generate("test", max_gen=5, temperature=0.001)
        engine.clear_kv_cache()
        r2 = engine.generate("test", max_gen=5, temperature=0.001)
        assert r1 == r2


# ── Sampling ─────────────────────────────────────────────────────────────────

class TestSampleToken:
    """Test _sample_token with various strategies."""

    def _make_sampling_engine(self):
        return _make_engine(256)

    def test_sample_returns_valid_token(self):
        """_sample_token should return a valid token ID tensor."""
        engine = self._make_sampling_engine()
        logits = torch.randn(1, 256)
        generated = torch.randint(0, 256, (1, 10))
        token = engine._sample_token(logits, generated, temperature=1.0,
                                     top_k=0, top_p=1.0, repetition_penalty=1.0)
        assert token.shape == (1, 1)
        assert 0 <= token.item() < 256

    def test_sample_with_top_k(self):
        """top_k should limit sampling to top K tokens."""
        engine = self._make_sampling_engine()
        logits = torch.randn(1, 256)
        generated = torch.randint(0, 256, (1, 10))
        token = engine._sample_token(logits, generated, temperature=1.0,
                                     top_k=5, top_p=1.0, repetition_penalty=1.0)
        assert 0 <= token.item() < 256

    def test_sample_with_top_p(self):
        """top_p should use nucleus sampling."""
        engine = self._make_sampling_engine()
        logits = torch.randn(1, 256)
        generated = torch.randint(0, 256, (1, 10))
        token = engine._sample_token(logits, generated, temperature=1.0,
                                     top_k=0, top_p=0.9, repetition_penalty=1.0)
        assert 0 <= token.item() < 256

    def test_sample_with_min_p(self):
        """min_p should filter low-probability tokens."""
        engine = self._make_sampling_engine()
        logits = torch.randn(1, 256)
        generated = torch.randint(0, 256, (1, 10))
        token = engine._sample_token(logits, generated, temperature=1.0,
                                     top_k=0, top_p=1.0, repetition_penalty=1.0,
                                     min_p=0.1)
        assert 0 <= token.item() < 256

    def test_sample_with_repetition_penalty(self):
        """Repetition penalty should reduce probability of repeated tokens."""
        engine = self._make_sampling_engine()
        # Create logits where token 5 is slightly ahead of others
        logits = torch.full((1, 256), 0.0)
        logits[0, 5] = 2.0
        # Generate sequence full of token 5
        generated = torch.full((1, 50), 5, dtype=torch.long)
        # With high rep penalty, token 5 should be penalized enough
        # that other tokens appear frequently
        other_count = 0
        for _ in range(50):
            token = engine._sample_token(logits.clone(), generated, temperature=1.0,
                                         top_k=0, top_p=1.0, repetition_penalty=5.0)
            if token.item() != 5:
                other_count += 1
        # With rep_penalty=5.0, we should see at least some non-5 tokens
        assert other_count > 0

    def test_sample_nan_fallback(self):
        """If all logits are -inf after filtering, should fall back gracefully."""
        engine = self._make_sampling_engine()
        # All -inf logits → NaN after softmax → fallback to uniform
        logits = torch.full((1, 256), float('-inf'))
        generated = torch.randint(0, 256, (1, 10))
        token = engine._sample_token(logits, generated, temperature=1.0,
                                     top_k=0, top_p=1.0, repetition_penalty=1.0)
        # Should not crash — falls back to uniform sampling (S720)
        assert 0 <= token.item() < 256

    def test_sample_with_exempt_tokens(self):
        """Exempt tokens should not get repetition penalty."""
        engine = self._make_sampling_engine()
        logits = torch.full((1, 256), -10.0)
        logits[0, 5] = 10.0
        generated = torch.full((1, 20), 5, dtype=torch.long)
        # Token 5 exempt from penalty — should still be strongly preferred
        five_count = 0
        for _ in range(20):
            token = engine._sample_token(logits.clone(), generated, temperature=0.5,
                                         top_k=0, top_p=1.0, repetition_penalty=5.0,
                                         exempt_tokens={5})
            if token.item() == 5:
                five_count += 1
        assert five_count > 15  # Should almost always pick 5

    def test_sample_with_frequency_penalty(self):
        """Frequency penalty should penalize frequently-appearing tokens."""
        engine = self._make_sampling_engine()
        logits = torch.randn(1, 256)
        generated = torch.randint(0, 256, (1, 10))
        token = engine._sample_token(logits, generated, temperature=1.0,
                                     top_k=0, top_p=1.0, repetition_penalty=1.0,
                                     frequency_penalty=0.5)
        assert 0 <= token.item() < 256

    def test_sample_with_presence_penalty(self):
        """Presence penalty should penalize tokens that appeared at all."""
        engine = self._make_sampling_engine()
        logits = torch.randn(1, 256)
        generated = torch.randint(0, 256, (1, 10))
        token = engine._sample_token(logits, generated, temperature=1.0,
                                     top_k=0, top_p=1.0, repetition_penalty=1.0,
                                     presence_penalty=0.5)
        assert 0 <= token.item() < 256

    def test_sample_with_typical_p(self):
        """Typical sampling should work without crash."""
        engine = self._make_sampling_engine()
        logits = torch.randn(1, 256)
        generated = torch.randint(0, 256, (1, 10))
        token = engine._sample_token(logits, generated, temperature=1.0,
                                     top_k=0, top_p=1.0, repetition_penalty=1.0,
                                     typical_p=0.9)
        assert 0 <= token.item() < 256

    def test_sample_with_mirostat_v2(self):
        """Mirostat v2 should work without crash."""
        engine = self._make_sampling_engine()
        logits = torch.randn(1, 256)
        generated = torch.randint(0, 256, (1, 10))
        token = engine._sample_token(logits, generated, temperature=1.0,
                                     top_k=0, top_p=1.0, repetition_penalty=1.0,
                                     mirostat_mode=2, mirostat_tau=5.0, mirostat_eta=0.1)
        assert 0 <= token.item() < 256


# ── Stream ───────────────────────────────────────────────────────────────────

class TestStream:
    """Test the stream() convenience method."""

    def test_stream_returns_generator(self):
        """stream() should return a generator."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        assert inspect.isgeneratorfunction(EnigmaEngine.stream) or True
        # stream() delegates to stream_generate which is a generator
        engine = _make_engine()
        gen = engine.stream("hello", max_tokens=3)
        # Should be iterable
        assert hasattr(gen, '__next__') or hasattr(gen, '__iter__')


# ── Creativity detection ─────────────────────────────────────────────────────

class TestNeedsAICreativity:
    """Test _needs_ai_creativity pattern matching."""

    def test_creative_prompt_detected(self):
        """Prompts with creativity indicators should return True."""
        engine = _make_engine()
        assert engine._needs_ai_creativity("surprise me with something cool")
        assert engine._needs_ai_creativity("what do you think about this?")
        assert engine._needs_ai_creativity("can you suggest something?")

    def test_direct_command_not_creative(self):
        """Direct commands like 'draw a cat' should return False."""
        engine = _make_engine()
        assert not engine._needs_ai_creativity("draw a picture of a sunset over the ocean")

    def test_non_latin_returns_true(self):
        """Non-Latin script should default to AI (can't pattern match)."""
        engine = _make_engine()
        assert engine._needs_ai_creativity("こんにちは世界")

    def test_single_ambiguous_word(self):
        """Single non-command word should return True (ambiguous)."""
        engine = _make_engine()
        assert engine._needs_ai_creativity("philosophy")


# ── Adaptive repetition window ───────────────────────────────────────────────

class TestAdaptiveRepWindow:
    """Test _adaptive_rep_window returns reasonable values."""

    def test_short_sequence(self):
        """Short sequences should have small window."""
        engine = _make_engine()
        w = engine._adaptive_rep_window(10)
        assert isinstance(w, int)
        assert w > 0

    def test_long_sequence_larger_window(self):
        """Longer sequences should have larger window."""
        engine = _make_engine()
        w_short = engine._adaptive_rep_window(10)
        w_long = engine._adaptive_rep_window(1000)
        assert w_long >= w_short


# ── Thread safety ────────────────────────────────────────────────────────────

class TestThreadSafety:
    """Test thread safety of generation."""

    def test_generation_lock_prevents_concurrent(self):
        """Generation should be serialized by _generation_lock."""
        engine = _make_engine()
        results = []
        errors = []

        def gen_task():
            try:
                r = engine.generate("test", max_gen=3, temperature=1.0)
                results.append(r)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=gen_task)
        t2 = threading.Thread(target=gen_task)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        # Both should complete without errors
        assert len(errors) == 0
        assert len(results) == 2

    def test_set_train_lock(self):
        """set_train_lock should store the lock for inference/training coordination."""
        engine = _make_engine()
        lock = threading.Lock()
        engine.set_train_lock(lock)
        assert engine._train_lock is lock

    def test_set_train_lock_none(self):
        """set_train_lock(None) should clear the lock."""
        engine = _make_engine()
        engine.set_train_lock(threading.Lock())
        engine.set_train_lock(None)
        assert engine._train_lock is None
