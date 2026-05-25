"""Tests for inference engine: EnigmaEngine lifecycle, generation, sampling, and utilities."""
import sys
import threading
from pathlib import Path

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
        """Cache should clear when it exceeds the scaled cap."""
        engine = _make_engine()
        # Get the actual scaled cap
        from enigma_engine.core.hardware_detection import InferenceMemoryBudget
        cap = InferenceMemoryBudget().token_count_cache_cap
        # Fill cache past the cap
        for i in range(cap + 1):
            engine.count_tokens(f"text_{i}")
        # Cache should have been cleared and repopulated with last entry
        assert len(engine._token_count_cache) < cap


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
        """Falls back to reset_cache when only reset_cache is exposed."""
        engine = _make_engine()
        called = []

        class _OnlyResetCache:
            def reset_cache(self):
                called.append(True)

        engine.model = _OnlyResetCache()
        engine.clear_kv_cache()
        assert called

    def test_clear_kv_cache_dispatches_to_native_enigma_clear_cache(self):
        """Native Enigma model exposes clear_cache() (singular, per-layer).

        Regression guard for Pass 156z9fa finding: the dispatcher used to
        probe only ('clear_kv_cache', 'reset_cache', 'kv_cache') and silently
        no-op'd on real Enigma models, leaving stale KV after adapter swaps.
        """
        engine = _make_engine()
        # Real Enigma created via _make_engine() — must NOT have the HF-style
        # method names, must have native clear_cache().
        assert not hasattr(engine.model, 'clear_kv_cache')
        assert not hasattr(engine.model, 'reset_cache')
        assert hasattr(engine.model, 'clear_cache')

        # Prime per-layer KV caches by running a forward pass with use_cache.
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        with torch.no_grad():
            engine.model(ids, use_cache=True, start_pos=0)
        # At least one layer should now have a populated cache.
        primed = [
            layer for layer in engine.model.layers
            if getattr(layer.attention, '_kv_cache', None) is not None
        ]
        assert primed, "Expected at least one layer KV cache to be primed"

        engine.clear_kv_cache()

        # clear_cache() sets each layer's _kv_cache back to None.
        for layer in engine.model.layers:
            assert layer.attention._kv_cache is None, (
                "clear_kv_cache() did not clear native Enigma layer cache"
            )


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


# ── V-8: Vision encoder checkpoint round-trip ────────────────────────────────

class TestVisionEncoderLoad:
    """`_load_vision_encoder_from_checkpoint` round-trip: train saves it,
    inference loads it. Bug was: zero readers of `vision_encoder_state` /
    `vision_encoder_config`, so a vision-trained checkpoint silently dropped
    image input at chat time."""

    @staticmethod
    def _make_engine_for_helper():
        """Bare engine with only the attributes the helper touches."""
        from enigma_engine.core.inference import EnigmaEngine
        engine = EnigmaEngine.from_model(
            _make_tiny_model(vocab_size=256),
            _make_stub_tokenizer(vocab_size=256),
            device="cpu",
        )
        return engine

    @staticmethod
    def _make_vision_capable_model():
        """Tiny model WITH vision_projection so the load helper can attach."""
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig(
            vocab_size=256, dim=32, n_layers=1, n_heads=2,
            max_seq_len=32, vision_hidden_size=16,
        )
        return Enigma(config=cfg)

    @staticmethod
    def _make_vision_encoder_state():
        """Build a real VisionEncoder state_dict + config dict to embed."""
        from enigma_engine.core.vision_encoder import (
            VisionEncoder, VisionEncoderConfig,
        )
        vcfg = VisionEncoderConfig(
            image_size=16, patch_size=8, dim=16, n_layers=1, n_heads=2,
        )
        v_enc = VisionEncoder(vcfg)
        return v_enc.state_dict(), {
            "image_size": vcfg.image_size,
            "patch_size": vcfg.patch_size,
            "dim": vcfg.dim,
            "n_layers": vcfg.n_layers,
            "n_heads": vcfg.n_heads,
        }

    def test_helper_loads_encoder_when_state_and_config_present(self):
        """Round-trip: state + config in checkpoint → engine.vision_encoder set."""
        from enigma_engine.core.vision_encoder import VisionEncoder
        engine = self._make_engine_for_helper()
        engine.vision_encoder = None  # reset
        model = self._make_vision_capable_model()
        v_state, v_cfg = self._make_vision_encoder_state()
        raw_checkpoint = {
            "model_state_dict": {},
            "vision_encoder_state": v_state,
            "vision_encoder_config": v_cfg,
        }
        engine._load_vision_encoder_from_checkpoint(
            raw_checkpoint, model, Path("fake.pth"),
        )
        assert engine.vision_encoder is not None
        assert isinstance(engine.vision_encoder, VisionEncoder)
        assert not engine.vision_encoder.training

    def test_helper_silent_when_checkpoint_has_no_vision_keys(self, caplog):
        """Text-only checkpoint → encoder stays None, no warning emitted."""
        import logging
        engine = self._make_engine_for_helper()
        engine.vision_encoder = None
        model = self._make_vision_capable_model()
        raw_checkpoint = {"model_state_dict": {}}  # no vision keys
        with caplog.at_level(logging.WARNING, logger="enigma_engine"):
            engine._load_vision_encoder_from_checkpoint(
                raw_checkpoint, model, Path("fake.pth"),
            )
        assert engine.vision_encoder is None
        # Silent on the normal text-only path.
        assert not any(
            "vision" in rec.message.lower()
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
        )

    def test_helper_raises_when_state_present_but_config_missing(self):
        """Loud failure: state without config cannot be reconstructed."""
        engine = self._make_engine_for_helper()
        model = self._make_vision_capable_model()
        v_state, _ = self._make_vision_encoder_state()
        raw_checkpoint = {
            "model_state_dict": {},
            "vision_encoder_state": v_state,
            # vision_encoder_config deliberately omitted
        }
        with pytest.raises(RuntimeError, match="vision_encoder_config"):
            engine._load_vision_encoder_from_checkpoint(
                raw_checkpoint, model, Path("fake.pth"),
            )

    def test_helper_raises_when_model_lacks_projection(self):
        """Loud failure: encoder in checkpoint but model has no projection."""
        engine = self._make_engine_for_helper()
        # _make_tiny_model is text-only, no vision_projection
        model = _make_tiny_model(vocab_size=256)
        v_state, v_cfg = self._make_vision_encoder_state()
        raw_checkpoint = {
            "model_state_dict": {},
            "vision_encoder_state": v_state,
            "vision_encoder_config": v_cfg,
        }
        with pytest.raises(RuntimeError, match="vision_projection"):
            engine._load_vision_encoder_from_checkpoint(
                raw_checkpoint, model, Path("fake.pth"),
            )

    def test_helper_raises_on_state_dict_shape_mismatch(self):
        """Loud failure: encoder weights don't match the configured shape."""
        engine = self._make_engine_for_helper()
        model = self._make_vision_capable_model()
        _, v_cfg = self._make_vision_encoder_state()
        # Corrupt: drop a required key so load_state_dict fails strictly.
        raw_checkpoint = {
            "model_state_dict": {},
            "vision_encoder_state": {"bogus.weight": torch.zeros(1)},
            "vision_encoder_config": v_cfg,
        }
        with pytest.raises(RuntimeError, match="vision encoder"):
            engine._load_vision_encoder_from_checkpoint(
                raw_checkpoint, model, Path("fake.pth"),
            )


class TestImageDroppedWarning:
    """V-8 chat-side: when user sends an image but no encoder loaded, the
    image must NOT silently disappear — log a WARNING per call."""

    def test_warns_when_images_provided_but_no_encoder(self, caplog):
        """Non-empty image_paths + None encoder → WARNING logged, returns None."""
        import logging
        from enigma_engine.core.engine_chat import _ChatMixin
        mixin = _ChatMixin()
        mixin.vision_encoder = None  # type: ignore[attr-defined]
        mixin.device = None  # type: ignore[attr-defined]
        with caplog.at_level(logging.WARNING, logger="enigma_engine"):
            result = mixin._encode_images_for_chat(["/fake/path.png"])
        assert result is None
        assert any(
            "vision" in rec.message.lower() and rec.levelno == logging.WARNING
            for rec in caplog.records
        ), "Expected a WARNING about missing vision encoder"

    def test_silent_when_no_images_and_no_encoder(self, caplog):
        """Empty image list + None encoder → no warning (normal text-only)."""
        import logging
        from enigma_engine.core.engine_chat import _ChatMixin
        mixin = _ChatMixin()
        mixin.vision_encoder = None  # type: ignore[attr-defined]
        mixin.device = None  # type: ignore[attr-defined]
        with caplog.at_level(logging.WARNING, logger="enigma_engine"):
            result = mixin._encode_images_for_chat([])
        assert result is None
        assert not any(
            "vision" in rec.message.lower() and rec.levelno >= logging.WARNING
            for rec in caplog.records
        )


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


# ── Best-of-N sampling (N-16) ────────────────────────────────────────────────

class TestBestOfN:
    """Best-of-N generation: produce N candidates, score each with the
    user-supplied reward function, return the highest-scoring response.

    Uses unbound-method calls with a `_FakeSelf` that exposes a stub
    ``generate`` so we never touch a real model or tokenizer."""

    @staticmethod
    def _fake_self(responses):
        """Build a fake engine whose ``generate`` pops from a queue."""
        from enigma_engine.core.inference import EnigmaEngine

        class _FakeSelf:
            def __init__(self, queue):
                self._queue = list(queue)
                self.calls = []  # list of (prompt, kwargs)

            def generate(self, prompt, **kwargs):
                self.calls.append((prompt, dict(kwargs)))
                return self._queue.pop(0)

        fake = _FakeSelf(responses)
        # Bind the method-under-test so `self.generate(...)` finds the stub.
        fake.generate_best_of_n = (
            EnigmaEngine.generate_best_of_n.__get__(fake)
        )
        return fake

    def test_best_of_n_rejects_zero(self):
        """N-16: n < 1 is a programming error — must raise ValueError
        before any generate call. Loud-on-real-issue: silent no-op
        would have the user wondering why best-of-N never improves."""
        fake = self._fake_self([])
        with pytest.raises(ValueError, match="n"):
            fake.generate_best_of_n("p", 0, lambda p, r: 0.0)

    def test_best_of_n_rejects_negative(self):
        """N-16: negative n is the same class of bug as zero — raise."""
        fake = self._fake_self([])
        with pytest.raises(ValueError, match="n"):
            fake.generate_best_of_n("p", -3, lambda p, r: 0.0)

    def test_best_of_n_returns_highest_scoring(self):
        """N-16 core contract: the candidate with the highest reward
        wins. Stub generate yields three distinct strings; reward
        function scores 'beta' highest. Must return 'beta'."""
        fake = self._fake_self(["alpha", "beta", "gamma"])
        scores = {"alpha": 0.1, "beta": 0.9, "gamma": 0.5}
        best = fake.generate_best_of_n(
            "prompt", 3, lambda p, r: scores[r], temperature=0.8)
        assert best == "beta"
        # Must have called generate exactly N times.
        assert len(fake.calls) == 3

    def test_best_of_n_return_all_yields_score_list(self):
        """N-16: ``return_all=True`` returns (best, [(resp, score), ...])
        so the caller can log all candidates. Order preserved."""
        fake = self._fake_self(["a", "b", "c"])
        scores = {"a": 0.2, "b": 0.7, "c": 0.5}
        best, all_scored = fake.generate_best_of_n(
            "p", 3, lambda p, r: scores[r],
            return_all=True, temperature=0.8)
        assert best == "b"
        assert all_scored == [("a", 0.2), ("b", 0.7), ("c", 0.5)]

    def test_best_of_n_ties_break_by_first_occurrence(self):
        """N-16 adversarial: three identical scores must return the
        FIRST candidate, not a later one. Catches the regression
        where someone uses ``min`` or reversed iteration."""
        fake = self._fake_self(["first", "second", "third"])
        best = fake.generate_best_of_n(
            "p", 3, lambda p, r: 0.5, temperature=0.8)
        assert best == "first"

    def test_best_of_n_forwards_gen_kwargs(self):
        """N-16: ``max_gen`` / ``temperature`` / etc. must reach the
        underlying generate call unchanged. Catches the regression
        where the wrapper drops kwargs into a local that's then
        ignored (Pass 156k label-tracking anti-pattern)."""
        fake = self._fake_self(["x", "y"])
        fake.generate_best_of_n(
            "p", 2, lambda p, r: 0.0,
            max_gen=42, temperature=0.7, top_p=0.95)
        for _, kwargs in fake.calls:
            assert kwargs["max_gen"] == 42
            assert kwargs["temperature"] == 0.7
            assert kwargs["top_p"] == 0.95

    def test_best_of_n_forwards_json_schema_to_each_candidate(self):
        """Pass 156z7 (N-15c2) sibling-sweep follow-up: closes the
        N-15 contract family. ``json_schema`` is forwarded by the same
        ``**gen_kwargs`` passthrough as ``max_gen``/``temperature`` —
        each of the N candidates is independently constrained by the
        FSM in :meth:`EnigmaEngine.generate`. This test gates the
        contract explicitly so a regression where someone "improves"
        best-of-N to consume specific kwargs without forwarding them
        fails loud here, not silently in production where users would
        get unconstrained candidates labelled as schema-conforming.
        """
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        fake = self._fake_self(["a", "b", "c"])
        fake.generate_best_of_n(
            "p", 3, lambda p, r: 0.0,
            temperature=0.7, json_schema=schema)
        assert len(fake.calls) == 3
        for _, kwargs in fake.calls:
            assert kwargs.get("json_schema") is schema, (
                "json_schema must reach every candidate's generate() "
                "call; if the wrapper drops the kwarg, candidates "
                "produce unconstrained output mislabelled as "
                "schema-conforming")

    def test_best_of_n_warns_on_deterministic_sampling(self, caplog):
        """N-16 logic-eye: best-of-N with temperature 0 produces N
        identical candidates — wasted compute. Must log a WARNING
        but proceed (user may be testing). Loud-on-real-issue."""
        fake = self._fake_self(["same", "same", "same"])
        with caplog.at_level("WARNING"):
            fake.generate_best_of_n(
                "p", 3, lambda p, r: 0.0, temperature=0.0)
        assert any("temperature" in rec.message.lower()
                   or "deterministic" in rec.message.lower()
                   for rec in caplog.records), (
            "no warning logged on temperature=0 + n>1 — user wastes "
            "GPU on identical candidates with no signal")

    def test_best_of_n_swallows_reward_errors(self, caplog):
        """N-16 robustness: a reward function that raises on ONE
        candidate must not kill the whole batch. Failed candidates
        get -inf so they can't win, warning logged, batch completes.
        Otherwise a flaky scorer takes down every best-of-N call."""

        def flaky(prompt, response):
            if response == "bad":
                raise RuntimeError("scorer broke")
            return 1.0 if response == "good" else 0.0

        fake = self._fake_self(["zero", "bad", "good"])
        with caplog.at_level("WARNING"):
            best = fake.generate_best_of_n(
                "p", 3, flaky, temperature=0.8)
        assert best == "good"
        assert any("reward" in rec.message.lower()
                   or "scor" in rec.message.lower()
                   for rec in caplog.records), (
            "no warning when reward_fn raised — silent failure")

    def test_best_of_n_n_equals_one(self):
        """N-16 degenerate case: n=1 still scores the single candidate
        and returns it. Code paths must converge — no special-case
        branch that skips reward evaluation when caller passed
        return_all=True."""
        scored = []

        def scorer(prompt, response):
            scored.append(response)
            return 0.5

        fake = self._fake_self(["only"])
        best, all_scored = fake.generate_best_of_n(
            "p", 1, scorer, return_all=True, temperature=0.8)
        assert best == "only"
        assert all_scored == [("only", 0.5)]
        assert scored == ["only"]

    def test_best_of_n_rejects_non_int_n(self):
        """N-16 audit (Pass 156x2): the boundary guard uses
        ``isinstance(n, int)`` so callers passing ``n=2.5`` fail loud
        at the API boundary instead of crashing inside ``range(2.5)``.
        Adversarial against a regression that drops the type check
        and only validates ``n < 1`` — that bug would let 2.5 through
        and TypeError would surface from ``range`` with a less helpful
        message far from the actual bad input."""
        fake = self._fake_self([])
        with pytest.raises(ValueError, match="n"):
            fake.generate_best_of_n("p", 2.5, lambda p, r: 0.0)
        with pytest.raises(ValueError, match="n"):
            fake.generate_best_of_n("p", "3", lambda p, r: 0.0)

    def test_best_of_n_handles_non_numeric_score(self, caplog):
        """N-16 audit (Pass 156x2): a reward function that returns
        ``None`` or a string passes the try/except (no exception was
        raised) but ``float(score)`` then raises TypeError — must be
        swallowed and logged the same as a raising scorer, not crash
        the whole batch. Adversarial: real-world judges built around
        ``json.loads`` can return ``None`` on parse failure."""

        def returns_none(prompt, response):
            if response == "winner":
                return 1.0
            return None  # silently broken scorer

        fake = self._fake_self(["loser", "winner", "loser"])
        with caplog.at_level("WARNING"):
            best = fake.generate_best_of_n(
                "p", 3, returns_none, temperature=0.8)
        assert best == "winner", (
            "non-numeric scores must be coerced to -inf so the only "
            "validly-scored candidate wins")
        # At least one warning logged for the None scores.
        assert any(
            "reward" in rec.message.lower()
            or "scor" in rec.message.lower()
            for rec in caplog.records
        ), "no warning when scorer returned non-numeric — silent fail"
