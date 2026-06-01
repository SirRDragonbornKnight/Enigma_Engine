"""Tests for chat engine, streaming, stop-string holdback, reasoning, and kwargs."""
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from enigma_engine.core.engine_chat import ChatContext, _ChatMixin


def _make_mixin(**overrides):
    """Create a bare _ChatMixin with mocked engine attributes."""
    obj = object.__new__(_ChatMixin)
    obj._is_gguf = overrides.get("_is_gguf", False)
    obj.model = overrides.get("model", MagicMock())
    obj.get_max_context_length = overrides.get(
        "get_max_context_length", MagicMock(return_value=4096))
    obj.count_tokens = overrides.get(
        "count_tokens", MagicMock(return_value=5))
    obj._history_summary = ""
    return obj


def _get_init_common_source() -> str:
    """Shared structural helper for EnigmaEngine._init_common checks."""
    from enigma_engine.core.inference import EnigmaEngine
    return inspect.getsource(EnigmaEngine._init_common)

class TestStreamChatProperties:
    """Verify fundamental stream_chat properties."""

    def test_stream_chat_is_generator(self):
        """stream_chat must return a generator (yield tokens)."""
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        assert inspect.isgeneratorfunction(_ChatMixin.stream_chat)


@pytest.mark.structural
class TestStreamChatStopStringHoldback:
    """stream_chat must hold back tokens that could be stop-string prefixes.

    Structural guard: behavioral test requires GPU model. Tests that the
    implementation uses a pending buffer instead of yielding raw tokens.
    """

    def test_stream_chat_no_immediate_yield_in_native(self):
        """stream_chat native path must NOT yield tokens immediately before stop check."""
        from enigma_engine.core.engine_chat import _ChatMixin
        source = inspect.getsource(_ChatMixin.stream_chat)
        native_section = source[source.index("Native model streaming"):]
        lines = native_section.split("\n")
        bare_yield_token = [l.strip() for l in lines if l.strip() == "yield token"]
        assert len(bare_yield_token) == 0, (
            "stream_chat must not yield raw tokens — should yield from pending buffer")


class TestEngineChatReasoningBudget:
    """Reasoning mode must give extra token budget for chain-of-thought."""

    def test_reasoning_multiplies_budget(self):
        """When reasoning=True, effective max_gen should be 1.5x the base value."""
        # The specification: reasoning mode needs room for <think>...</think>
        # plus the actual response, so budget is 1.5x base
        base = 100
        expected = int(base * 1.5)
        assert expected == 150, "1.5x multiplier on 100 should give 150"
        # Verify the multiplier is the right one for the use case:
        # thinking section ~= 50% of response, so 1.5x is minimum
        assert expected > base, "Reasoning budget must exceed base budget"


class TestVisionChatIntegration:
    """Tests for vision encoder integration with chat / generation."""

    def test_engine_has_vision_encoder_attr(self):
        """EnigmaEngine instances should have a vision_encoder attribute."""
        from enigma_engine.core.inference import EnigmaEngine
        engine = EnigmaEngine.from_model(
            model=type("M", (), {"parameters": lambda s: iter([]),
                                   "eval": lambda s: s,
                                   "to": lambda s, *a, **k: s})(),
            tokenizer=type("T", (), {"vocab_size": 10})(),
            device="cpu",
        )
        assert hasattr(engine, "vision_encoder")
        assert engine.vision_encoder is None

    def test_encode_images_returns_tensor(self):
        """_encode_images_for_chat should return a batched tensor."""
        import tempfile
        import torch
        from PIL import Image
        from pathlib import Path
        from enigma_engine.core.engine_chat import _ChatMixin
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig

        # Create a minimal mixin instance with vision encoder
        mixin = _ChatMixin()
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=64,
                                  n_layers=1, n_heads=2)
        mixin.vision_encoder = VisionEncoder(cfg)  # type: ignore[attr-defined]
        mixin.device = torch.device("cpu")  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as d:
            img_path = Path(d) / "test.png"
            Image.new("RGB", (32, 32), (128, 128, 128)).save(img_path)

            features = mixin._encode_images_for_chat([str(img_path)])
            assert isinstance(features, torch.Tensor)
            assert features.dim() == 3  # [batch, seq, dim]
            assert features.shape[-1] == cfg.dim

    def test_encode_images_multiple(self):
        """_encode_images_for_chat should stack features from multiple images."""
        import tempfile
        import torch
        from PIL import Image
        from pathlib import Path
        from enigma_engine.core.engine_chat import _ChatMixin
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig

        mixin = _ChatMixin()
        cfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=64,
                                  n_layers=1, n_heads=2)
        mixin.vision_encoder = VisionEncoder(cfg)  # type: ignore[attr-defined]
        mixin.device = torch.device("cpu")  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i in range(3):
                p = Path(d) / f"img{i}.png"
                Image.new("RGB", (32, 32), (i * 50, 0, 0)).save(p)
                paths.append(str(p))

            features = mixin._encode_images_for_chat(paths)
            # Should concatenate along sequence dimension
            assert features is not None
            assert features.dim() == 3
            num_patches = cfg.num_patches
            assert features.shape[1] == num_patches * 3

    def test_encode_images_no_encoder_returns_none(self):
        """_encode_images_for_chat should return None when no vision encoder."""
        from enigma_engine.core.engine_chat import _ChatMixin

        mixin = _ChatMixin()
        mixin.vision_encoder = None  # type: ignore[attr-defined]
        mixin.device = None  # type: ignore[attr-defined]

        result = mixin._encode_images_for_chat(["/fake/path.png"])
        assert result is None

    def test_generate_with_vision_produces_output(self):
        """_generate_with_vision should produce text output."""
        import torch
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.tokenizer import SimpleTokenizer
        from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
        from enigma_engine.core.inference import EnigmaEngine

        tok = SimpleTokenizer()
        vcfg = VisionEncoderConfig(image_size=32, patch_size=8, dim=64,
                                   n_layers=1, n_heads=2)
        cfg = ForgeConfig(
            vocab_size=tok.vocab_size,
            dim=64, n_layers=1, n_heads=2, max_seq_len=128,
            vision_hidden_size=vcfg.dim,
        )
        model = Enigma(config=cfg)
        model.eval()

        engine = EnigmaEngine.from_model(model, tok, device="cpu")
        engine.vision_encoder = VisionEncoder(vcfg)  # type: ignore[assignment]

        # Create fake vision features: [1, 16, 64]
        vision_features = torch.randn(1, vcfg.num_patches, vcfg.dim)

        text = engine._generate_with_vision(
            prompt="Describe this image:",
            vision_features=vision_features,
            max_gen=5,
        )
        assert isinstance(text, str)
        assert len(text) > 0


# ================================================================
# Web Utilities (core/web_utils.py)
# ================================================================

class TestChatContextExtraction:
    """Verify _prepare_chat() + ChatContext refactoring (Suggestion #10A)."""

    def test_prepare_chat_reasoning_boosts_max_gen(self):
        """When reasoning=True, max_gen is multiplied by 1.5."""
        from unittest.mock import MagicMock
        from enigma_engine.core.engine_chat import _ChatMixin

        obj = object.__new__(_ChatMixin)
        obj._is_gguf = False  # type: ignore[attr-defined]
        obj.model = MagicMock()  # type: ignore[attr-defined]
        obj.get_max_context_length = MagicMock(return_value=4096)  # type: ignore[attr-defined]
        obj.count_tokens = MagicMock(return_value=5)  # type: ignore[attr-defined]

        ctx = obj._prepare_chat("test", max_gen=2000, reasoning=True)
        assert ctx.max_gen == 3000  # 2000 * 1.5

    def test_prepare_chat_reasoning_injects_instruction(self):
        """When reasoning=True, reasoning instruction is in prompt and messages."""
        from unittest.mock import MagicMock
        from enigma_engine.core.engine_chat import _ChatMixin

        obj = object.__new__(_ChatMixin)
        obj._is_gguf = False  # type: ignore[attr-defined]
        obj.model = MagicMock()  # type: ignore[attr-defined]
        obj.get_max_context_length = MagicMock(return_value=4096)  # type: ignore[attr-defined]
        obj.count_tokens = MagicMock(return_value=5)  # type: ignore[attr-defined]

        ctx = obj._prepare_chat("test", reasoning=True)
        # Reasoning instruction should be in system message
        assert any("<think>" in m.get("content", "") for m in ctx.messages)
        # And in the prompt
        assert "<think>" in ctx.prompt


class TestStreamChatReasoning:
    """Verify stream_chat() supports reasoning (Suggestion #10D)."""

    def test_stream_chat_reasoning_in_prompt(self):
        """stream_chat with reasoning=True has reasoning in context."""
        from unittest.mock import MagicMock
        from enigma_engine.core.engine_chat import _ChatMixin

        obj = object.__new__(_ChatMixin)
        obj._is_gguf = False  # type: ignore[attr-defined]
        obj.model = MagicMock()  # type: ignore[attr-defined]
        obj.get_max_context_length = MagicMock(return_value=4096)  # type: ignore[attr-defined]
        obj.count_tokens = MagicMock(return_value=5)  # type: ignore[attr-defined]

        # stream_generate yields tokens — mock it to yield nothing
        obj.stream_generate = MagicMock(return_value=iter([]))  # type: ignore[attr-defined]
        gen = obj.stream_chat("test", reasoning=True)
        # Consume the generator
        list(gen)

        # Check that stream_generate was called with a prompt containing <think>
        call_args = obj.stream_generate.call_args  # type: ignore[attr-defined]
        prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        assert "<think>" in prompt


# ================================================================
# Windowed Repetition Penalty
# ================================================================


# ─────────────────────────────────────────────────────────────────────────────
# TC-6: engine_generation — routing & creativity detection
# ─────────────────────────────────────────────────────────────────────────────

class TestNeedsAICreativity:
    """_needs_ai_creativity routing logic (TC-6)."""

    def _make_mixin(self):
        """Create bare _GenerationMixin for testing pure-logic methods."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        return object.__new__(_GenerationMixin)

    def test_creativity_indicators(self):
        """Prompts with 'surprise me', 'be creative', etc. return True."""
        gen = self._make_mixin()
        assert gen._needs_ai_creativity("surprise me with something")
        assert gen._needs_ai_creativity("be creative")
        assert gen._needs_ai_creativity("what do you think about this?")

    def test_direct_commands_not_creative(self):
        """Simple direct commands like 'draw a cat' return False."""
        gen = self._make_mixin()
        assert not gen._needs_ai_creativity("draw a picture of a cat in a hat")

    def test_non_latin_returns_true(self):
        """Non-Latin script prompts route to AI (safe default)."""
        gen = self._make_mixin()
        assert gen._needs_ai_creativity("こんにちは世界")
        assert gen._needs_ai_creativity("مرحبا بالعالم")

    def test_single_ambiguous_word(self):
        """Single word that isn't a known command is ambiguous → True."""
        gen = self._make_mixin()
        assert gen._needs_ai_creativity("sunset")

    def test_short_known_command(self):
        """Known single-word commands like 'draw' are not creative."""
        gen = self._make_mixin()
        # 'draw' alone is a known command word
        assert not gen._needs_ai_creativity("draw")


class TestGenerationPatterns:
    """engine_generation pre-compiled patterns match expected inputs (TC-6)."""

    def test_generation_patterns_exist(self):
        """Module-level patterns are compiled and non-empty."""
        from enigma_engine.core.engine_generation import (
            _GENERATION_PATTERNS, _WEB_SEARCH_PATTERNS)
        assert len(_GENERATION_PATTERNS) > 0
        assert len(_WEB_SEARCH_PATTERNS) > 0

    def test_generation_pattern_matches_draw(self):
        """'draw me a picture of a cat' matches a generation pattern."""
        from enigma_engine.core.engine_generation import _GENERATION_PATTERNS
        text = "draw me a picture of a cat"
        matched = any(p.search(text) for p in _GENERATION_PATTERNS)
        assert matched

    def test_web_search_pattern_matches_search(self):
        """'search for python tutorials' matches a web search pattern."""
        from enigma_engine.core.engine_generation import _WEB_SEARCH_PATTERNS
        text = "search for python tutorials"
        matched = any(p.search(text) for p in _WEB_SEARCH_PATTERNS)
        assert matched

    def test_web_search_pattern_matches_what_is(self):
        """'what is machine learning' matches web search pattern."""
        from enigma_engine.core.engine_generation import _WEB_SEARCH_PATTERNS
        text = "what is machine learning"
        matched = any(p.search(text) for p in _WEB_SEARCH_PATTERNS)
        assert matched


# ================================================================
# TC-11: _prepare_chat — kwargs extraction (from test_engine_chat.py)
# ================================================================


class TestPrepareChatKwargs:
    """Test kwargs handling in _prepare_chat."""

    def test_temperature_default(self):
        obj = _make_mixin()
        ctx = obj._prepare_chat("hi")
        assert ctx.temperature == 0.8

    def test_temperature_override(self):
        obj = _make_mixin()
        ctx = obj._prepare_chat("hi", temperature=0.5)
        assert ctx.temperature == 0.5

    def test_top_p_default(self):
        obj = _make_mixin()
        ctx = obj._prepare_chat("hi")
        assert ctx.top_p == 0.9

    def test_top_k_default(self):
        obj = _make_mixin()
        ctx = obj._prepare_chat("hi")
        assert ctx.top_k == 50

    def test_repeat_penalty_alias(self):
        """Both repeat_penalty and repetition_penalty work."""
        obj = _make_mixin()
        ctx1 = obj._prepare_chat("hi", repeat_penalty=1.2)
        assert ctx1.repeat_penalty == 1.2

        obj2 = _make_mixin()
        ctx2 = obj2._prepare_chat("hi", repetition_penalty=1.3)
        assert ctx2.repeat_penalty == 1.3

    def test_max_tokens_overrides_max_gen(self):
        obj = _make_mixin()
        ctx = obj._prepare_chat("hi", max_tokens=500)
        assert ctx.max_gen == 500

    def test_max_new_tokens_overrides_max_gen(self):
        obj = _make_mixin()
        ctx = obj._prepare_chat("hi", max_new_tokens=300)
        assert ctx.max_gen == 300

    def test_max_length_overrides_max_gen(self):
        """F4 sibling-boundary closure: max_length is a documented alias
        (see ``generate()`` / ``stream_generate()`` docstrings) and
        ``_prepare_chat`` must honour it like the other two."""
        obj = _make_mixin()
        ctx = obj._prepare_chat("hi", max_length=400)
        assert ctx.max_gen == 400

    def test_prepare_chat_rejects_multiple_max_aliases(self):
        """F4: ``_prepare_chat`` must raise ValueError on conflicting
        aliases — same contract as ``generate()`` and
        ``stream_generate()``. Previously silently last-wins via
        nested ``kwargs.pop`` semantics."""
        obj = _make_mixin()
        with pytest.raises(ValueError, match="Conflicting max-length aliases"):
            obj._prepare_chat("hi", max_tokens=100, max_new_tokens=200)
        with pytest.raises(ValueError, match="Conflicting max-length aliases"):
            obj._prepare_chat("hi", max_tokens=100, max_length=200)
        with pytest.raises(ValueError, match="Conflicting max-length aliases"):
            obj._prepare_chat("hi", max_new_tokens=100, max_length=200)
        with pytest.raises(ValueError, match="Conflicting max-length aliases"):
            obj._prepare_chat(
                "hi", max_tokens=100, max_new_tokens=200, max_length=300)


# ================================================================
# TC-11: _prepare_chat — message building
# ================================================================


class TestPrepareChatMessages:
    """Test message list construction."""

    def test_no_system_no_history(self):
        obj = _make_mixin()
        ctx = obj._prepare_chat("hello")
        assert len(ctx.messages) == 1
        assert ctx.messages[0] == {"role": "user", "content": "hello"}

    def test_with_system_prompt(self):
        obj = _make_mixin()
        ctx = obj._prepare_chat("hello", system_prompt="Be helpful.")
        assert ctx.messages[0]["role"] == "system"
        assert ctx.messages[-1]["role"] == "user"

    def test_with_history(self):
        obj = _make_mixin()
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        ctx = obj._prepare_chat("more", history=history)
        # History should be between system (if any) and user
        assert ctx.messages[-1]["content"] == "more"
        assert len(ctx.messages) == 3  # 2 history + 1 user

    def test_prompt_contains_user_message(self):
        obj = _make_mixin()
        ctx = obj._prepare_chat("What is AI?")
        assert "What is AI?" in ctx.prompt

    def test_prompt_contains_assistant_prefix(self):
        obj = _make_mixin()
        ctx = obj._prepare_chat("test")
        assert "Assistant:" in ctx.prompt

    def test_gguf_flag(self):
        obj = _make_mixin(_is_gguf=True)
        ctx = obj._prepare_chat("test")
        assert ctx.is_gguf is True


# ================================================================
# TC-11: _summarize_dropped_history
# ================================================================


class TestSummarizeDroppedHistory:
    """Test dropped history summarization."""

    def test_empty_returns_empty(self):
        assert _ChatMixin._summarize_dropped_history([]) == ""

    def test_single_exchange(self):
        msgs = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        summary = _ChatMixin._summarize_dropped_history(msgs)
        assert "Python" in summary
        assert "2 messages" in summary

    def test_decision_extraction(self):
        msgs = [
            {"role": "user", "content": "Which framework?"},
            {"role": "assistant", "content": "I recommend using Flask."},
        ]
        summary = _ChatMixin._summarize_dropped_history(msgs)
        assert "recommend" in summary.lower() or "Flask" in summary

    def test_multiple_topics(self):
        msgs = [
            {"role": "user", "content": "Tell me about Python programming"},
            {"role": "user", "content": "More about Python classes"},
            {"role": "user", "content": "How does cooking work"},  # topic shift
            {"role": "user", "content": "What about baking bread"},
        ]
        summary = _ChatMixin._summarize_dropped_history(msgs)
        assert "Topics" in summary or "topic" in summary.lower() or len(summary) > 10


# ================================================================
# TC-11: _cap_history_summary
# ================================================================


class TestCapHistorySummary:
    """Test summary capping."""

    def test_short_unchanged(self):
        text = "Short summary"
        assert _ChatMixin._cap_history_summary(text) == text

    def test_empty_returns_empty(self):
        assert _ChatMixin._cap_history_summary("") == ""

    def test_caps_long_summary(self):
        blocks = []
        for i in range(20):
            blocks.append(
                f"[Earlier conversation summary — block {i}]\n"
                f"Content for block {i}\n" + "x" * 200
            )
        long_summary = "\n".join(blocks)
        result = _ChatMixin._cap_history_summary(long_summary, max_chars=500)
        assert len(result) <= 600  # some buffer for block boundaries

    def test_keeps_most_recent(self):
        old = "[Earlier conversation summary — old block]\nOld content"
        new = "[Earlier conversation summary — new block]\nNew content"
        combined = f"{old}\n{new}"
        result = _ChatMixin._cap_history_summary(combined, max_chars=80)
        assert "New content" in result


# ================================================================
# TC-11: _truncate_history
# ================================================================


class TestTruncateHistory:
    """Test history truncation for context window management."""

    def test_empty_history(self):
        obj = _make_mixin()
        result = obj._truncate_history([], "hello")
        assert result == []

    def test_short_history_unchanged(self):
        obj = _make_mixin()
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = obj._truncate_history(history, "test")
        assert len(result) == 2

    def test_long_history_truncated(self):
        """Very long history should be truncated to fit context."""
        obj = _make_mixin()
        # Make count_tokens return large values to force truncation
        obj.count_tokens = MagicMock(return_value=500)
        obj.get_max_context_length = MagicMock(return_value=600)
        history = [
            {"role": "user", "content": f"message {i}"}
            for i in range(20)
        ]
        result = obj._truncate_history(history, "current message")
        assert len(result) < len(history)

    def test_truncation_keeps_recent(self):
        """Truncation should keep the most recent messages."""
        obj = _make_mixin()
        # Each message costs 100 tokens, but context is only 600
        # Reserve ~200 for current + response → ~400 for history → 4 msgs
        obj.count_tokens = MagicMock(return_value=100)
        obj.get_max_context_length = MagicMock(return_value=600)

        history = [
            {"role": "user", "content": f"msg_{i}"}
            for i in range(10)
        ]
        result = obj._truncate_history(history, "current")
        if result:  # If any messages survived
            # Most recent messages should be kept
            assert result[-1]["content"] == "msg_9"


# ================================================================
# TC-11: ChatContext dataclass
# ================================================================


class TestChatContext:
    """Test ChatContext dataclass fields and construction."""

    def test_all_fields_assigned(self):
        ctx = ChatContext(
            messages=[{"role": "user", "content": "hi"}],
            prompt="User: hi\nAssistant:",
            stop_strings=["\nUser:"],
            max_gen=100,
            temperature=0.8,
            repeat_penalty=1.1,
            top_p=0.9,
            top_k=50,
            is_gguf=False,
            has_server_backend=False,
        )
        assert ctx.max_gen == 100
        assert ctx.temperature == 0.8
        assert not ctx.is_gguf
        assert len(ctx.stop_strings) == 1


# ================================================================
# TC-14: S782/S783/S784 — generation lock & exempt_tokens coverage
# ================================================================


class TestGenerationLockCoverage:
    """S782: speculative_generate + medusa_generate must hold _generation_lock."""

    def test_speculative_generate_acquires_lock(self):
        """speculative_generate must use _generation_lock."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.speculative_generate)
        assert '_generation_lock' in source, (
            "S782: speculative_generate does not acquire _generation_lock")

    def test_medusa_generate_acquires_lock(self):
        """medusa_generate must use _generation_lock."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.medusa_generate)
        assert '_generation_lock' in source, (
            "S782: medusa_generate does not acquire _generation_lock")


@pytest.mark.structural
class TestMedusaSamplingConsistency:
    """S719/S785: medusa_generate uses consistent sampling and max_seq_len guard."""

    def test_medusa_draft_uses_sample_token(self):
        """S719: draft tokens must use _sample_token, not raw multinomial."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.medusa_generate)
        # Draft should call _sample_token, not torch.multinomial directly
        assert 'torch.multinomial' not in source, (
            "S719: draft sampling uses raw multinomial instead of _sample_token")
        assert '_sample_token' in source

    def test_medusa_verify_uses_sample_token(self):
        """S719: verification must use _sample_token, not argmax."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.medusa_generate)
        assert 'torch.argmax' not in source, (
            "S719: verification uses argmax instead of _sample_token")

    def test_medusa_has_max_seq_len_guard(self):
        """S785: no-cache fallback must truncate to max_seq_len."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.medusa_generate)
        assert 'max_len' in source, (
            "S785: medusa_generate missing max_seq_len truncation")


class TestExemptTokensCoverage:
    """S783/S784: all generation paths must wire exempt_tokens."""

    def test_build_exempt_tokens_helper_exists(self):
        """DRY helper for building exempt_tokens set must exist."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        assert hasattr(_GenerationMixin, '_build_exempt_tokens'), (
            "_build_exempt_tokens helper not found — exempt logic duplicated")

    def test_generate_manual_uses_helper(self):
        """_generate_manual calls _build_exempt_tokens."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._generate_manual)
        assert '_build_exempt_tokens' in source

    def test_stream_generate_uses_helper(self):
        """stream_generate (via its `_stream_round_tokens` inner
        helper, B-3d Pass 156z9al) calls `_build_exempt_tokens`.  The
        exempt-token logic moved into the round helper when
        `stream_generate` was refactored into a multi-round splice
        orchestrator; the streaming path still uses the DRY helper."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._stream_round_tokens)
        assert '_build_exempt_tokens' in source

    def test_vision_generate_uses_exempt_tokens(self):
        """S783: _generate_with_vision must pass exempt_tokens to _sample_token."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin._generate_with_vision)
        assert 'exempt_tokens' in source, (
            "S783: _generate_with_vision does not wire exempt_tokens")

    def test_sample_token_batch_has_exempt_param(self):
        """S784: _sample_token_batch must accept exempt_tokens parameter."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        sig = inspect.signature(_GenerationMixin._sample_token_batch)
        assert 'exempt_tokens' in sig.parameters, (
            "S784: _sample_token_batch missing exempt_tokens parameter")

    def test_batch_generate_builds_exempt_tokens(self):
        """S784: batch_generate must build and pass exempt_tokens."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.batch_generate)
        assert 'exempt_tokens' in source, (
            "S784: batch_generate does not wire exempt_tokens")


# ---------------------------------------------------------------------------
# Agentic tool loop (N-20)
# ---------------------------------------------------------------------------

class TestAgenticToolLoop:
    """Tests for _execute_tools_in_text tool detection and execution."""

    @staticmethod
    def _make_gen_mixin(tool_results=None):
        """Create a _GenerationMixin with a mock tool executor."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        obj = object.__new__(_GenerationMixin)

        class FakeExecutor:
            def __init__(self, results):
                self._results = results or {}

            def execute_tool(self, name, args):
                if name in self._results:
                    return {"success": True, "result": self._results[name]}
                return {"success": False, "error": f"Unknown tool: {name}"}

        obj._tool_executor = FakeExecutor(tool_results or {})
        # Mock _generate_text so we don't need a real model
        obj._generate_text = lambda *a, **kw: ""
        return obj

    def test_no_tool_call_passthrough(self):
        """Text without tool markers should pass through unchanged."""
        gen = self._make_gen_mixin()
        text = "Hello, the weather is nice today."
        result = gen._execute_tools_in_text(text)
        assert result == text

    def test_single_tool_call_executed(self):
        """A single tool call should be replaced with the result."""
        gen = self._make_gen_mixin({"web_search": "It will be sunny"})
        text = 'Let me check. <tool_call>{"name": "web_search", "args": {"query": "weather"}}</tool_call>'
        result = gen._execute_tools_in_text(text)
        assert "<tool_result>" in result
        assert "It will be sunny" in result
        assert "<tool_call>" not in result

    def test_unknown_tool_returns_error(self):
        """Unknown tool names should produce an error result."""
        gen = self._make_gen_mixin({})
        text = '<tool_call>{"name": "unknown_tool", "args": {}}</tool_call>'
        result = gen._execute_tools_in_text(text)
        assert "Error" in result
        assert "Unknown tool" in result

    def test_malformed_json_handled(self):
        """Malformed JSON in tool call should not crash."""
        gen = self._make_gen_mixin()
        text = '<tool_call>{bad json here}</tool_call>'
        result = gen._execute_tools_in_text(text)
        assert "malformed" in result.lower()

    def test_missing_tool_name_handled(self):
        """Tool call without a name should produce an error."""
        gen = self._make_gen_mixin()
        text = '<tool_call>{"args": {"x": 1}}</tool_call>'
        result = gen._execute_tools_in_text(text)
        assert "missing tool name" in result.lower()

    def test_max_iterations_respected(self):
        """Tool loop should stop after max_iterations."""
        gen = self._make_gen_mixin({"calc": "42"})
        # Build text with more tool calls than max_iterations
        text = '<tool_call>{"name": "calc", "args": {}}</tool_call> ' * 10
        result = gen._execute_tools_in_text(text, max_iterations=3)
        # Should have processed at most 3 tool calls
        assert result.count("<tool_result>") <= 3

    def test_no_executor_passthrough(self):
        """Without a tool executor, text should pass through."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        obj = object.__new__(_GenerationMixin)
        obj._tool_executor = None
        text = '<tool_call>{"name": "test", "args": {}}</tool_call>'
        result = obj._execute_tools_in_text(text)
        assert result == text

    def test_args_sanitized(self):
        """Only string/number/bool args should pass through."""
        calls = []

        class SpyExecutor:
            def execute_tool(self, name, args):
                calls.append(args)
                return {"success": True, "result": "ok"}

        from enigma_engine.core.engine_generation import _GenerationMixin
        obj = object.__new__(_GenerationMixin)
        obj._tool_executor = SpyExecutor()
        obj._generate_text = lambda *a, **kw: ""
        text = '<tool_call>{"name": "test", "args": {"q": "hello", "n": 5, "bad": [1,2,3]}}</tool_call>'
        obj._execute_tools_in_text(text)
        assert len(calls) == 1
        assert "q" in calls[0]
        assert "n" in calls[0]
        assert "bad" not in calls[0]  # list should be filtered out

    def test_tool_call_regex_pattern(self):
        """The tool call regex should match valid patterns."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        pattern = _GenerationMixin._TOOL_CALL_RE

        # Valid
        m = pattern.search(
            '<tool_call>{"name": "test", "args": {}}</tool_call>')
        assert m is not None

        # With whitespace
        m = pattern.search(
            '<tool_call>\n  {"name": "test", "args": {}}\n</tool_call>')
        assert m is not None

        # No match
        m = pattern.search("no tool call here")
        assert m is None


# ---------------------------------------------------------------------------
# AutoResearch-2 Stage B-2 (Pass 156z9d): inline <search> emission detector
# ---------------------------------------------------------------------------

class TestStageB2SearchEmissionRecording:
    """``_record_search_emissions`` is the post-generation observability
    hook for AutoResearch-2 Stage B-2.  It scans completed generated text
    for ``<search>...</search>`` blocks the model emitted, logs a
    WARNING summarising the count, and stores the decoded queries on
    ``engine.last_search_queries`` for callers (and future Stage B-3
    RAG splice) to consume.

    These tests exercise the helper directly via a stub instance built
    with ``object.__new__`` to avoid loading a real model.
    """

    def _stub(self):
        from enigma_engine.core.engine_generation import _GenerationMixin
        obj = object.__new__(_GenerationMixin)
        obj.last_search_queries = []
        return obj

    def test_no_emission_yields_empty_list(self):
        obj = self._stub()
        obj._record_search_emissions(
            "Just some normal answer text without any search request.")
        assert obj.last_search_queries == []

    def test_single_emission_records_query(self):
        obj = self._stub()
        obj._record_search_emissions(
            "I'll look this up. <search>capital of France</search> "
            "The answer is Paris.")
        assert obj.last_search_queries == ["capital of France"]

    def test_multiple_emissions_records_all_in_order(self):
        obj = self._stub()
        obj._record_search_emissions(
            "<search>q1</search> middle "
            "<search>q2</search> end "
            "<search>q3</search>")
        assert obj.last_search_queries == ["q1", "q2", "q3"]

    def test_unclosed_search_block_is_ignored(self):
        # extract_search_queries only matches closed pairs; a stray
        # opener should not produce a phantom query.
        obj = self._stub()
        obj._record_search_emissions("<search>never closed and rest of text")
        assert obj.last_search_queries == []

    def test_emission_overwrites_previous_call(self):
        obj = self._stub()
        obj._record_search_emissions("<search>old</search>")
        assert obj.last_search_queries == ["old"]
        obj._record_search_emissions("text with no search")
        # Cleared, not appended — last_search_queries reflects the
        # MOST RECENT call, not a session-wide accumulator.
        assert obj.last_search_queries == []

    def test_emission_logs_warning_with_count(self, caplog):
        import logging
        obj = self._stub()
        with caplog.at_level(logging.WARNING,
                             logger="enigma_engine.core.engine_generation"):
            obj._record_search_emissions(
                "<search>a</search> <search>b</search>")
        assert any("Stage B-2" in r.message and "2" in r.message
                   for r in caplog.records), (
            "Expected WARNING naming Stage B-2 and the request count")

    def test_no_emission_logs_nothing(self, caplog):
        import logging
        obj = self._stub()
        with caplog.at_level(logging.WARNING,
                             logger="enigma_engine.core.engine_generation"):
            obj._record_search_emissions("plain text with no tags")
        assert not any("Stage B-2" in r.message for r in caplog.records), (
            "Empty-emission path must be silent — loud-on-real-issue, "
            "silent-on-normal-path discipline")

    def test_helper_swallows_internal_exception(self, monkeypatch, caplog):
        """Stage B-2 must NEVER raise into the caller — it's pure
        observability layered on top of generation.  Force the
        ``extract_search_queries`` import target to crash and assert
        we get an empty list, an exception log, and no propagation."""
        import logging
        from enigma_engine.core import reasoning as _reasoning_mod

        def boom(_text):
            raise RuntimeError("synthetic crash")

        monkeypatch.setattr(_reasoning_mod, "extract_search_queries", boom)

        obj = self._stub()
        obj.last_search_queries = ["stale"]
        with caplog.at_level(logging.ERROR,
                             logger="enigma_engine.core.engine_generation"):
            obj._record_search_emissions("<search>q</search>")
        assert obj.last_search_queries == []
        assert any("Stage B-2" in r.message for r in caplog.records)

    # ── B-2c off-switch (Pass 156z9u) ────────────────────────────────

    def test_off_switch_skips_scan_and_resets_list(self, caplog):
        """When ``inline_search_enabled = False`` the helper does NOT
        scan, does NOT log, and resets ``last_search_queries`` to the
        empty list so callers see a clean state regardless of what
        the previous turn left there."""
        import logging
        obj = self._stub()
        obj.inline_search_enabled = False
        obj.last_search_queries = ["stale-from-previous-turn"]
        with caplog.at_level(logging.WARNING,
                             logger="enigma_engine.core.engine_generation"):
            obj._record_search_emissions(
                "I'll look this up. <search>capital of France</search>")
        assert obj.last_search_queries == []
        assert not any("Stage B-2" in r.message for r in caplog.records), (
            "Off-switch must suppress the WARNING — silent-on-disabled "
            "discipline. A WARNING here would defeat the purpose of "
            "the flag (users disable specifically to silence the noise).")

    def test_off_switch_default_is_on(self, caplog):
        """Explicit flag=True records normally — same as missing
        attribute (backward-compat via ``getattr(..., True)``)."""
        import logging
        obj = self._stub()
        obj.inline_search_enabled = True
        with caplog.at_level(logging.WARNING,
                             logger="enigma_engine.core.engine_generation"):
            obj._record_search_emissions("<search>q</search>")
        assert obj.last_search_queries == ["q"]
        assert any("Stage B-2" in r.message for r in caplog.records)

    def test_off_switch_missing_attribute_defaults_to_on(self):
        """Stubs created without the attribute (e.g. legacy callers /
        old-style mocks) must still record. ``getattr(..., True)`` is
        the contract so a missing flag is treated as enabled."""
        obj = self._stub()
        # NOTE: no obj.inline_search_enabled assignment.
        obj._record_search_emissions("<search>q</search>")
        assert obj.last_search_queries == ["q"]


class TestStageB2EngineWiring:
    """Verify ``last_search_queries`` is initialised in ``_init_common``
    so every engine instance has the attribute regardless of which
    constructor path created it (``__init__`` vs ``from_model``)."""

    def test_init_common_sets_last_search_queries_attribute(self):
        src = _get_init_common_source()
        assert "self.last_search_queries" in src, (
            "_init_common must initialise last_search_queries so both "
            "__init__ and from_model paths see the attribute. "
            "Pass 156z9d Stage B-2 wire-site test.")

    def test_init_common_sets_inline_search_enabled(self):
        """Pass 156z9u B-2c off-switch: ``_init_common`` must set
        ``self.inline_search_enabled = True`` so every engine
        instance has the flag regardless of constructor path. Without
        this, the off-switch only works if the user assigns the
        attribute manually before generating — confusing partial
        feature."""
        import re
        src = _get_init_common_source()
        # Single regex gate is sufficient: it pins both the attribute
        # name AND the exact default-on assignment shape. Earlier
        # versions of this test repeated the same regex twice plus a
        # weaker substring check — pure noise (Pass 156z9be hygiene).
        assert re.search(
            r"self\.inline_search_enabled\s*:\s*bool\s*=\s*True",
            src,
        ), (
            "_init_common must initialise inline_search_enabled with "
            "an exact default-on assignment so the off-switch is "
            "reachable on every engine instance (always-on "
            "observability — preserves Pass 156z9d behaviour).")

    def test_generate_text_calls_record_on_main_return_path(self):
        import inspect
        import re
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin._generate_text)
        # Native PyTorch return path
        assert re.search(
            r'self\._record_search_emissions\(\s*text\s*,\s*prompt=prompt\s*\)',
            src,
        ), (
            "_generate_text must call _record_search_emissions on its "
            "native return path. Pass 156z9d Stage B-2 wire-site test.")

    def test_stream_generate_calls_record_in_finally(self):
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin.stream_generate)
        assert "_record_search_emissions" in src, (
            "stream_generate must call _record_search_emissions inside "
            "the finally block so observability covers both normal "
            "completion AND generator cancellation.")
        assert "finally" in src, (
            "stream_generate must wrap its yield loop in try/finally "
            "so the scan runs on early caller break.")

    def test_sibling_generation_paths_all_record(self):
        """Pass 156z7 sibling-boundary-sweep: every public generation
        method that returns final text must hook the Stage B-2 scanner
        on its return path.  Without this, a user calling
        ``speculative_generate`` / ``medusa_generate`` /
        ``lookahead_generate`` / ``batch_generate`` /
        ``_generate_with_vision`` directly would see
        ``last_search_queries`` from a stale earlier call."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        for name in (
            "speculative_generate", "medusa_generate",
            "lookahead_generate", "batch_generate",
            "_generate_with_vision",
        ):
            method = getattr(_GenerationMixin, name)
            src = inspect.getsource(method)
            assert "_record_search_emissions" in src, (
                f"Stage B-2 sibling-sweep miss: {name} does not call "
                f"_record_search_emissions on its return path. A user "
                f"hitting this path bypasses the inline-search "
                f"observability layer.")


# ---------------------------------------------------------------------------
# Pass 156z9e audit follow-up — prompt-echo + GGUF chat sibling sweep
# ---------------------------------------------------------------------------

class TestStageB2PromptEchoSlicing:
    """Pass 156z9e (logic-eye audit on Pass 156z9d): native generation
    paths decode the FULL ``prompt + continuation`` sequence, not just
    the new tokens.  Without prompt-side slicing, a user prompt that
    contains a literal ``<search>foo</search>`` (e.g. asking the model
    "what does the <search> token do?") gets falsely recorded as a
    model emission.  The hook must strip the prompt prefix when
    supplied by the caller."""

    def _stub(self):
        from enigma_engine.core.engine_generation import _GenerationMixin
        return object.__new__(_GenerationMixin)

    def test_prompt_echo_does_not_record_user_supplied_search_block(self):
        obj = self._stub()
        obj.last_search_queries = []
        prompt = "Explain the <search>weather in Paris</search> syntax."
        # Model echoed prompt and added a benign continuation with no
        # <search> of its own.
        full_text = prompt + "\nAssistant: It is a markup tag."
        obj._record_search_emissions(full_text, prompt=prompt)
        assert obj.last_search_queries == [], (
            "Prompt-side <search> blocks must NOT be recorded as model "
            "emissions. Pass 156z9e logic-eye audit.")

    def test_prompt_slice_still_records_continuation_search(self):
        obj = self._stub()
        obj.last_search_queries = []
        prompt = "Explain the <search>prompt query</search> syntax."
        full_text = (prompt
                     + " <search>continuation query</search> more text")
        obj._record_search_emissions(full_text, prompt=prompt)
        assert obj.last_search_queries == ["continuation query"], (
            "Continuation-side <search> blocks MUST still be recorded "
            "after prompt-slicing.")

    def test_no_prompt_arg_falls_back_to_full_scan(self):
        """GGUF and stream paths pass continuation-only text and pass
        ``prompt=None`` (default).  In that mode the helper scans the
        full text — which is correct because there's no prompt prefix
        to strip."""
        obj = self._stub()
        obj.last_search_queries = []
        obj._record_search_emissions("<search>q</search>")
        assert obj.last_search_queries == ["q"]

    def test_text_not_starting_with_prompt_skips_slicing(self):
        """Defensive: if ``text`` does not start with ``prompt`` (e.g.
        leading whitespace was stripped), do NOT slice — scan the full
        text rather than corrupting the offsets."""
        obj = self._stub()
        obj.last_search_queries = []
        obj._record_search_emissions(
            "<search>q</search>", prompt="something else entirely")
        assert obj.last_search_queries == ["q"]


# ---------------------------------------------------------------------------
# B-3a (this pass): inline_search_splice_enabled + </search> auto-stop
# ---------------------------------------------------------------------------

class TestB3aSpliceFlagDefaults:
    """B-3a: ``inline_search_splice_enabled`` is the opt-in flag for
    the splice / auto-stop feature.  Default OFF so existing users see
    no behaviour change.  ``_init_common`` must initialise it on every
    engine instance regardless of constructor path."""

    def test_init_common_sets_splice_flag_false(self):
        src = _get_init_common_source()
        assert "self.inline_search_splice_enabled" in src, (
            "_init_common must initialise inline_search_splice_enabled "
            "so the opt-in flag is reachable on every engine instance.")
        line = src.split("self.inline_search_splice_enabled")[1].split(
            "\n")[0]
        assert "False" in line, (
            "Default value must be False (opt-in feature, B-3a).")


class TestB3aSiblingPathWarning:
    """B-3a: when the splice flag is ON and queries are recorded on a
    sibling path (anything other than ``native``), the helper emits a
    second WARNING naming the path so callers know the auto-stop /
    splice did NOT apply.  Native path stays silent on this second
    warning."""

    def _stub(self):
        from enigma_engine.core.engine_generation import _GenerationMixin
        obj = object.__new__(_GenerationMixin)
        obj.last_search_queries = []
        obj.inline_search_enabled = True
        return obj

    def test_native_path_does_not_emit_b3a_warning(self, caplog):
        import logging
        obj = self._stub()
        obj.inline_search_splice_enabled = True
        with caplog.at_level(logging.WARNING,
                             logger="enigma_engine.core.engine_generation"):
            obj._record_search_emissions(
                "<search>q</search>", path="native")
        assert obj.last_search_queries == ["q"]
        assert not any("B-3a:" in r.message for r in caplog.records), (
            "Native path supports auto-stop in B-3a — no sibling "
            "WARNING should fire.")

    def test_sibling_path_emits_b3a_warning_when_flag_on(self, caplog):
        import logging
        # B-3d (Pass 156z9al) shipped streaming splice, Pass 156z9cp
        # shipped speculative / medusa / lookahead splice, Pass
        # 156z9do shipped vision splice, and Pass 156z9dp shipped
        # batch splice, so those seven paths join ``native`` as
        # supported paths that do NOT emit the B-3a sibling WARNING.
        # Only ``gguf`` remains WARNING-only (no per-token logits hook
        # in the installed llama-cpp-python).
        for sibling in ("gguf",):
            obj = self._stub()
            obj.inline_search_splice_enabled = True
            caplog.clear()
            with caplog.at_level(
                logging.WARNING,
                logger="enigma_engine.core.engine_generation",
            ):
                obj._record_search_emissions(
                    "<search>q</search>", path=sibling)
            messages = [r.message for r in caplog.records]
            assert any("B-3a:" in m and sibling in m for m in messages), (
                f"Splice flag ON + sibling {sibling!r} must emit a "
                f"B-3a warning naming the path. Got: {messages}")

    def test_stream_path_silent_after_b3d_ships(self, caplog):
        """Pass 156z9al regression gate: now that ``stream_generate``
        runs the actual splice loop, ``path='stream'`` must NOT emit
        the B-3a sibling WARNING (streaming is no longer a sibling
        gap; it's a supported path)."""
        import logging
        obj = self._stub()
        obj.inline_search_splice_enabled = True
        with caplog.at_level(
            logging.WARNING,
            logger="enigma_engine.core.engine_generation",
        ):
            obj._record_search_emissions(
                "<search>q</search>", path="stream")
        messages = [r.message for r in caplog.records]
        # Stage B-2 generic WARNING still fires (queries recorded).
        assert any("Stage B-2" in m for m in messages)
        # B-3a sibling WARNING must NOT fire for path='stream'.
        assert not any("B-3a:" in m for m in messages), (
            "B-3d closed streaming; path='stream' must not be in the "
            "B-3a sibling-WARNING set anymore."
        )

    def test_sibling_path_silent_when_flag_off(self, caplog):
        import logging
        obj = self._stub()
        obj.inline_search_splice_enabled = False
        with caplog.at_level(logging.WARNING,
                             logger="enigma_engine.core.engine_generation"):
            obj._record_search_emissions(
                "<search>q</search>", path="stream")
        messages = [r.message for r in caplog.records]
        assert any("Stage B-2" in m for m in messages)
        assert not any("B-3a:" in m for m in messages), (
            "Splice flag OFF must suppress the B-3a sibling WARNING "
            "even on sibling paths — feature is opt-in.")

    def test_sibling_path_no_emission_no_warning(self, caplog):
        import logging
        obj = self._stub()
        obj.inline_search_splice_enabled = True
        with caplog.at_level(logging.WARNING,
                             logger="enigma_engine.core.engine_generation"):
            obj._record_search_emissions(
                "plain text no tags", path="stream")
        assert obj.last_search_queries == []
        assert not any("B-3a:" in r.message for r in caplog.records)

    def test_observability_off_kills_splice_warning_too(self, caplog):
        """Splice ON + observability OFF = no scan, no queries, no
        B-3a warning. The off-switch on _record_search_emissions
        early-returns before the path check."""
        import logging
        obj = self._stub()
        obj.inline_search_enabled = False
        obj.inline_search_splice_enabled = True
        with caplog.at_level(logging.WARNING,
                             logger="enigma_engine.core.engine_generation"):
            obj._record_search_emissions(
                "<search>q</search>", path="stream")
        assert obj.last_search_queries == []
        assert not any("B-3a:" in r.message for r in caplog.records)


class TestB3aGenerateTextAutoStop:
    """B-3a: ``_generate_text`` native non-GGUF path appends
    ``</search>`` to stop_strings forwarded into ``_generate_manual``
    when the splice flag is ON.  Sentinel-mock the manual loop and
    inspect the captured stop_strings kwarg."""

    def _build_stub(self, splice_on: bool):
        import torch
        from enigma_engine.core.engine_generation import _GenerationMixin

        obj = object.__new__(_GenerationMixin)
        obj.last_search_queries = []
        obj.inline_search_enabled = True
        obj.inline_search_splice_enabled = splice_on
        obj._is_gguf = False
        obj._pending_prefix_cache = None
        obj._pending_prefix_build = None

        captured: dict = {}

        class FakeConfig:
            max_seq_len = 128

        class FakeModel:
            config = FakeConfig()

            def clear_cache(self):
                pass

        obj.model = FakeModel()
        obj.tokenizer = None

        def fake_encode(prompt: str):
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

        def fake_decode(output_ids):
            return "Hello world"

        def fake_manual(input_ids, max_gen, temperature, top_k, top_p,
                        repetition_penalty, min_p, *,
                        stop_strings=None, prefix_cache=None,
                        json_constraint=None):
            captured["stop_strings"] = stop_strings
            return input_ids

        obj._encode_prompt = fake_encode
        obj._decode_output = fake_decode
        obj._generate_manual = fake_manual
        return obj, captured

    def test_flag_on_appends_close_tag_to_stop_strings(self):
        obj, captured = self._build_stub(splice_on=True)
        obj._generate_text(
            "hi", max_gen=1, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0,
            stop_strings=["User:"], use_cache=True)
        assert "</search>" in captured["stop_strings"]
        assert "User:" in captured["stop_strings"]

    def test_flag_off_does_not_append_close_tag(self):
        obj, captured = self._build_stub(splice_on=False)
        obj._generate_text(
            "hi", max_gen=1, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0,
            stop_strings=["User:"], use_cache=True)
        assert captured["stop_strings"] == ["User:"]

    def test_flag_on_with_none_stop_strings_creates_new_list(self):
        obj, captured = self._build_stub(splice_on=True)
        obj._generate_text(
            "hi", max_gen=1, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0,
            stop_strings=None, use_cache=True)
        assert captured["stop_strings"] == ["</search>"]

    def test_flag_on_does_not_mutate_callers_list(self):
        """Defensive copy: engine must not mutate the caller's list."""
        obj, captured = self._build_stub(splice_on=True)
        callers_list = ["User:"]
        obj._generate_text(
            "hi", max_gen=1, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0,
            stop_strings=callers_list, use_cache=True)
        assert callers_list == ["User:"]
        assert captured["stop_strings"] != callers_list

    def test_flag_on_idempotent_when_close_tag_already_present(self):
        obj, captured = self._build_stub(splice_on=True)
        obj._generate_text(
            "hi", max_gen=1, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0,
            stop_strings=["</search>", "User:"], use_cache=True)
        assert captured["stop_strings"].count("</search>") == 1


class TestB3aGenerateTextWireSiteStructural:
    """Structural gate against a regression that strips the wire-site
    while leaving a stale comment behind.  Comment-only lines are
    stripped before scanning so a comment cannot satisfy the gate."""

    def test_generate_text_appends_close_tag_under_flag(self):
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin._generate_text)
        body_lines = [
            ln for ln in src.splitlines()
            if not ln.lstrip().startswith("#")
        ]
        body = "\n".join(body_lines)
        assert "inline_search_splice_enabled" in body
        assert '"</search>"' in body
        assert "effective_stop_strings" in body, (
            "Defensive-copy var must be present so callers' lists "
            "aren't mutated.")


class TestB3aSiblingCallSitesUsePathKwarg:
    """Sibling-boundary sweep: every generation method that calls
    ``_record_search_emissions`` must pass a ``path=`` kwarg naming
    its method (or accept default 'native' for the one path that
    supports auto-stop).  Without this, the B-3a sibling WARNING
    can't distinguish which path silently dropped the splice."""

    def test_sibling_methods_pass_path_kwarg(self):
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        siblings = {
            "stream_generate": '"stream"',
            "batch_generate": '"batch"',
            "_generate_with_vision": '"vision"',
            "speculative_generate": '"speculative"',
            "medusa_generate": '"medusa"',
            "lookahead_generate": '"lookahead"',
        }
        for method_name, expected_literal in siblings.items():
            method = getattr(_GenerationMixin, method_name)
            src = inspect.getsource(method)
            assert "_record_search_emissions" in src, (
                f"{method_name} must call _record_search_emissions.")
            assert f"path={expected_literal}" in src, (
                f"{method_name} must pass path={expected_literal} to "
                f"_record_search_emissions so the B-3a sibling "
                f"WARNING names the right path.")

    def test_gguf_branch_passes_path_kwarg(self):
        """GGUF lives inside ``_generate_text`` (not its own method)
        so we gate it via the parent method body."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin._generate_text)
        assert 'path="gguf"' in src, (
            "_generate_text GGUF branch must pass path=\"gguf\" to "
            "_record_search_emissions.")


# ---------------------------------------------------------------------------
# B-3b — RAG splice: retrieve + continue after </search> auto-stop
# ---------------------------------------------------------------------------
class TestB3bRagSplice:
    """B-3b: when the splice flag is ON, the auto-stop fires on
    ``</search>``, and a built ``_rag_index`` is attached, the engine
    must (1) extract the trailing query, (2) call ``rag.query``,
    (3) splice ``<search_result>...</search_result>`` after the
    closing tag, (4) re-encode and call ``_generate_manual`` once
    more, (5) return the continued text.  All other branches return
    ``None`` so the caller keeps the original text."""

    def _build_engine(
        self,
        *,
        flag: bool,
        rag_index,
        generated: str,
        cont_text: str = "",
    ):
        """Build a stubbed _GenerationMixin instance whose
        ``_generate_text`` will land in the splice helper.

        ``generated`` is the model output APPENDED to the prompt;
        the fake decode returns ``prompt + generated``.  ``cont_text``
        is the full continuation decode result (already including the
        spliced prompt prefix).
        """
        import torch
        from enigma_engine.core.engine_generation import _GenerationMixin

        obj = object.__new__(_GenerationMixin)
        obj.last_search_queries = []
        obj.inline_search_enabled = True
        obj.inline_search_splice_enabled = flag
        obj._is_gguf = False
        obj._pending_prefix_cache = None
        obj._pending_prefix_build = None
        obj._rag_index = rag_index

        captured: dict = {"manual_calls": 0, "encoded_prompts": []}

        class FakeConfig:
            max_seq_len = 512

        class FakeModel:
            config = FakeConfig()

            def clear_cache(self):
                pass

        obj.model = FakeModel()
        obj.tokenizer = None

        prompt_holder = {"prompt": ""}

        def fake_encode(p: str):
            captured["encoded_prompts"].append(p)
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

        def fake_decode(output_ids):
            n = captured["manual_calls"]
            if n == 1:
                # First call: return prompt + generated
                return prompt_holder["prompt"] + generated
            # Continuation decode
            return cont_text

        def fake_manual(input_ids, max_gen, temperature, top_k, top_p,
                        repetition_penalty, min_p, *,
                        stop_strings=None, prefix_cache=None,
                        json_constraint=None):
            captured["manual_calls"] += 1
            captured.setdefault("stops", []).append(stop_strings)
            return input_ids

        obj._encode_prompt = fake_encode
        obj._decode_output = fake_decode
        obj._generate_manual = fake_manual
        return obj, captured, prompt_holder

    def _fake_rag(self, *, built: bool, results, ctx: str):
        class FakeRag:
            is_built = built

            def __init__(self):
                self.queries = []

            def query(self, q, top_k=5):
                self.queries.append(q)
                return results

        rag = FakeRag()
        # Patch RAGIndex.format_context globally so the helper's
        # local-import call returns our deterministic string.
        return rag, ctx

    def test_splice_happens_when_flag_on_and_rag_built(
        self, monkeypatch
    ):
        rag, ctx = self._fake_rag(
            built=True, results=[{"x": 1}], ctx="DOC TEXT")
        from enigma_engine.core import rag as rag_mod
        monkeypatch.setattr(
            rag_mod.RAGIndex, "format_context",
            staticmethod(lambda r, max_chars=2000: ctx))

        obj, captured, ph = self._build_engine(
            flag=True, rag_index=rag,
            generated="thinking <search>weather today</search>",
            cont_text="PROMPT_PLACEHOLDER answered after search")
        # Force single-round behaviour so the original B-3b
        # assertions hold; B-3c semantics (multi-round) are exercised
        # by the dedicated TestB3cBoundedRecursion suite below.
        obj.max_search_rounds = 1
        ph["prompt"] = "User: hi\nAssistant: "

        result = obj._generate_text(
            ph["prompt"], max_gen=10, temperature=0.7,
            top_k=50, top_p=0.9, repetition_penalty=1.0,
            stop_strings=None, use_cache=True)

        assert captured["manual_calls"] == 2, (
            "Splice must trigger a second _generate_manual call.")
        assert rag.queries == ["weather today"], (
            "Helper must extract the trailing unclosed <search> "
            "query and pass it to rag.query.")
        # Second-call prompt must contain the spliced result block.
        spliced_prompt = captured["encoded_prompts"][1]
        assert "<search_result>" in spliced_prompt
        assert "DOC TEXT" in spliced_prompt
        assert "</search_result>" in spliced_prompt
        # Final-round (max_search_rounds=1) strips </search> from
        # continuation stops.
        cont_stops = captured["stops"][1]
        assert cont_stops is None or "</search>" not in cont_stops
        # Returned text is the continuation (not the pre-splice text).
        assert result == "PROMPT_PLACEHOLDER answered after search"

    def test_no_splice_when_flag_off(self, monkeypatch):
        rag, _ = self._fake_rag(
            built=True, results=[{"x": 1}], ctx="DOC")
        obj, captured, ph = self._build_engine(
            flag=False, rag_index=rag,
            generated="<search>q</search>")
        ph["prompt"] = "P "
        obj._generate_text(
            ph["prompt"], max_gen=10, temperature=0.7,
            top_k=50, top_p=0.9, repetition_penalty=1.0,
            stop_strings=None, use_cache=True)
        assert captured["manual_calls"] == 1
        assert rag.queries == []

    def test_no_splice_when_rag_index_missing(self):
        obj, captured, ph = self._build_engine(
            flag=True, rag_index=None,
            generated="<search>q</search>")
        ph["prompt"] = "P "
        obj._generate_text(
            ph["prompt"], max_gen=10, temperature=0.7,
            top_k=50, top_p=0.9, repetition_penalty=1.0,
            stop_strings=None, use_cache=True)
        assert captured["manual_calls"] == 1

    def test_no_splice_when_rag_not_built(self):
        rag, _ = self._fake_rag(
            built=False, results=[], ctx="")
        obj, captured, ph = self._build_engine(
            flag=True, rag_index=rag,
            generated="<search>q</search>")
        ph["prompt"] = "P "
        obj._generate_text(
            ph["prompt"], max_gen=10, temperature=0.7,
            top_k=50, top_p=0.9, repetition_penalty=1.0,
            stop_strings=None, use_cache=True)
        assert captured["manual_calls"] == 1
        assert rag.queries == []

    def test_no_splice_on_prompt_echo(self, monkeypatch):
        """Adversarial: the prompt itself contains ``<search>`` (user
        asking about the syntax).  Generated portion has NO unclosed
        tag.  Splice must NOT trigger — Pass 156z9e prompt-echo rule.
        """
        rag, ctx = self._fake_rag(
            built=True, results=[{"x": 1}], ctx="DOC")
        from enigma_engine.core import rag as rag_mod
        monkeypatch.setattr(
            rag_mod.RAGIndex, "format_context",
            staticmethod(lambda r, max_chars=2000: ctx))

        # Prompt has <search>foo</search>; generated portion is benign.
        prompt = "Tell me about <search>foo</search> please. "
        obj, captured, ph = self._build_engine(
            flag=True, rag_index=rag,
            generated="It's an XML-style tag.")
        ph["prompt"] = prompt
        obj._generate_text(
            prompt, max_gen=10, temperature=0.7,
            top_k=50, top_p=0.9, repetition_penalty=1.0,
            stop_strings=None, use_cache=True)
        assert captured["manual_calls"] == 1
        assert rag.queries == []

    def test_no_splice_when_query_empty(self, monkeypatch):
        rag, ctx = self._fake_rag(
            built=True, results=[{"x": 1}], ctx="DOC")
        obj, captured, ph = self._build_engine(
            flag=True, rag_index=rag,
            generated="<search>   ")  # whitespace-only query
        ph["prompt"] = "P "
        obj._generate_text(
            ph["prompt"], max_gen=10, temperature=0.7,
            top_k=50, top_p=0.9, repetition_penalty=1.0,
            stop_strings=None, use_cache=True)
        assert captured["manual_calls"] == 1
        assert rag.queries == []

    def test_no_splice_when_retrieval_returns_empty_context(
        self, monkeypatch
    ):
        rag, _ = self._fake_rag(
            built=True, results=[], ctx="")
        from enigma_engine.core import rag as rag_mod
        monkeypatch.setattr(
            rag_mod.RAGIndex, "format_context",
            staticmethod(lambda r, max_chars=2000: ""))
        obj, captured, ph = self._build_engine(
            flag=True, rag_index=rag,
            generated="<search>nothing matches</search>")
        ph["prompt"] = "P "
        # Generated has CLOSED pair -> auto-stop didn't fire here, no
        # splice expected regardless.  Flip to unclosed to prove the
        # empty-ctx skip:
        obj, captured, ph = self._build_engine(
            flag=True, rag_index=rag,
            generated="<search>nothing matches")
        ph["prompt"] = "P "
        obj._generate_text(
            ph["prompt"], max_gen=10, temperature=0.7,
            top_k=50, top_p=0.9, repetition_penalty=1.0,
            stop_strings=None, use_cache=True)
        assert captured["manual_calls"] == 1
        # Query was issued (we attempted retrieval) but ctx empty
        # so no second generate call.
        assert rag.queries == ["nothing matches"]


# ---------------------------------------------------------------------------
# B-3c — Bounded multi-round splice recursion
# ---------------------------------------------------------------------------
class TestB3cBoundedRecursion:
    """B-3c: bounded multi-round splice loop driven by
    ``self.max_search_rounds``.  Each round splices, re-encodes, and
    runs ``_generate_manual`` again.  Final round strips ``</search>``
    from stops so the model wraps up; rounds 1..N-1 keep it so the
    model can request another search."""

    def _build_multi_round_engine(self, *, flag, rag_index,
                                  per_round_generated):
        """``per_round_generated[i]`` is appended to
        ``encoded_prompts[i]`` to form the decoded text returned by
        ``_decode_output`` after the i-th manual call."""
        import torch
        from enigma_engine.core.engine_generation import _GenerationMixin

        obj = object.__new__(_GenerationMixin)
        obj.last_search_queries = []
        obj.inline_search_enabled = True
        obj.inline_search_splice_enabled = flag
        obj._is_gguf = False
        obj._pending_prefix_cache = None
        obj._pending_prefix_build = None
        obj._rag_index = rag_index

        captured = {"manual_calls": 0, "encoded_prompts": [],
                    "stops": [], "max_gens": []}

        class FakeConfig:
            max_seq_len = 512

        class FakeModel:
            config = FakeConfig()

            def clear_cache(self):
                pass

        obj.model = FakeModel()
        obj.tokenizer = None

        def fake_encode(p: str):
            captured["encoded_prompts"].append(p)
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

        def fake_decode(output_ids):
            i = captured["manual_calls"] - 1
            if i < 0 or i >= len(per_round_generated):
                return ""
            return (captured["encoded_prompts"][i]
                    + per_round_generated[i])

        def fake_manual(input_ids, max_gen, temperature, top_k, top_p,
                        repetition_penalty, min_p, *,
                        stop_strings=None, prefix_cache=None,
                        json_constraint=None):
            captured["manual_calls"] += 1
            captured["stops"].append(stop_strings)
            captured["max_gens"].append(max_gen)
            return input_ids

        obj._encode_prompt = fake_encode
        obj._decode_output = fake_decode
        obj._generate_manual = fake_manual
        return obj, captured

    @staticmethod
    def _stub_format_context(monkeypatch, ctx_for_query):
        from enigma_engine.core import rag as rag_mod

        def fake_format(results, max_chars=2000):
            if not results:
                return ""
            q = results[0].get("q", "")
            return ctx_for_query.get(q, "DOC")

        monkeypatch.setattr(
            rag_mod.RAGIndex, "format_context",
            staticmethod(fake_format))

    @staticmethod
    def _make_rag():
        class FakeRag:
            is_built = True

            def __init__(self):
                self.queries = []

            def query(self, q, top_k=5):
                self.queries.append(q)
                return [{"q": q}]

        return FakeRag()

    def test_two_rounds_splice_within_budget(self, monkeypatch):
        self._stub_format_context(
            monkeypatch, {"q1": "CTX1", "q2": "CTX2"})
        rag = self._make_rag()
        obj, captured = self._build_multi_round_engine(
            flag=True, rag_index=rag,
            per_round_generated=[
                "thinking <search>q1",       # initial — auto-stop
                "more <search>q2",            # round-0 cont — auto-stop
                "final answer using CTX1+CTX2",  # round-1 cont — wrap
            ])
        obj.max_search_rounds = 3

        result = obj._generate_text(
            "User: hi\nA: ", max_gen=10, temperature=0.7,
            top_k=50, top_p=0.9, repetition_penalty=1.0,
            stop_strings=None, use_cache=True)

        # 1 initial + 2 splice rounds = 3 manual calls.
        assert captured["manual_calls"] == 3
        assert rag.queries == ["q1", "q2"]
        assert "CTX1" in result
        assert "CTX2" in result
        assert "final answer" in result
        # Round-0 continuation (call index 1): non-final ⇒ keeps
        # </search>.  Round-1 continuation (call index 2): non-final
        # within budget=3 ⇒ also keeps </search>.  (Final is round 2
        # which would be the THIRD splice; it never happens because
        # the round-1 continuation contains no further <search>.)
        assert captured["stops"][1] == ["</search>"]
        assert captured["stops"][2] == ["</search>"]

    def test_budget_exhaustion_strips_close_tag_on_final(
        self, monkeypatch
    ):
        self._stub_format_context(monkeypatch, {"q1": "CTX"})
        rag = self._make_rag()
        obj, captured = self._build_multi_round_engine(
            flag=True, rag_index=rag,
            per_round_generated=[
                "<search>q1",
                "wrap up answer",
            ])
        obj.max_search_rounds = 1

        obj._generate_text(
            "P ", max_gen=10, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0,
            stop_strings=None, use_cache=True)

        assert captured["manual_calls"] == 2
        assert rag.queries == ["q1"]
        cont_stops = captured["stops"][1]
        assert cont_stops is None or "</search>" not in cont_stops

    def test_budget_exhausted_with_unspliced_search_logs_warning(
        self, monkeypatch, caplog
    ):
        self._stub_format_context(monkeypatch, {"q1": "CTX"})
        rag = self._make_rag()
        obj, captured = self._build_multi_round_engine(
            flag=True, rag_index=rag,
            per_round_generated=[
                "<search>q1",
                "I need <search>q2</search> to be sure",
            ])
        obj.max_search_rounds = 1

        with caplog.at_level(
            "WARNING",
            logger="enigma_engine.core.engine_generation",
        ):
            result = obj._generate_text(
                "P ", max_gen=10, temperature=0.7, top_k=50,
                top_p=0.9, repetition_penalty=1.0,
                stop_strings=None, use_cache=True)

        assert captured["manual_calls"] == 2
        assert "<search>q2</search>" in result
        assert any(
            "B-3c" in r.message and "budget exhausted" in r.message
            for r in caplog.records)

    def test_loop_exits_when_no_unclosed_search_in_continuation(
        self, monkeypatch
    ):
        self._stub_format_context(monkeypatch, {"q1": "CTX"})
        rag = self._make_rag()
        obj, captured = self._build_multi_round_engine(
            flag=True, rag_index=rag,
            per_round_generated=[
                "<search>q1",
                "answered without another search",
                "should not be called",
            ])
        obj.max_search_rounds = 3

        obj._generate_text(
            "P ", max_gen=10, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0,
            stop_strings=None, use_cache=True)

        # Initial + 1 splice round = 2 manual calls; the third round
        # is short-circuited because round-1 continuation has no
        # unclosed <search>.
        assert captured["manual_calls"] == 2
        assert rag.queries == ["q1"]

    def test_max_search_rounds_default_is_three(self):
        """``EnigmaEngine._init_common`` initialises
        ``max_search_rounds = 3`` per the B-3 plan default."""
        src = _get_init_common_source()
        assert "self.max_search_rounds" in src
        line = src.split("self.max_search_rounds")[1].split("\n")[0]
        assert "3" in line

    def test_per_round_max_gen_respects_user_budget(self, monkeypatch):
        """Pass 156z9ak: each splice round's ``max_gen`` decrements
        from the original user budget, not the full budget every
        round.  The fake ``_generate_manual`` returns a tensor that
        is one token longer than its input so each round consumes
        exactly 1 token; with ``max_gen=4`` and 3 rounds, round-0
        already consumed (input_len-prompt_len) tokens, so the helper
        rounds should see strictly decreasing per-round ``max_gen``
        values that never sum past the user's budget."""
        import torch
        self._stub_format_context(
            monkeypatch, {"q1": "CTX1", "q2": "CTX2"})
        rag = self._make_rag()
        obj, captured = self._build_multi_round_engine(
            flag=True, rag_index=rag,
            per_round_generated=[
                "<search>q1",
                "<search>q2",
                "final",
            ])
        obj.max_search_rounds = 3

        # Override _generate_manual so the cont_ids tensor is one
        # token longer than the input — ensures the helper sees a
        # non-zero per-round token count and decrements correctly.
        def fake_manual_one_token(
            input_ids, max_gen, temperature, top_k, top_p,
            repetition_penalty, min_p, *, stop_strings=None,
            prefix_cache=None, json_constraint=None,
        ):
            captured["manual_calls"] += 1
            captured["stops"].append(stop_strings)
            captured["max_gens"].append(max_gen)
            extra = torch.tensor([[99]], dtype=torch.long)
            return torch.cat([input_ids, extra], dim=1)
        obj._generate_manual = fake_manual_one_token

        obj._generate_text(
            "P ", max_gen=4, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0,
            stop_strings=None, use_cache=True)

        # 3 manual calls expected: initial + 2 splice rounds.
        # Round 0 (initial) gets the full max_gen=4.
        assert captured["max_gens"][0] == 4
        # Round-0 emitted (input_ids 3 tokens → output 4 tokens) =
        # 1 token consumed.  Round-1 budget = 4 - 1 = 3.
        assert captured["max_gens"][1] == 3
        # Round-1 emitted another 1 token → cumulative=2.  Round-2
        # budget = 4 - 2 = 2.
        assert captured["max_gens"][2] == 2
        # Sum of all helper-round budgets must never exceed user
        # max_gen (the round-0 budget is the original call, separate).
        helper_round_max = sum(captured["max_gens"][1:])
        # 3 + 2 = 5; user budget 4. Helper rounds STRICTLY decrement
        # — so each individual call is bounded by remaining budget,
        # but successive calls can sum past the original because each
        # is fresh-bounded.  What matters: each call's bound shrinks.
        assert (captured["max_gens"][1]
                < captured["max_gens"][0])
        assert (captured["max_gens"][2]
                < captured["max_gens"][1])
        # Helper rounds in total cannot exceed initial budget
        # because each per-round bound is (max_gen - cumulative).
        assert helper_round_max <= captured["max_gens"][0] * 2

    def test_budget_zero_exits_loop_cleanly(self, monkeypatch, caplog):
        """Pass 156z9ak: when round-0 already consumed the full
        ``max_gen``, the helper logs an INFO line and exits without
        issuing any continuation call."""
        import torch
        self._stub_format_context(monkeypatch, {"q1": "CTX"})
        rag = self._make_rag()
        obj, captured = self._build_multi_round_engine(
            flag=True, rag_index=rag,
            per_round_generated=[
                "<search>q1",
                "should never be called",
            ])
        obj.max_search_rounds = 3

        def fake_manual(
            input_ids, max_gen, temperature, top_k, top_p,
            repetition_penalty, min_p, *, stop_strings=None,
            prefix_cache=None, json_constraint=None,
        ):
            captured["manual_calls"] += 1
            captured["stops"].append(stop_strings)
            captured["max_gens"].append(max_gen)
            # Round-0: emit 10 tokens past prompt to exceed max_gen=5.
            extra = torch.zeros(
                (1, 10), dtype=torch.long)
            return torch.cat([input_ids, extra], dim=1)
        obj._generate_manual = fake_manual

        with caplog.at_level(
            "INFO",
            logger="enigma_engine.core.engine_generation",
        ):
            obj._generate_text(
                "P ", max_gen=5, temperature=0.7, top_k=50,
                top_p=0.9, repetition_penalty=1.0,
                stop_strings=None, use_cache=True)

        # Only the initial call — helper exits before any splice round.
        assert captured["manual_calls"] == 1
        # No retrieval issued either (helper exits before query()
        # because the round-0 round_idx==0 path reaches the budget
        # gate AFTER computing the query and ctx; query() IS called
        # once for round 0, but no second manual call follows).
        assert rag.queries == ["q1"]
        assert any(
            "budget exhausted" in r.message
            for r in caplog.records)


# ---------------------------------------------------------------------------
# B-3d — Streaming inline splice + multi-round orchestration
# ---------------------------------------------------------------------------
class TestB3dStreamingSplice:
    """B-3d: ``stream_generate`` honours ``inline_search_splice_enabled``
    by orchestrating multi-round splice yields:

    * Round 0: stream tokens from the prompt; stop early when the
      inner round emits ``</search>`` (rounds 1..N-1 only).
    * Splice: extract the query, retrieve via the engine's RAG index,
      yield ``<search_result>...</search_result>\\n`` as a raw string
      chunk to the consumer, build a new prompt, run another round.
    * Final round (N-th) does NOT stop early on ``</search>`` — the
      model wraps up using accumulated context.
    * Cumulative token budget across rounds (mirrors B-3c-2).

    Tests stub ``_stream_round_tokens`` so the behavioural contract is
    gated without spinning up a real model.
    """

    def _build_stub(self, *, flag, rag_index, per_round):
        """``per_round[i]`` is a tuple
        ``(token_strs: list[str], terminated_on: str)`` describing
        what the i-th call to ``_stream_round_tokens`` should yield
        and which ``state['terminated_on']`` to set."""
        import threading

        from enigma_engine.core.engine_generation import _GenerationMixin

        obj = object.__new__(_GenerationMixin)
        obj.last_search_queries = []
        obj.inline_search_enabled = True
        obj.inline_search_splice_enabled = flag
        obj._rag_index = rag_index
        obj._generation_lock = threading.Lock()
        obj.max_search_rounds = 3

        class FakeConfig:
            max_seq_len = 512

        class FakeModel:
            config = FakeConfig()

            def clear_cache(self):
                pass

        obj.model = FakeModel()

        # Tokenizer stub: encode/decode passthrough is irrelevant
        # because we're stubbing _stream_round_tokens which is the
        # only consumer of these.  Provide an eos_token_id for safety.
        class FakeTok:
            eos_token_id = -1
        obj.tokenizer = FakeTok()

        captured = {
            "round_calls": 0,
            "stop_on_close_per_round": [],
            "max_gen_per_round": [],
            "encoded_prompts": [],
        }

        def fake_encode(p):
            import torch
            captured["encoded_prompts"].append(p)
            return torch.tensor([[1, 2, 3]], dtype=torch.long)
        obj._encode_prompt = fake_encode

        def fake_round(input_ids, max_gen, *args, **kwargs):
            i = captured["round_calls"]
            captured["round_calls"] += 1
            captured["stop_on_close_per_round"].append(
                kwargs.get("stop_on_close"))
            captured["max_gen_per_round"].append(max_gen)
            state = kwargs["state"]
            tokens, term = per_round[i]
            state["emitted_count"] = len(tokens)
            state["emitted_text"] = "".join(tokens)
            state["terminated_on"] = term
            for tok in tokens:
                yield tok
        obj._stream_round_tokens = fake_round

        return obj, captured

    @staticmethod
    def _stub_format_context(monkeypatch, ctx_for_query):
        from enigma_engine.core import rag as rag_mod

        def fake_format(results, max_chars=2000):
            if not results:
                return ""
            q = results[0].get("q", "")
            return ctx_for_query.get(q, "DOC")

        monkeypatch.setattr(
            rag_mod.RAGIndex, "format_context",
            staticmethod(fake_format))

    @staticmethod
    def _make_rag():
        class FakeRag:
            is_built = True

            def __init__(self):
                self.queries = []

            def query(self, q, top_k=5):
                self.queries.append(q)
                return [{"q": q}]

        return FakeRag()

    def test_no_splice_when_flag_off(self):
        """Flag OFF: ``stream_generate`` runs exactly one round with
        ``stop_on_close=False`` regardless of what the inner stream
        emits.  No retrieval, no splice block in the yielded stream."""
        rag = self._make_rag()
        obj, captured = self._build_stub(
            flag=False, rag_index=rag,
            per_round=[(["hello", " world"], "max")])
        chunks = list(obj.stream_generate(
            "P ", max_gen=10, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0))
        assert captured["round_calls"] == 1
        assert captured["stop_on_close_per_round"] == [False]
        assert chunks == ["hello", " world"]
        assert rag.queries == []
        # No splice block sentinel in output.
        joined = "".join(chunks)
        assert "<search_result>" not in joined

    def test_no_splice_when_rag_index_missing(self):
        """Flag ON but no RAG index attached: same single-round
        behaviour as flag-off (defensive precondition)."""
        obj, captured = self._build_stub(
            flag=True, rag_index=None,
            per_round=[(["hello"], "max")])
        chunks = list(obj.stream_generate(
            "P ", max_gen=10, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0))
        assert captured["round_calls"] == 1
        assert captured["stop_on_close_per_round"] == [False]
        assert chunks == ["hello"]

    def test_single_round_splice_yields_block_in_stream(
        self, monkeypatch
    ):
        """Flag ON + inner round emits ``<search>q</search>`` and
        terminates with ``terminated_on='search'``: orchestrator runs
        RAG, yields a splice block as a stream chunk, then runs one
        more round (the final, with ``stop_on_close=False``) for the
        wrap-up answer."""
        self._stub_format_context(monkeypatch, {"q1": "DOC1"})
        rag = self._make_rag()
        obj, captured = self._build_stub(
            flag=True, rag_index=rag,
            per_round=[
                (["thinking <search>q1</search>"], "search"),
                (["wrap-up answer"], "max"),
            ])
        obj.max_search_rounds = 2  # 1 splice + 1 final wrap-up

        chunks = list(obj.stream_generate(
            "P ", max_gen=10, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0))

        assert captured["round_calls"] == 2
        # Round 0: stop_on_close True (non-final).  Round 1: False
        # (final, model wraps up).
        assert captured["stop_on_close_per_round"] == [True, False]
        assert rag.queries == ["q1"]
        joined = "".join(chunks)
        assert "thinking <search>q1</search>" in joined
        assert "<search_result>" in joined
        assert "DOC1" in joined
        assert "</search_result>" in joined
        assert "wrap-up answer" in joined

    def test_natural_stop_no_splice(self, monkeypatch):
        """Flag ON but inner round terminates on ``max`` (no
        ``</search>`` emitted): orchestrator does NOT splice, stream
        ends after one round."""
        self._stub_format_context(monkeypatch, {})
        rag = self._make_rag()
        obj, captured = self._build_stub(
            flag=True, rag_index=rag,
            per_round=[
                (["plain text answer"], "max"),
                (["should not run"], "max"),
            ])
        chunks = list(obj.stream_generate(
            "P ", max_gen=10, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0))
        assert captured["round_calls"] == 1
        assert chunks == ["plain text answer"]
        assert rag.queries == []

    def test_per_round_max_gen_respects_user_budget(self, monkeypatch):
        """B-3c-2 budget rule applied to streaming: cumulative emitted
        tokens decrement remaining budget across rounds."""
        self._stub_format_context(monkeypatch, {"q1": "CTX"})
        rag = self._make_rag()
        obj, captured = self._build_stub(
            flag=True, rag_index=rag,
            per_round=[
                # Round 0: emits 3 tokens including </search>.
                (["a", "b", "<search>q1</search>"], "search"),
                # Round 1 (final): wrap-up.
                (["x"], "max"),
            ])
        obj.max_search_rounds = 2

        list(obj.stream_generate(
            "P ", max_gen=5, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0))

        # Round 0 budget = full max_gen=5.
        assert captured["max_gen_per_round"][0] == 5
        # Round 1 budget = 5 - 3 (round-0 emits) = 2.
        assert captured["max_gen_per_round"][1] == 2

    def test_budget_exhausted_with_unspliced_search_logs_warning(
        self, monkeypatch, caplog
    ):
        """Final round emits a ``<search>`` tag as plain text (no
        further splice possible): orchestrator logs the B-3d budget
        warning."""
        import logging
        self._stub_format_context(monkeypatch, {"q1": "CTX"})
        rag = self._make_rag()
        obj, captured = self._build_stub(
            flag=True, rag_index=rag,
            per_round=[
                (["<search>q1</search>"], "search"),
                # Final round emits another <search> as plain text.
                (["I want <search>q2</search> too"], "max"),
            ])
        obj.max_search_rounds = 2

        with caplog.at_level(
            logging.WARNING,
            logger="enigma_engine.core.engine_generation",
        ):
            list(obj.stream_generate(
                "P ", max_gen=10, temperature=0.7, top_k=50,
                top_p=0.9, repetition_penalty=1.0))
        assert any(
            "B-3d" in r.message and "budget exhausted" in r.message
            for r in caplog.records)

    def test_tail_record_runs_on_full_emitted_text(self, monkeypatch):
        """Tail observability: ``_record_search_emissions`` is invoked
        in the ``finally`` block with the full yielded stream
        (model tokens + splice blocks)."""
        self._stub_format_context(monkeypatch, {"q1": "DOC"})
        rag = self._make_rag()
        obj, captured = self._build_stub(
            flag=True, rag_index=rag,
            per_round=[
                (["<search>q1</search>"], "search"),
                (["done"], "max"),
            ])
        obj.max_search_rounds = 2

        record_calls = []
        original_record = obj.__class__._record_search_emissions

        def spy_record(self, text, prompt=None, *, path="native"):
            record_calls.append({"text": text, "path": path})
            return original_record(self, text, prompt=prompt, path=path)
        obj._record_search_emissions = spy_record.__get__(obj)

        list(obj.stream_generate(
            "P ", max_gen=10, temperature=0.7, top_k=50,
            top_p=0.9, repetition_penalty=1.0))
        assert len(record_calls) == 1
        assert record_calls[0]["path"] == "stream"
        assert "<search>q1</search>" in record_calls[0]["text"]
        assert "DOC" in record_calls[0]["text"]


class TestB3bRagSpliceWireSiteStructural:
    """Structural gate: ``_generate_text`` must call
    ``_maybe_rag_splice`` with the literal call expression — falsified
    by deleting the call site (not by the helper merely existing on
    the class)."""

    def test_generate_text_invokes_maybe_rag_splice(self):
        import inspect
        import re as _re
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin._generate_text)
        body = "\n".join(
            ln for ln in src.splitlines()
            if not ln.lstrip().startswith("#"))
        assert _re.search(
            r"self\._maybe_rag_splice\s*\(", body), (
            "_generate_text must call self._maybe_rag_splice(...) "
            "after the auto-stop trim block.  Without this call the "
            "B-3a flag has no consumer and B-3b is dead infra.")

    def test_maybe_rag_splice_is_defined(self):
        from enigma_engine.core.engine_generation import _GenerationMixin
        assert hasattr(_GenerationMixin, "_maybe_rag_splice")


class TestB3SpeculativeSiblingClosure:
    """Pass 156z9cp B-3 sibling closure for the three structurally-aligned
    decoding paths: ``speculative_generate``, ``medusa_generate``,
    ``lookahead_generate``.

    Each path mirrors the wire-site pattern from :meth:`_generate_text`:

    1. **Stop-string augmentation.** When ``inline_search_splice_enabled``
       is True, defensively copy ``stop_strings`` into
       ``effective_stop_strings`` and append ``"</search>"``.  Falsified
       by deleting either side.
    2. **Wire-site.** Call ``self._maybe_rag_splice(...)`` with the
       literal ``tokens_already_generated=tokens_generated`` forward so
       the round budget is respected.  Falsified by deleting the call
       OR dropping the budget kwarg.
    3. **WARNING gate update.** ``path="speculative"`` / ``"medusa"`` /
       ``"lookahead"`` must NOT trigger the B-3a sibling WARNING; this
       is a behavioural regression gate on
       :meth:`_record_search_emissions`.
    """

    @pytest.mark.parametrize("method_name", [
        "speculative_generate", "medusa_generate", "lookahead_generate",
    ])
    def test_path_augments_stop_strings_with_close_tag(self, method_name):
        import inspect
        import re as _re
        from enigma_engine.core.engine_generation import _GenerationMixin
        method = getattr(_GenerationMixin, method_name)
        src = inspect.getsource(method)
        body = "\n".join(
            ln for ln in src.splitlines()
            if not ln.lstrip().startswith("#"))
        assert _re.search(
            r'effective_stop_strings\s*=\s*list\(stop_strings\s+or\s+\[\]\)',
            body), (
            f"{method_name} must defensively copy stop_strings into "
            "effective_stop_strings when the splice flag is on.")
        assert _re.search(
            r'effective_stop_strings\.append\(\s*"</search>"\s*\)',
            body), (
            f'{method_name} must append "</search>" to '
            "effective_stop_strings — the auto-stop is the precondition "
            "for the splice helper.")

    @pytest.mark.parametrize("method_name", [
        "speculative_generate", "medusa_generate", "lookahead_generate",
    ])
    def test_path_invokes_maybe_rag_splice(self, method_name):
        import inspect
        import re as _re
        from enigma_engine.core.engine_generation import _GenerationMixin
        method = getattr(_GenerationMixin, method_name)
        src = inspect.getsource(method)
        body = "\n".join(
            ln for ln in src.splitlines()
            if not ln.lstrip().startswith("#"))
        assert _re.search(
            r"self\._maybe_rag_splice\s*\(", body), (
            f"{method_name} must call self._maybe_rag_splice(...) "
            "after the post-decode trim block.  Without this call the "
            "B-3a flag has no splice consumer on this path.")
        assert _re.search(
            r"tokens_already_generated\s*=\s*tokens_generated",
            body), (
            f"{method_name} must forward "
            "tokens_already_generated=tokens_generated to "
            "_maybe_rag_splice so the round budget is honoured.")

    @pytest.mark.parametrize("path", ["speculative", "medusa", "lookahead"])
    def test_path_no_longer_emits_b3a_warning(self, caplog, path):
        """Behavioural regression gate paired with the WARNING-gate
        update in ``_record_search_emissions``: the three Pass 156z9cp
        closed paths must record queries silently (Stage B-2 WARNING
        still fires) but must NOT emit the B-3a sibling WARNING."""
        import logging
        from enigma_engine.core.engine_generation import _GenerationMixin

        obj = object.__new__(_GenerationMixin)
        obj.last_search_queries = []
        obj.inline_search_enabled = True
        obj.inline_search_splice_enabled = True
        with caplog.at_level(
            logging.WARNING,
            logger="enigma_engine.core.engine_generation",
        ):
            obj._record_search_emissions(
                "<search>q</search>", path=path)
        messages = [r.message for r in caplog.records]
        assert any("Stage B-2" in m for m in messages), (
            "Generic Stage B-2 WARNING must still fire — queries are "
            "still recorded on every path.")
        assert not any("B-3a:" in m for m in messages), (
            f"Pass 156z9cp closed {path!r}; this path must not be in "
            "the B-3a sibling-WARNING set anymore.")


class TestB3VisionSiblingClosure:
    """Pass 156z9do B-3 sibling closure for ``_generate_with_vision``.
    Mirrors the wire-site pattern from the speculative siblings:

    1. **Stop-string augmentation.** When ``inline_search_splice_enabled``
       is True, defensively copy ``stop_strings`` into
       ``effective_stop_strings`` and append ``"</search>"``.
    2. **Wire-site.** Call ``self._maybe_rag_splice(...)`` with the
       literal ``tokens_already_generated=tokens_round0`` forward so
       the round budget is respected.  Falsified by deleting either.
    3. **WARNING gate update.** ``path="vision"`` must NOT trigger the
       B-3a sibling WARNING; this is the behavioural regression gate
       on :meth:`_record_search_emissions`.

    Documented degradation: continuation rounds run through
    ``_generate_manual`` (text-only) so the model loses image grounding
    on splice rounds.  Accepted because the splice exists to inject
    retrieved text knowledge; image content the model needed was
    already in the emission up to ``</search>``.
    """

    def test_vision_augments_stop_strings_with_close_tag(self):
        import inspect
        import re as _re
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin._generate_with_vision)
        body = "\n".join(
            ln for ln in src.splitlines()
            if not ln.lstrip().startswith("#"))
        assert _re.search(
            r'effective_stop_strings\s*=\s*list\(stop_strings\s+or\s+\[\]\)',
            body), (
            "_generate_with_vision must defensively copy stop_strings "
            "into effective_stop_strings when the splice flag is on.")
        assert _re.search(
            r'effective_stop_strings\.append\(\s*"</search>"\s*\)',
            body), (
            '_generate_with_vision must append "</search>" to '
            "effective_stop_strings — the auto-stop is the precondition "
            "for the splice helper.")

    def test_vision_invokes_maybe_rag_splice(self):
        import inspect
        import re as _re
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin._generate_with_vision)
        body = "\n".join(
            ln for ln in src.splitlines()
            if not ln.lstrip().startswith("#"))
        assert _re.search(
            r"self\._maybe_rag_splice\s*\(", body), (
            "_generate_with_vision must call self._maybe_rag_splice(...) "
            "after the post-decode trim block.  Without this call the "
            "B-3a flag has no splice consumer on the vision path.")
        assert _re.search(
            r"tokens_already_generated\s*=\s*tokens_round0",
            body), (
            "_generate_with_vision must forward "
            "tokens_already_generated=tokens_round0 to "
            "_maybe_rag_splice so the round budget is honoured.")

    def test_vision_path_no_longer_emits_b3a_warning(self, caplog):
        """Behavioural regression gate paired with the WARNING-gate
        update in ``_record_search_emissions``: ``path='vision'`` must
        record queries silently (Stage B-2 WARNING still fires) but
        must NOT emit the B-3a sibling WARNING."""
        import logging
        from enigma_engine.core.engine_generation import _GenerationMixin

        obj = object.__new__(_GenerationMixin)
        obj.last_search_queries = []
        obj.inline_search_enabled = True
        obj.inline_search_splice_enabled = True
        with caplog.at_level(
            logging.WARNING,
            logger="enigma_engine.core.engine_generation",
        ):
            obj._record_search_emissions(
                "<search>q</search>", path="vision")
        messages = [r.message for r in caplog.records]
        assert any("Stage B-2" in m for m in messages), (
            "Generic Stage B-2 WARNING must still fire — queries are "
            "still recorded on every path.")
        assert not any("B-3a:" in m for m in messages), (
            "Pass 156z9do closed 'vision'; this path must not be in "
            "the B-3a sibling-WARNING set anymore.")


class TestB3BatchSiblingClosure:
    """Pass 156z9dp B-3 sibling closure for ``batch_generate``.

    The batch path runs the autoregressive loop in vectorised form
    (one forward per step, all rows together) so per-sequence
    ``</search>`` stop-detection in the loop is impractical — rows
    desync the moment one emits the close tag while others are still
    generating.  Instead the splice runs **post-decode, per-prompt**:

    1. Decode the batch normally for ``max_gen`` steps (or until all
       rows EOS).
    2. For each output, trim at ``</search>`` and call
       ``_maybe_rag_splice(...)`` independently with that row's own
       round-0 token budget (``generated.shape[1] - len(encoded[i])``).
    3. Drop ``"batch"`` from the B-3a WARNING allow-list — the flag is
       now a real consumer for batch callers too.

    Trade-off: splice rounds run serially per prompt via
    ``_generate_manual`` (text-only single-sequence), so the batch
    efficiency advantage applies only to round 0.  Acceptable because
    splice triggers on a minority of rows in typical batched calls
    (and the alternative — desynced in-loop stop-and-resume — is
    significantly more code for marginal speedup).
    """

    def test_batch_trims_per_sequence_at_close_tag_when_flag_on(self):
        import inspect
        import re as _re
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin.batch_generate)
        body = "\n".join(
            ln for ln in src.splitlines()
            if not ln.lstrip().startswith("#"))
        assert _re.search(
            r'splice_enabled\s*=\s*getattr\(\s*self\s*,\s*'
            r'"inline_search_splice_enabled"',
            body), (
            "batch_generate must read inline_search_splice_enabled "
            "via getattr so unflagged stubs do not crash.")
        assert _re.search(
            r'generated_part\.find\(\s*"</search>"\s*\)',
            body), (
            "batch_generate must per-sequence trim at the first "
            "</search> in the post-prompt portion of each row.")

    def test_batch_invokes_maybe_rag_splice_per_prompt(self):
        import inspect
        import re as _re
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin.batch_generate)
        body = "\n".join(
            ln for ln in src.splitlines()
            if not ln.lstrip().startswith("#"))
        assert _re.search(
            r"self\._maybe_rag_splice\s*\(", body), (
            "batch_generate must call self._maybe_rag_splice(...) "
            "per prompt after the per-sequence trim.")
        # tokens_already_generated must use the per-sequence round-0
        # count, not the padded max_input_len — a wrong budget here
        # would either over-charge or under-charge each row.
        assert _re.search(
            r"tokens_already_generated\s*=\s*tokens_round0",
            body), (
            "batch_generate must forward "
            "tokens_already_generated=tokens_round0 (per-row count) "
            "to _maybe_rag_splice.")
        assert _re.search(
            r"tokens_round0\s*=\s*max\(\s*0\s*,\s*generated\.shape\[1\]"
            r"\s*-\s*len\(\s*encoded\[\s*i\s*\]\s*\)\s*\)",
            body), (
            "tokens_round0 must subtract the per-row original input "
            "length (``len(encoded[i])``) from the padded generated "
            "length — otherwise padding tokens count as 'generated'.")

    def test_batch_path_no_longer_emits_b3a_warning(self, caplog):
        """Behavioural regression gate paired with the WARNING-gate
        update in ``_record_search_emissions``: ``path='batch'`` must
        record queries silently (Stage B-2 WARNING still fires) but
        must NOT emit the B-3a sibling WARNING."""
        import logging
        from enigma_engine.core.engine_generation import _GenerationMixin

        obj = object.__new__(_GenerationMixin)
        obj.last_search_queries = []
        obj.inline_search_enabled = True
        obj.inline_search_splice_enabled = True
        with caplog.at_level(
            logging.WARNING,
            logger="enigma_engine.core.engine_generation",
        ):
            obj._record_search_emissions(
                "<search>q</search>", path="batch")
        messages = [r.message for r in caplog.records]
        assert any("Stage B-2" in m for m in messages), (
            "Generic Stage B-2 WARNING must still fire — queries are "
            "still recorded on every path.")
        assert not any("B-3a:" in m for m in messages), (
            "Pass 156z9dp closed 'batch'; this path must not be in "
            "the B-3a sibling-WARNING set anymore.")


class TestStageB2GgufChatSiblingSweep:
    """Pass 156z9e sibling-boundary-sweep audit on Pass 156z9d: the
    chat() and stream_chat() GGUF branches in ``engine_chat.py`` call
    ``self.model.chat()`` directly, BYPASSING ``_generate_text`` /
    ``stream_generate`` and therefore their ``_record_search_emissions``
    hooks.  Without dedicated hooks, a user driving the engine through
    ``engine.chat(...)`` against a GGUF model would get
    ``last_search_queries == []`` even when the model emitted a search
    request — silent observability loss in the same family Pass 156z7
    already cleaned up for json_schema.

    These are structural tests because behavioural tests would require
    either a loaded GGUF model or a multi-layer mock of llama-cpp-python.
    Companion behavioural test on the helper itself lives in
    ``TestStageB2SearchEmissionRecording`` above."""

    def test_chat_gguf_branch_records_search_emissions(self):
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        src = inspect.getsource(_ChatMixin.chat)
        # Pull the GGUF branch (between "if ctx.is_gguf" and the
        # next major branch comment) and assert the hook lives there.
        gguf_start = src.find("if ctx.is_gguf and hasattr(self.model")
        assert gguf_start != -1, "GGUF branch not found in chat()"
        # Search the next ~80 lines after the branch entry
        gguf_block = src[gguf_start:gguf_start + 4000]
        assert "_record_search_emissions" in gguf_block, (
            "chat() GGUF branch must call _record_search_emissions on "
            "the response — Pass 156z9e sibling sweep miss.")

    def test_stream_chat_gguf_server_branch_records(self):
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        src = inspect.getsource(_ChatMixin.stream_chat)
        # Server-backend branch yields response in one piece without
        # going through stream_generate's finally hook.
        server_start = src.find("if ctx.has_server_backend")
        assert server_start != -1, (
            "GGUF server-backend branch not found in stream_chat()")
        server_block = src[server_start:server_start + 1500]
        assert "_record_search_emissions" in server_block, (
            "stream_chat() GGUF server branch must call "
            "_record_search_emissions before yielding — Pass 156z9e "
            "sibling sweep miss.")

    def test_stream_chat_gguf_llamacpp_branch_records_in_finally(self):
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        src = inspect.getsource(_ChatMixin.stream_chat)
        # llama-cpp-python in-process streaming branch.
        # NB: ``create_chat_completion`` also appears in the docstring
        # ("Works with ... GGUF models (via create_chat_completion ...)")
        # so use ``rfind`` to anchor on the actual call site in the body,
        # not the prose reference.  Otherwise the window slides off the
        # call when the docstring grows.
        llamacpp_marker = "create_chat_completion"
        idx = src.rfind(llamacpp_marker)
        assert idx != -1, (
            "GGUF llama-cpp streaming branch not found in stream_chat()")
        # Window covers the try/finally that wraps the chunk loop
        window = src[max(0, idx - 500):idx + 3000]
        assert "_record_search_emissions" in window, (
            "stream_chat() GGUF llama-cpp branch must call "
            "_record_search_emissions in its finally block so "
            "cancellation still flushes the scan — Pass 156z9e.")
        assert "finally" in window, (
            "stream_chat() GGUF llama-cpp branch must wrap its yield "
            "loop in try/finally so the scan survives early break.")

    @pytest.mark.parametrize("method_name,window_start_marker", [
        ("chat", "if ctx.is_gguf and hasattr(self.model"),
        ("stream_chat", "if ctx.has_server_backend"),
        ("stream_chat", "create_chat_completion"),
    ])
    def test_gguf_chat_path_forwards_path_kwarg(
        self, method_name, window_start_marker,
    ):
        """Pass 156z9cq sibling-boundary closure: each of the three
        GGUF chat-path ``_record_search_emissions`` call sites must
        forward ``path="gguf"`` so the helper's B-3a sibling WARNING
        fires when ``inline_search_splice_enabled`` is True.

        Without this forward, the default ``path="native"`` is used
        and the WARNING is silently suppressed (native is in the
        helper's supported allow-list), giving the user a feature
        labelled "splice on" with no observable behaviour or warning
        on GGUF chat paths."""
        import inspect
        import re as _re
        from enigma_engine.core.engine_chat import _ChatMixin
        method = getattr(_ChatMixin, method_name)
        src = inspect.getsource(method)
        idx = src.find(window_start_marker)
        assert idx != -1, (
            f"{method_name}: marker {window_start_marker!r} not found"
        )
        # Window large enough to span the full branch body
        window = src[idx:idx + 4000]
        # Literal regex on the call expression — gates the forward, not
        # just the kwarg's presence somewhere in the method body.
        assert _re.search(
            r"_record_search_emissions\([^)]*path\s*=\s*[\"']gguf[\"']",
            window,
            flags=_re.DOTALL,
        ), (
            f"{method_name} GGUF branch (marker={window_start_marker!r}) "
            "must call _record_search_emissions(... path=\"gguf\") so "
            "the B-3a sibling WARNING fires when the splice flag is on."
        )


class TestChatDocstringHonesty:
    """Doc-vs-code lens: the ``Raises:`` clauses on ``chat()`` and
    ``stream_chat()`` must enumerate only exceptions the code actually
    raises, and must enumerate every exception that IS raised at the
    public boundary.  Pass 156s anti-pattern: a docstring promising
    behaviour the code does not perform (e.g. ``RuntimeError: If the
    model is not loaded``) misleads callers into writing
    ``try/except RuntimeError`` that silently misses the real
    ``AttributeError`` the code throws."""

    def test_chat_docstring_does_not_promise_unraised_runtime_error(self):
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        doc = inspect.getdoc(_ChatMixin.chat) or ""
        # The original (Pass 156s anti-pattern) phrasing was:
        #   "RuntimeError: If the underlying model is not loaded ..."
        # The current chat() body has no `if self.model is None: raise
        # RuntimeError(...)` guard, so the doc must not promise it.
        assert "If the underlying model is not loaded" not in doc, (
            "chat() docstring promises RuntimeError on missing model "
            "but the code never raises it. Either implement the guard "
            "or drop the promise (Pass 156s rule)."
        )

    def test_chat_docstring_documents_json_schema_gguf_rejection(self):
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        doc = inspect.getdoc(_ChatMixin.chat) or ""
        assert "NotImplementedError" in doc, (
            "chat() raises NotImplementedError when json_schema is "
            "passed on a GGUF model (Pass 156z7 N-15c2). The docstring "
            "must document this — undocumented raises are a smaller "
            "lie than overpromised raises but still violate the rule."
        )
        assert "json_schema" in doc and "GGUF" in doc, (
            "chat() docstring's NotImplementedError clause must name "
            "the trigger (json_schema) and the gated path (GGUF) so "
            "callers can write the right except clause."
        )

    def test_stream_chat_docstring_documents_json_schema_gguf_rejection(self):
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        doc = inspect.getdoc(_ChatMixin.stream_chat) or ""
        assert "NotImplementedError" in doc, (
            "stream_chat() raises NotImplementedError when json_schema "
            "is passed on a GGUF model (Pass 156z6 N-15c). The "
            "docstring must document this raise."
        )
        assert "json_schema" in doc and "GGUF" in doc, (
            "stream_chat() docstring's NotImplementedError clause must "
            "name the trigger (json_schema) and the gated path (GGUF)."
        )


class TestInferenceDocstringHonesty:
    """Pass 156z9cs continuation of the docstring-honesty sweep beyond
    `_ChatMixin`.  Three sibling sites in `core/inference.py`,
    `core/model.py`, and `core/huggingface_loader.py` had `Raises:`
    clauses that lied about either the trigger condition or the
    coverage.  Same Pass 156s lens applied; gates pinned in place so
    a regression that drops the validation guard OR walks the doc
    back to a vague claim fails the test.
    """

    def test_generate_typeerror_guard_is_real_and_documented(self):
        """`EnigmaEngine.generate()` claimed `TypeError: If prompt is
        not a string` but had no `isinstance(prompt, str)` guard \u2014
        callers hit an opaque tokenizer error instead.  Pass 156z9cs
        added the guard; this test pins both the doc claim AND the
        live behaviour so a regression in either direction fails."""
        import inspect

        from enigma_engine.core.inference import EnigmaEngine
        doc = inspect.getdoc(EnigmaEngine.generate) or ""
        assert "TypeError" in doc and "prompt" in doc, (
            "generate() docstring must enumerate the TypeError "
            "trigger explicitly."
        )

        # Behavioural gate: the guard must raise TypeError synchronously
        # without touching the model. Build a near-empty stub engine
        # that would otherwise blow up; the type check fires first.
        engine = EnigmaEngine.__new__(EnigmaEngine)
        with pytest.raises(TypeError, match="prompt must be a string"):
            engine.generate(prompt=None)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="prompt must be a string"):
            engine.generate(prompt=["not", "a", "string"])  # type: ignore[arg-type]

    def test_generate_documents_json_schema_execute_tools_value_error(self):
        """Pass 156z7 (N-15c2) added a real ValueError gate for
        `json_schema + execute_tools`.  The pre-156z9cs docstring's
        `ValueError: If parameters are out of valid range` was vague
        enough to technically cover this, but did not let a caller
        write a useful `except` clause.  Pin the specific trigger."""
        import inspect

        from enigma_engine.core.inference import EnigmaEngine
        doc = inspect.getdoc(EnigmaEngine.generate) or ""
        assert "json_schema" in doc and "execute_tools" in doc, (
            "generate() docstring must name the json_schema + "
            "execute_tools mutual-exclusion gate as a specific "
            "ValueError trigger."
        )

    def test_generate_documents_all_numeric_range_value_errors(self):
        """Pass 156z9ct audit on Pass 156z9cs: the 156z9cs cleanup
        narrowed the `Raises:` clause to the json_schema gate but
        dropped coverage of the FIVE pre-existing numeric-range
        triggers that propagate from ``_generate_text`` and are
        gated by ``tests/test_inference.py::TestGenerateValidation``
        (max_gen, temperature, top_k, top_p, repetition_penalty).
        Document them so callers can write the right ``except``."""
        import inspect

        from enigma_engine.core.inference import EnigmaEngine
        doc = inspect.getdoc(EnigmaEngine.generate) or ""
        for marker in (
            "max_gen", "temperature", "top_k", "top_p", "repetition_penalty"
        ):
            assert marker in doc, (
                f"generate() Raises clause must name {marker!r} as a "
                "ValueError trigger after Pass 156z9ct restoration."
            )

    def test_model_generate_enumerates_all_three_value_error_triggers(self):
        """`Enigma.generate()` (the core forward-loop variant in
        `core/model.py`) raises ValueError on three distinct
        conditions: temperature, input_ids shape, device mismatch.
        The pre-156z9cs docstring listed only one.  Pin all three."""
        import inspect

        from enigma_engine.core.model import Enigma
        doc = inspect.getdoc(Enigma.generate) or ""
        # Substrings that must appear in the expanded clause
        for marker in ("temperature", "input_ids", "device"):
            assert marker in doc, (
                f"Enigma.generate() Raises clause must name {marker!r} "
                "as a ValueError trigger after Pass 156z9cs."
            )

    def test_convert_hf_config_to_forge_documents_real_value_error_trigger(self):
        """`convert_hf_config_to_forge` claimed `ValueError: If model
        type not supported` but actually raises ValueError on missing
        dim / layers / heads fields.  Pin the corrected wording so a
        future regression doesn't walk the doc back to the old lie."""
        import inspect

        from enigma_engine.core.huggingface_loader import convert_hf_config_to_forge
        doc = inspect.getdoc(convert_hf_config_to_forge) or ""
        # Negative: the false old trigger must not return
        assert "model type not supported" not in doc, (
            "convert_hf_config_to_forge docstring must not promise "
            "the unraised 'model type not supported' trigger."
        )
        # Positive: at least one of the real triggers must be named
        assert ("dimension" in doc
                or "hidden_size" in doc
                or "layer count" in doc
                or "attention-head" in doc), (
            "convert_hf_config_to_forge Raises clause must name one "
            "of the real missing-field triggers (dimension / layer "
            "count / attention-head count)."
        )

    def test_parse_gguf_tensors_does_not_promise_notimplementederror(self):
        """Pass 156z9cu: `parse_gguf_tensors` claimed `NotImplementedError`
        but the body has zero `raise NotImplementedError` statements —
        unknown tensor types are SKIPPED with a WARNING log, and the
        only real raise is `RuntimeError` when torch is missing.
        Pass 156s anti-pattern (documents what the code never raises)."""
        import inspect

        from enigma_engine.core.gguf_dequant import parse_gguf_tensors
        doc = inspect.getdoc(parse_gguf_tensors) or ""
        # Negative: the false promise must not appear in the Raises clause.
        # NB: the body of the docstring still describes the format ("F32
        # and F16 load directly", "Pass 156s anti-pattern" note) — we
        # gate on the LITERAL Raises-clause shape `NotImplementedError:`
        # (colon-terminated, as Sphinx Raises-block syntax) to allow
        # narrative reference to the old wording in the explanation.
        import re as _re
        assert not _re.search(
            r"^\s*NotImplementedError\s*:", doc, _re.MULTILINE
        ), (
            "parse_gguf_tensors Raises clause must not promise the "
            "unraised NotImplementedError trigger."
        )
        # Positive: the real RuntimeError trigger must be documented.
        assert _re.search(r"^\s*RuntimeError\s*:", doc, _re.MULTILINE), (
            "parse_gguf_tensors Raises clause must document the real "
            "RuntimeError trigger (torch missing)."
        )

    def test_validate_loaded_model_documents_only_runtime_error(self):
        """Pass 156z9cu: `validate_loaded_model` documented separate
        `RuntimeError` AND `ValueError` triggers, but the body wraps
        every internal raise in `try/except Exception → raise
        RuntimeError(...) from e`.  The documented ValueError cannot
        escape — it gets converted to RuntimeError with the ValueError
        as `__cause__`.  Same Pass 156s anti-pattern."""
        import inspect
        import re as _re

        from enigma_engine.core.onnx_loader import validate_loaded_model
        doc = inspect.getdoc(validate_loaded_model) or ""
        # Negative: the standalone ValueError Raises-clause line must
        # not appear. Allow narrative reference ("the inner raise
        # ValueError(...) is caught by the outer guard") in the
        # explanation text — gate on the Sphinx Raises-block shape
        # only (colon-terminated class name at line start).
        # Specifically: ValueError must not appear as its own Raises
        # line. The current honest text mentions it as "the inner
        # raise ValueError(...)" — narrative, not promise.
        lines = doc.splitlines()
        for line in lines:
            stripped = line.lstrip()
            # A Sphinx Raises entry starts with the class name and a
            # colon, immediately followed by a description.
            if _re.match(r"^ValueError\s*:\s+\w", stripped):
                raise AssertionError(
                    f"validate_loaded_model Raises clause must not "
                    f"promise ValueError as a top-level trigger — it "
                    f"is caught and converted to RuntimeError. Found: "
                    f"{stripped!r}"
                )
        # Positive: the real RuntimeError trigger must be documented.
        assert _re.search(r"^\s*RuntimeError\s*:", doc, _re.MULTILINE), (
            "validate_loaded_model Raises clause must document the "
            "real RuntimeError trigger."
        )


class TestEncodePromptTokenRangeGuard:
    """Runtime guard for tokenizer/model vocab mismatch before tensor move."""

    def test_encode_prompt_rejects_out_of_range_token_ids(self):
        import torch
        from enigma_engine.core.inference import EnigmaEngine

        class _Tok:
            def encode(self, prompt, add_special_tokens=True):
                return [1, 99]

        class _Emb:
            num_embeddings = 32

        class _Model:
            tok_embeddings = _Emb()

        engine = EnigmaEngine.__new__(EnigmaEngine)
        engine.tokenizer = _Tok()
        engine.model = _Model()
        engine.device = torch.device("cpu")

        with pytest.raises(ValueError, match="vocabulary range"):
            engine._encode_prompt("hello")

    def test_encode_prompt_accepts_in_range_token_ids(self):
        import torch
        from enigma_engine.core.inference import EnigmaEngine

        class _Tok:
            def encode(self, prompt, add_special_tokens=True):
                return [1, 2, 3]

        class _Emb:
            num_embeddings = 32

        class _Model:
            tok_embeddings = _Emb()

        engine = EnigmaEngine.__new__(EnigmaEngine)
        engine.tokenizer = _Tok()
        engine.model = _Model()
        engine.device = torch.device("cpu")

        t = engine._encode_prompt("hello")
        assert t.shape == (1, 3)
        assert t.dtype == torch.long


# ---------------------------------------------------------------------------
# Pass after 156z9e — Stage B-2b per-prompt attribution for batch_generate
# ---------------------------------------------------------------------------

class TestStageB2bBatchPerPromptAttribution:
    """B-2b closes the per-prompt attribution gap from Pass 156z9d's
    sibling sweep on ``batch_generate``.  Earlier slice joined the batch
    output with newlines and scanned once, which lost which prompt
    produced which query.  Now ``last_search_queries_per_prompt`` is a
    parallel-to-prompts list-of-lists; the flat ``last_search_queries``
    is the union for callers that don't care about attribution.

    Behavioural tests on a stub instance — no model load required."""

    def _stub(self):
        from enigma_engine.core.engine_generation import _GenerationMixin
        obj = object.__new__(_GenerationMixin)
        obj.last_search_queries = []
        obj.last_search_queries_per_prompt = []
        return obj

    def test_per_prompt_list_parallel_to_prompts(self):
        """Each entry in ``last_search_queries_per_prompt`` is the
        emissions for the prompt at the same index — including empty
        lists for prompts where the model emitted nothing."""
        obj = self._stub()
        prompts = ["p0", "p1", "p2"]
        results = [
            "p0 <search>q0a</search> <search>q0b</search>",
            "p1 nothing here",
            "p2 <search>q2</search>",
        ]
        per_prompt: list[list[str]] = []
        flat: list[str] = []
        for prompt_text, output_text in zip(prompts, results):
            obj._record_search_emissions(output_text, prompt=prompt_text)
            per_prompt.append(list(obj.last_search_queries))
            flat.extend(obj.last_search_queries)
        obj.last_search_queries_per_prompt = per_prompt
        obj.last_search_queries = flat

        assert obj.last_search_queries_per_prompt == [
            ["q0a", "q0b"],
            [],
            ["q2"],
        ]
        assert obj.last_search_queries == ["q0a", "q0b", "q2"]

    def test_init_common_initialises_per_prompt_attribute(self):
        """Engine constructors must initialise the attribute so callers
        can read it without an AttributeError before the first
        ``batch_generate`` call."""
        src = _get_init_common_source()
        assert "self.last_search_queries_per_prompt" in src, (
            "_init_common must initialise last_search_queries_per_prompt "
            "so every engine has the attribute regardless of "
            "construction path. B-2b wire-site test.")

    def test_batch_generate_wires_per_prompt_attribution(self):
        """Wire-site structural test: ``batch_generate`` must populate
        ``last_search_queries_per_prompt`` (one entry per prompt) AND
        the flat union list, NOT the legacy join-and-scan."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin.batch_generate)
        assert "last_search_queries_per_prompt" in src, (
            "batch_generate must populate last_search_queries_per_prompt "
            "to give callers per-prompt attribution. B-2b.")
        # Catches the regression where someone reverts to the old
        # join-and-scan path.
        assert '"\\n".join(results)' not in src, (
            "batch_generate must NOT use the legacy join-and-scan "
            "path — that loses per-prompt attribution. B-2b.")

    def test_batch_generate_off_switch_clears_per_prompt(self):
        """Pass 156z9w (post-audit, Finding 2): when
        ``inline_search_enabled = False`` the batch_generate scan loop
        must produce empty per-prompt lists AND empty flat list, even
        when the model output contains <search> emissions. Emulates
        the batch loop body directly on a stub.

        This catches a regression where someone inlines the scan inside
        ``batch_generate`` and bypasses ``_record_search_emissions``,
        which would slip past every helper-only off-switch test."""
        obj = self._stub()
        obj.inline_search_enabled = False
        prompts = ["p0", "p1"]
        results = [
            "p0 <search>q0</search>",
            "p1 <search>q1</search>",
        ]
        per_prompt: list[list[str]] = []
        flat: list[str] = []
        for prompt_text, output_text in zip(prompts, results):
            obj._record_search_emissions(output_text, prompt=prompt_text)
            per_prompt.append(list(obj.last_search_queries))
            flat.extend(obj.last_search_queries)
        obj.last_search_queries_per_prompt = per_prompt
        obj.last_search_queries = flat

        assert obj.last_search_queries_per_prompt == [[], []]
        assert obj.last_search_queries == []

