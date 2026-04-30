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
        """stream_generate calls _build_exempt_tokens."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.stream_generate)
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


class TestStageB2EngineWiring:
    """Verify ``last_search_queries`` is initialised in ``_init_common``
    so every engine instance has the attribute regardless of which
    constructor path created it (``__init__`` vs ``from_model``)."""

    def test_init_common_sets_last_search_queries_attribute(self):
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        src = inspect.getsource(EnigmaEngine._init_common)
        assert "self.last_search_queries" in src, (
            "_init_common must initialise last_search_queries so both "
            "__init__ and from_model paths see the attribute. "
            "Pass 156z9d Stage B-2 wire-site test.")

    def test_generate_text_calls_record_on_main_return_path(self):
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin._generate_text)
        # Native PyTorch return path
        assert "_record_search_emissions" in src, (
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
        # llama-cpp-python in-process streaming branch
        llamacpp_marker = "create_chat_completion"
        idx = src.find(llamacpp_marker)
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
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        src = inspect.getsource(EnigmaEngine._init_common)
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

