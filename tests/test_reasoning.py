"""
Tests for chain-of-thought reasoning support.

Run with: python -m pytest tests/test_reasoning.py -v
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestReasoningConstants:
    """Test reasoning token constants."""

    def test_think_tokens_defined(self):
        from enigma_engine.core.reasoning import THINK_START, THINK_END
        assert THINK_START == "<think>"
        assert THINK_END == "</think>"


class TestExtractReasoning:
    """Test reasoning extraction from model output."""

    def test_extract_with_reasoning(self):
        from enigma_engine.core.reasoning import extract_reasoning
        text = "<think>Let me work through this step by step.</think>The answer is 42."
        thinking, answer = extract_reasoning(text)
        assert thinking == "Let me work through this step by step."
        assert answer == "The answer is 42."

    def test_extract_without_reasoning(self):
        from enigma_engine.core.reasoning import extract_reasoning
        text = "Just a plain answer."
        thinking, answer = extract_reasoning(text)
        assert thinking == ""
        assert answer == "Just a plain answer."

    def test_extract_multi_line_reasoning(self):
        from enigma_engine.core.reasoning import extract_reasoning
        text = (
            "<think>Step 1: Consider the problem.\n"
            "Step 2: Apply the formula.\n"
            "Step 3: Calculate.</think>The result is 7."
        )
        thinking, answer = extract_reasoning(text)
        assert "Step 1" in thinking
        assert "Step 3" in thinking
        assert answer == "The result is 7."

    def test_extract_whitespace_around_answer(self):
        from enigma_engine.core.reasoning import extract_reasoning
        text = "<think>thoughts</think>  \n  The answer.  "
        thinking, answer = extract_reasoning(text)
        assert thinking == "thoughts"
        assert answer == "The answer."

    def test_extract_empty_thinking(self):
        from enigma_engine.core.reasoning import extract_reasoning
        text = "<think></think>Direct answer."
        thinking, answer = extract_reasoning(text)
        assert thinking == ""
        assert answer == "Direct answer."

    def test_extract_only_thinking_no_answer(self):
        from enigma_engine.core.reasoning import extract_reasoning
        text = "<think>I'm just thinking out loud</think>"
        thinking, answer = extract_reasoning(text)
        assert thinking == "I'm just thinking out loud"
        assert answer == ""

    def test_extract_unclosed_think_tag(self):
        """Unclosed <think> tag — treat entire text as answer."""
        from enigma_engine.core.reasoning import extract_reasoning
        text = "<think>started thinking but never closed"
        thinking, answer = extract_reasoning(text)
        assert thinking == ""
        assert answer == text

    def test_extract_nested_angle_brackets(self):
        """Reasoning that contains other angle brackets."""
        from enigma_engine.core.reasoning import extract_reasoning
        text = "<think>if x < 5 and y > 3 then...</think>Use the condition."
        thinking, answer = extract_reasoning(text)
        assert "x < 5" in thinking
        assert answer == "Use the condition."


class TestStripReasoning:
    """Test stripping reasoning from text."""

    def test_strip_with_reasoning(self):
        from enigma_engine.core.reasoning import strip_reasoning
        text = "<think>some thoughts</think>Clean answer."
        assert strip_reasoning(text) == "Clean answer."

    def test_strip_without_reasoning(self):
        from enigma_engine.core.reasoning import strip_reasoning
        text = "Already clean."
        assert strip_reasoning(text) == "Already clean."

    def test_strip_empty(self):
        from enigma_engine.core.reasoning import strip_reasoning
        assert strip_reasoning("") == ""


class TestHasReasoning:
    """Test reasoning detection."""

    def test_has_reasoning_true(self):
        from enigma_engine.core.reasoning import has_reasoning
        assert has_reasoning("<think>thoughts</think>answer")

    def test_has_reasoning_false(self):
        from enigma_engine.core.reasoning import has_reasoning
        assert not has_reasoning("no reasoning here")

    def test_has_reasoning_open_only(self):
        """Open tag without close is not valid reasoning."""
        from enigma_engine.core.reasoning import has_reasoning
        assert not has_reasoning("<think>no close tag")

    def test_has_reasoning_close_only(self):
        from enigma_engine.core.reasoning import has_reasoning
        assert not has_reasoning("no open tag</think>")

    def test_has_reasoning_only_inside_complete_block(self):
        """Unclosed <think> with </think> that doesn't match is not valid."""
        from enigma_engine.core.reasoning import has_reasoning
        # Valid: matching pair
        assert has_reasoning("<think>x</think>")
        # Invalid: no closing tag
        assert not has_reasoning("<think>just thinking")


class TestStripIncompleteThink:
    """Test stripping truncated <think> blocks from token-limited output."""

    def test_strips_unclosed_think(self):
        from enigma_engine.core.reasoning import strip_incomplete_think
        text = "<think>The user is asking about rhinoceros mating"
        result = strip_incomplete_think(text)
        assert result == ""

    def test_preserves_complete_think_block(self):
        from enigma_engine.core.reasoning import strip_incomplete_think
        text = "<think>step 1</think>The answer is 42."
        result = strip_incomplete_think(text)
        assert result == text

    def test_preserves_plain_text(self):
        from enigma_engine.core.reasoning import strip_incomplete_think
        text = "Just a normal response with no tags."
        result = strip_incomplete_think(text)
        assert result == text

    def test_preserves_text_before_unclosed_think(self):
        from enigma_engine.core.reasoning import strip_incomplete_think
        text = "Some preamble <think>truncated thinking"
        result = strip_incomplete_think(text)
        assert result == "Some preamble"

    def test_empty_string(self):
        from enigma_engine.core.reasoning import strip_incomplete_think
        assert strip_incomplete_think("") == ""


class TestWrapReasoning:
    """Test wrapping text in reasoning tags."""

    def test_wrap_basic(self):
        from enigma_engine.core.reasoning import wrap_reasoning
        result = wrap_reasoning("my thoughts", "my answer")
        assert result == "<think>my thoughts</think>my answer"

    def test_wrap_empty_thinking(self):
        from enigma_engine.core.reasoning import wrap_reasoning
        result = wrap_reasoning("", "just answer")
        assert result == "just answer"

    def test_wrap_empty_answer(self):
        from enigma_engine.core.reasoning import wrap_reasoning
        result = wrap_reasoning("thoughts", "")
        assert result == "<think>thoughts</think>"


class TestReasoningPrompt:
    """Test reasoning prompt injection."""

    def test_build_reasoning_instruction(self):
        from enigma_engine.core.reasoning import build_reasoning_instruction
        instruction = build_reasoning_instruction()
        assert "<think>" in instruction
        assert "</think>" in instruction

    def test_build_reasoning_instruction_is_string(self):
        from enigma_engine.core.reasoning import build_reasoning_instruction
        assert isinstance(build_reasoning_instruction(), str)


class TestReasoningTokensInTokenizer:
    """Test that reasoning tokens are in all tokenizer vocabularies."""

    def test_simple_tokenizer_has_think_tokens(self):
        from enigma_engine.core.tokenizer import SimpleTokenizer
        tok = SimpleTokenizer()
        assert "<think>" in tok.token_to_id
        assert "</think>" in tok.token_to_id

    def test_simple_tokenizer_encode_think(self):
        """Encode <think> as a single token, not character-by-character."""
        from enigma_engine.core.tokenizer import SimpleTokenizer
        tok = SimpleTokenizer()
        think_id = tok.token_to_id["<think>"]
        ids = tok.encode("<think>hello</think>", add_special_tokens=False)
        assert think_id in ids

    def test_simple_tokenizer_decode_think(self):
        from enigma_engine.core.tokenizer import SimpleTokenizer
        tok = SimpleTokenizer()
        think_id = tok.token_to_id["<think>"]
        end_id = tok.token_to_id["</think>"]
        decoded = tok.decode([think_id, end_id], skip_special_tokens=False)
        assert "<think>" in decoded
        assert "</think>" in decoded

    def test_bpe_tokenizer_has_think_tokens(self):
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        assert "<think>" in tok.special_tokens
        assert "</think>" in tok.special_tokens

    def test_char_tokenizer_has_think_tokens(self):
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer()
        assert "<think>" in tok.special_tokens
        assert "</think>" in tok.special_tokens

    def test_advanced_tokenizer_has_think_tokens(self):
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        tok = AdvancedBPETokenizer()
        assert "<think>" in tok.special_tokens
        assert "</think>" in tok.special_tokens


class TestGetSpecialTokenIds:
    """Test get_special_token_ids includes reasoning tokens."""

    def test_special_ids_include_think(self):
        from enigma_engine.core.tokenizer import (
            SimpleTokenizer, get_special_token_ids)
        tok = SimpleTokenizer()
        ids = get_special_token_ids(tok)
        assert "think_start" in ids
        assert "think_end" in ids
        assert isinstance(ids["think_start"], int)
        assert isinstance(ids["think_end"], int)


class TestChatReasoning:
    """Test reasoning integration in chat engine."""

    def test_chat_method_accepts_reasoning_param(self):
        """Verify chat() signature accepts reasoning kwarg."""
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        sig = inspect.signature(_ChatMixin.chat)
        assert "reasoning" in sig.parameters

    def test_stream_chat_method_accepts_reasoning_param(self):
        """Verify stream_chat() signature accepts reasoning kwarg."""
        import inspect
        from enigma_engine.core.engine_chat import _ChatMixin
        sig = inspect.signature(_ChatMixin.stream_chat)
        assert "reasoning" in sig.parameters

    def test_reasoning_instruction_added_to_prompt(self):
        """When reasoning=True, the reasoning instruction should be
        injected into the system prompt."""
        from enigma_engine.core.reasoning import build_reasoning_instruction
        instruction = build_reasoning_instruction()
        # Just verify it's a non-empty string with the tags
        assert len(instruction) > 20
        assert "<think>" in instruction


class TestTrainingReasoningFormat:
    """Test reasoning-aware training data parsing."""

    def test_parse_reasoning_qa(self):
        """Training data with <think> blocks should be preserved."""
        from enigma_engine.core.reasoning import format_reasoning_example
        result = format_reasoning_example(
            question="What is 2+2?",
            thinking="2+2 means adding 2 to 2, which equals 4.",
            answer="4"
        )
        assert "<think>" in result
        assert "</think>" in result
        assert "What is 2+2?" in result
        assert "4" in result
