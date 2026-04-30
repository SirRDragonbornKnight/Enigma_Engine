"""
Tests for chain-of-thought reasoning support.

Run with: python -m pytest tests/test_reasoning.py -v
"""

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


class TestSearchTokens:
    """AutoResearch-2 Stage B-1: <search>/</search> token primitives."""

    def test_search_constants_defined(self):
        from enigma_engine.core.reasoning import SEARCH_END, SEARCH_START
        assert SEARCH_START == "<search>"
        assert SEARCH_END == "</search>"

    def test_extract_single_query(self):
        from enigma_engine.core.reasoning import extract_search_queries
        text = "I need to look this up <search>weather in Tokyo</search>"
        assert extract_search_queries(text) == ["weather in Tokyo"]

    def test_extract_multiple_queries_in_order(self):
        from enigma_engine.core.reasoning import extract_search_queries
        text = "<search>first</search> middle <search>second</search>"
        assert extract_search_queries(text) == ["first", "second"]

    def test_extract_no_queries(self):
        from enigma_engine.core.reasoning import extract_search_queries
        assert extract_search_queries("plain answer") == []

    def test_extract_strips_whitespace(self):
        from enigma_engine.core.reasoning import extract_search_queries
        text = "<search>   padded query   </search>"
        assert extract_search_queries(text) == ["padded query"]

    def test_extract_ignores_unclosed(self):
        from enigma_engine.core.reasoning import extract_search_queries
        text = "<search>truncated mid-stream"
        assert extract_search_queries(text) == []

    def test_extract_multiline_query(self):
        from enigma_engine.core.reasoning import extract_search_queries
        text = "<search>line one\nline two</search>"
        assert extract_search_queries(text) == ["line one\nline two"]

    def test_strip_search_blocks(self):
        from enigma_engine.core.reasoning import strip_search_blocks
        text = "before <search>q</search> after"
        # Inner whitespace collapses to a double space; outer .strip() trims ends.
        assert strip_search_blocks(text) == "before  after"

    def test_strip_search_blocks_no_match(self):
        from enigma_engine.core.reasoning import strip_search_blocks
        assert strip_search_blocks("plain answer") == "plain answer"

    def test_has_search_request_true(self):
        from enigma_engine.core.reasoning import has_search_request
        assert has_search_request("<search>q</search>") is True

    def test_has_search_request_false(self):
        from enigma_engine.core.reasoning import has_search_request
        assert has_search_request("no tags here") is False

    def test_has_search_request_unclosed(self):
        from enigma_engine.core.reasoning import has_search_request
        assert has_search_request("<search>unclosed") is False

    def test_search_does_not_collide_with_think(self):
        """Stage B-1 contract: <search> and <think> are independent helpers."""
        from enigma_engine.core.reasoning import (
            extract_reasoning,
            extract_search_queries,
        )
        text = "<think>maybe I should look it up</think><search>capital of France</search>The answer is Paris."
        thinking, answer = extract_reasoning(text)
        queries = extract_search_queries(text)
        assert thinking == "maybe I should look it up"
        assert queries == ["capital of France"]
        # Answer still contains the unstripped <search> block (Stage B-1 only
        # registers tokens; downstream consumers strip what they don't want).
        assert "<search>capital of France</search>" in answer


class TestSearchTokenRegistry:
    """Stage B-1: every Python tokenizer registers <search>/</search> on fresh build."""

    def test_simple_tokenizer_registers_search(self):
        from enigma_engine.core.tokenizer import SimpleTokenizer
        tok = SimpleTokenizer()
        assert "<search>" in tok.special_tokens
        assert "</search>" in tok.special_tokens
        assert tok.search_start_id == tok.special_tokens["<search>"]
        assert tok.search_end_id == tok.special_tokens["</search>"]
        # IDs must differ from <think>/</think> so the model can distinguish.
        assert tok.search_start_id != tok.think_start_id
        assert tok.search_end_id != tok.think_end_id

    def test_bpe_tokenizer_registers_search(self):
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        assert "<search>" in tok.special_tokens
        assert "</search>" in tok.special_tokens
        assert tok.search_start_id == tok.special_tokens["<search>"]
        assert tok.search_end_id == tok.special_tokens["</search>"]
        assert tok.search_start_id != tok.think_start_id

    def test_advanced_tokenizer_registers_search(self):
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        tok = AdvancedBPETokenizer()
        assert "<search>" in tok.special_tokens
        assert "</search>" in tok.special_tokens
        assert tok.search_start_id == tok.special_tokens["<search>"]
        assert tok.search_end_id == tok.special_tokens["</search>"]

    def test_char_tokenizer_registers_search(self):
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer()
        assert "<search>" in tok.special_tokens
        assert "</search>" in tok.special_tokens
        assert tok.search_start_id == tok.special_tokens["<search>"]
        assert tok.search_end_id == tok.special_tokens["</search>"]

    def test_legacy_vocab_load_yields_none_search_ids(self, tmp_path):
        """Stage B-1 contract: vocab files saved before <search> existed must
        load with search_start_id/search_end_id == None so Stage B-2's
        generation hook can detect 'feature unavailable on this model' and
        skip its detection logic instead of crashing or aliasing a
        learned-merge ID.
        """
        import json

        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        path = tmp_path / "legacy.json"
        tok.save(path)

        # Simulate a pre-Stage-B-1 vocab file by stripping <search>/</search>
        # from both special_tokens and token_to_id maps on disk.
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("<search>", "</search>"):
            data["special_tokens"].pop(key, None)
            data["token_to_id"].pop(key, None)
        path.write_text(json.dumps(data), encoding="utf-8")

        legacy = BPETokenizer(vocab_file=path)
        assert legacy.search_start_id is None
        assert legacy.search_end_id is None
        # Existing think IDs still work.
        assert legacy.think_start_id == legacy.special_tokens["<think>"]

    def test_simple_tokenizer_legacy_load_yields_none_search_ids(
        self, tmp_path
    ):
        import json

        from enigma_engine.core.tokenizer import SimpleTokenizer
        tok = SimpleTokenizer()
        path = tmp_path / "legacy_vocab.json"
        tok.save_vocab(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("<search>", "</search>"):
            data.pop(key, None)
        path.write_text(json.dumps(data), encoding="utf-8")

        legacy = SimpleTokenizer(vocab_file=path)
        assert legacy.search_start_id is None
        assert legacy.search_end_id is None

    def test_search_token_round_trips_through_save_load(self, tmp_path):
        """Save AND load (Top-9 #3): a fresh tokenizer's <search> ID must
        survive a save/load round trip."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        original_id = tok.search_start_id
        path = tmp_path / "rt.json"
        tok.save(path)

        tok2 = BPETokenizer(vocab_file=path)
        assert tok2.search_start_id == original_id
        assert tok2.search_end_id == tok.search_end_id

    def test_bpe_pre_tokenize_keeps_search_as_single_token(self):
        """Stage B-1 wiring: bpe_tokenizer special_pattern regex must
        recognize <search>/</search> so they survive pre-tokenization
        intact (otherwise BPE would merge their characters with neighbors)."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        # _pre_tokenize is the internal split; we go through the public
        # encode path and assert the single token IDs appear in order.
        ids = tok.encode("ask <search>weather</search> please",
                         add_special_tokens=False)
        assert tok.search_start_id in ids
        assert tok.search_end_id in ids
        # And in the right order (start before end).
        assert ids.index(tok.search_start_id) < ids.index(tok.search_end_id)

    def test_char_tokenizer_keeps_search_as_single_token(self):
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer()
        ids = tok.encode("hello <search>q</search> world")
        assert tok.search_start_id in ids
        assert tok.search_end_id in ids
        assert ids.index(tok.search_start_id) < ids.index(tok.search_end_id)

    def test_advanced_bpe_keeps_search_as_single_token_when_trained(self):
        """Stage B-1 sibling-sweep: AdvancedBPETokenizer's encode builds
        its split regex dynamically from self.special_tokens, but only
        when ``self.merges`` is non-empty (otherwise byte-level
        fallback).  Force the merge path with a dummy merge so the
        regex builder runs and assert <search> survives as one ID."""
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        tok = AdvancedBPETokenizer()
        # Trigger the regex-builder branch.  A single dummy merge that
        # never fires on our test text is enough.
        tok.merges = [("z", "z")]
        tok.merge_ranks = {("z", "z"): 0}
        ids = tok.encode("ask <search>weather</search> please",
                         add_special_tokens=False)
        assert tok.search_start_id in ids
        assert tok.search_end_id in ids
        assert ids.index(tok.search_start_id) < ids.index(tok.search_end_id)
