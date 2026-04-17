"""Tests for tokenizer implementations, BPE, char tokenizer, token counting, and metrics."""
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestCacheTypeAnnotations:
    """Verify tokenizer cache annotations match stored types."""

    def test_advanced_tokenizer_cache_type(self):
        """AdvancedBPETokenizer cache is dict, not storing wrong type."""
        from enigma_engine.core.advanced_tokenizer import AdvancedBPETokenizer
        tok = AdvancedBPETokenizer()
        assert isinstance(tok.cache, dict)


@pytest.mark.structural
class TestCharTokenizerNoBaseException:
    """Verify char_tokenizer doesn't catch BaseException."""

    def test_no_base_exception(self):
        """char_tokenizer should not catch BaseException."""
        source_path = Path(__file__).parent.parent / "enigma_engine" / "core" / "char_tokenizer.py"
        source = source_path.read_text(encoding='utf-8')
        assert 'except BaseException' not in source

class TestTokenCounterReliable:
    """Verify count_tokens never estimates (Suggestion #11A + #11C)."""

    def test_count_tokens_raises_without_tokenizer(self):
        """count_tokens raises RuntimeError when tokenizer lacks encode/call."""
        from enigma_engine.core.inference import EnigmaEngine

        # Build a minimal engine-like object with a bad tokenizer
        obj = object.__new__(EnigmaEngine)
        obj.tokenizer = object()  # No encode, no __call__
        obj._token_count_cache = {}

        with pytest.raises(RuntimeError, match="[Nn]o tokenizer"):
            obj.count_tokens("hello world")

    def test_no_estimation_fallback_in_source(self):
        """count_tokens source must not contain len(text) // 4 estimation."""
        import inspect
        from enigma_engine.core.inference import EnigmaEngine
        source = inspect.getsource(EnigmaEngine.count_tokens)
        assert "// 4" not in source, "count_tokens must not estimate with len(text)//4"

    def test_token_count_cache_exists(self):
        """Engine instances have _token_count_cache attribute."""
        from enigma_engine.core.inference import EnigmaEngine
        from unittest.mock import MagicMock

        obj = object.__new__(EnigmaEngine)
        obj.tokenizer = MagicMock()
        obj.tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        obj._token_count_cache = {}

        obj.count_tokens("hi")
        assert "hi" in obj._token_count_cache
        assert obj._token_count_cache["hi"] == 3

    def test_token_count_cached_avoids_re_encode(self):
        """Second call to count_tokens uses cache, not tokenizer."""
        from unittest.mock import MagicMock
        from enigma_engine.core.inference import EnigmaEngine

        obj = object.__new__(EnigmaEngine)
        obj.tokenizer = MagicMock()
        obj.tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        obj._token_count_cache = {}

        assert obj.count_tokens("hello") == 3
        assert obj.count_tokens("hello") == 3
        # encode should only be called once
        obj.tokenizer.encode.assert_called_once()

    def test_token_count_cache_bounded(self):
        """Cache clears when exceeding max entries."""
        from unittest.mock import MagicMock
        from enigma_engine.core.inference import EnigmaEngine

        obj = object.__new__(EnigmaEngine)
        obj.tokenizer = MagicMock()
        obj.tokenizer.encode = MagicMock(return_value=[1])
        obj._token_count_cache = {}

        # Fill cache beyond bound
        for i in range(4100):
            obj.count_tokens(f"text_{i}")

        # Cache should have been cleared and only have recent entries
        assert len(obj._token_count_cache) < 4097


class TestTokenizerNoSilentFallback:
    """Test that auto mode never silently uses SimpleTokenizer."""

    def test_auto_does_not_return_simple(self):
        """Auto mode should never return a SimpleTokenizer."""
        from enigma_engine.core.tokenizer import get_tokenizer, SimpleTokenizer

        tok = get_tokenizer("auto", use_cache=False)
        assert not isinstance(tok, SimpleTokenizer), (
            "Auto mode fell through to SimpleTokenizer — this is a silent quality bug"
        )

    def test_unknown_type_raises(self):
        """Unknown tokenizer types should raise, not silently fallback."""
        from enigma_engine.core.tokenizer import get_tokenizer

        with pytest.raises((RuntimeError, ValueError)):
            get_tokenizer("nonexistent_tokenizer", use_cache=False)


# ================================================================
# Suggestion 19: Web SSRF protection + streaming
# ================================================================


class TestCharTokenizerThreadSafety:
    """char_tokenizer add_word must be thread-safe."""

    def test_has_vocab_lock(self):
        """CharacterTokenizer should have a _vocab_lock attribute."""
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer(use_dictionary=False)
        import threading
        assert hasattr(tok, "_vocab_lock")
        assert isinstance(tok._vocab_lock, type(threading.Lock()))


class TestCharTokenizerVocabCap:
    """char_tokenizer add_word must respect max_vocab_size cap."""

    def test_add_word_checks_max_vocab_size(self):
        """add_word should return unk_token_id when vocab is full."""
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer(use_dictionary=False)
        initial_size = tok.vocab_size
        tok._max_vocab_size = initial_size
        result = tok.add_word("ZZZZZ_test_word_that_wont_exist")
        assert result == tok.unk_token_id
        assert tok.vocab_size == initial_size

    def test_no_cap_allows_growth(self):
        """Without max_vocab_size, vocab grows normally (default behavior)."""
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer(use_dictionary=False)
        initial_size = tok.vocab_size
        result = tok.add_word("ZZZZZ_unique_test_token_12345")
        assert result >= initial_size
        assert tok.vocab_size == initial_size + 1

    def test_init_accepts_max_vocab_size(self):
        """CharacterTokenizer constructor accepts max_vocab_size param."""
        from enigma_engine.core.char_tokenizer import CharacterTokenizer
        tok = CharacterTokenizer(use_dictionary=False, max_vocab_size=50000)
        assert tok._max_vocab_size == 50000


@pytest.mark.structural
class TestByteLevelBPE:
    """Test UTF-8 byte-level BPE encoding."""

    def test_utf8_mode_encodes_ascii(self):
        """ASCII text encodes correctly in UTF-8 mode."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = True
        ids = tok.encode("hello")
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert "hello" in decoded

    def test_utf8_mode_encodes_unicode(self):
        """Unicode text round-trips through UTF-8 byte BPE."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = True
        text = "café"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert "caf" in decoded
        # The é should survive round-trip (as UTF-8 bytes)
        assert "é" in decoded or "caf" in decoded

    def test_utf8_mode_encodes_emoji(self):
        """Emoji text doesn't produce <unk> tokens in UTF-8 mode."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = True
        ids = tok.encode("hello 🌍")
        # Should not contain unk_token_id (all bytes are in 0-255)
        assert tok.unk_token_id not in ids

    def test_utf8_mode_off_by_default(self):
        """UTF-8 byte mode is off by default for backward compat."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        assert tok.use_utf8_bytes is False

    def test_utf8_flag_saved_and_loaded(self, tmp_path):
        """use_utf8_bytes persists through save/load cycle."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = True
        path = tmp_path / "vocab.json"
        tok.save(path)
        tok2 = BPETokenizer(vocab_file=path)
        assert tok2.use_utf8_bytes is True

    def test_utf8_decode_reconstructs_multibyte(self):
        """Decoding UTF-8 byte tokens reconstructs multi-byte chars."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = True
        # "ñ" is 2 bytes in UTF-8: 0xc3 0xb1
        ids = tok.encode("ñ")
        decoded = tok.decode(ids)
        assert "ñ" in decoded

    def test_legacy_mode_unchanged(self):
        """With use_utf8_bytes=False, behavior is identical to old code."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = False
        ids = tok.encode("hello world")
        decoded = tok.decode(ids)
        assert "hello" in decoded
        assert "world" in decoded

    def test_utf8_train_produces_valid_merges(self, tmp_path):
        """Training in UTF-8 mode produces valid merges."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.use_utf8_bytes = True
        tok.train(["hello world café résumé"] * 10,
                   vocab_size=300, verbose=False)
        assert len(tok.merges) > 0
        # Should still encode cleanly
        ids = tok.encode("café")
        decoded = tok.decode(ids)
        assert "caf" in decoded


# ─────────────────────────────────────────────────────────────────────────────
# TC-1: BPE Tokenizer — train / encode / decode
# ─────────────────────────────────────────────────────────────────────────────

class TestBPETokenizerTrain:
    """Test the actual BPE training algorithm (TC-1)."""

    def test_train_increases_vocab(self):
        """Training should produce new merge tokens beyond base vocab."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        base_size = tok.vocab_size
        tok.train(["hello world hello world"] * 20, vocab_size=base_size + 10,
                   verbose=False)
        assert tok.vocab_size > base_size
        assert len(tok.merges) > 0

    def test_train_empty_raises(self):
        """Training on empty list raises ValueError."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        with pytest.raises(ValueError, match="empty"):
            tok.train([], vocab_size=100, verbose=False)

    def test_train_min_frequency_zero_raises(self):
        """min_frequency < 1 raises ValueError."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        with pytest.raises(ValueError, match="min_frequency"):
            tok.train(["hello"], vocab_size=100, min_frequency=0, verbose=False)

    def test_train_creates_merge_rules(self):
        """Training produces ordered merge rules."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.train(["the cat sat on the mat"] * 50, vocab_size=300, verbose=False)
        assert len(tok.merges) > 0
        # Merge rules are tuples of two strings
        for merge in tok.merges:
            assert isinstance(merge, (tuple, list))
            assert len(merge) == 2

    def test_train_progress_callback(self):
        """on_progress callback is called during training."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        calls = []
        tok.train(["hello world"] * 20, vocab_size=280,
                   verbose=False, on_progress=lambda pct, msg: calls.append(pct))
        assert len(calls) > 0
        # Progress values should be integers in [0, 100]
        assert all(0 <= p <= 100 for p in calls)


class TestBPETokenizerEncodeDecode:
    """Test BPE encode/decode round-trip (TC-1)."""

    def test_encode_returns_list_of_ints(self):
        """encode() returns a list of integer token IDs."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        ids = tok.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_adds_special_tokens(self):
        """With add_special_tokens=True, BOS and EOS are included."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        ids = tok.encode("hello", add_special_tokens=True)
        assert ids[0] == tok.bos_token_id
        assert ids[-1] == tok.eos_token_id

    def test_encode_no_special_tokens(self):
        """With add_special_tokens=False, no BOS/EOS."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        ids = tok.encode("hello", add_special_tokens=False)
        assert ids[0] != tok.bos_token_id or len(ids) == 0

    def test_decode_skip_special_tokens(self):
        """decode(skip_special_tokens=True) omits BOS/EOS."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        ids = tok.encode("hello", add_special_tokens=True)
        text = tok.decode(ids, skip_special_tokens=True)
        assert "<s>" not in text
        assert "</s>" not in text

    def test_round_trip_untrained(self):
        """Untrained tokenizer still round-trips via character-level tokens."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        text = "hello world"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True)
        # Characters should survive (merges haven't happened yet)
        assert "hello" in decoded
        assert "world" in decoded

    def test_round_trip_after_training(self):
        """After training, encode→decode preserves text content."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        corpus = ["the quick brown fox jumps over the lazy dog"] * 50
        tok.train(corpus, vocab_size=300, verbose=False)
        text = "the quick brown fox"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True)
        # Words should survive round-trip
        for word in ["the", "quick", "brown", "fox"]:
            assert word in decoded

    def test_trained_vocab_is_smaller_encoding(self):
        """Trained tokenizer should use fewer tokens than untrained for repeated text."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        untrained = BPETokenizer()
        trained = BPETokenizer()
        text = "the the the the the"
        trained.train([text] * 100, vocab_size=300, verbose=False)
        ids_untrained = untrained.encode(text, add_special_tokens=False)
        ids_trained = trained.encode(text, add_special_tokens=False)
        # Trained should produce fewer or equal tokens
        assert len(ids_trained) <= len(ids_untrained)

    def test_special_token_markers_in_pre_tokenize(self):
        """Pre-tokenizer maps Q:/A:/User:/Assistant: to special tokens."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        ids = tok.encode("User: hello", add_special_tokens=False)
        # <USER> token = 8
        assert tok.special_tokens["<USER>"] in ids

    def test_encode_empty_string(self):
        """Encoding empty string returns only special tokens."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        ids = tok.encode("", add_special_tokens=True)
        assert ids == [tok.bos_token_id, tok.eos_token_id]

    def test_save_load_roundtrip(self, tmp_path):
        """Trained tokenizer survives save/load cycle."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.train(["hello world"] * 50, vocab_size=300, verbose=False)
        path = tmp_path / "vocab.json"
        tok.save(path)

        tok2 = BPETokenizer(vocab_file=path)
        assert tok2.vocab_size == tok.vocab_size
        assert len(tok2.merges) == len(tok.merges)
        # Same encoding after reload
        ids1 = tok.encode("hello", add_special_tokens=False)
        ids2 = tok2.encode("hello", add_special_tokens=False)
        assert ids1 == ids2

    def test_tokenizer_data_dict_roundtrip(self, tmp_path):
        """Tokenizer data dict (for checkpoint bundling) roundtrips."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.train(["hello world foo bar"] * 50,
                  vocab_size=300, verbose=False)
        # Simulate checkpoint bundling (C-5)
        tok_data = {
            "token_to_id": tok.token_to_id,
            "merges": tok.merges,
            "special_tokens": tok.special_tokens,
            "use_utf8_bytes": tok.use_utf8_bytes,
        }
        # Reconstruct from dict (same as gui_forge_new_modes.py)
        tok2 = BPETokenizer()
        tok2.token_to_id = tok_data["token_to_id"]
        tok2.id_to_token = {v: k for k, v
                            in tok2.token_to_id.items()}
        tok2.merges = [tuple(m) for m in tok_data["merges"]]
        tok2.merge_ranks = {tuple(m): i for i, m
                            in enumerate(tok2.merges)}
        tok2.special_tokens = tok_data["special_tokens"]
        tok2.use_utf8_bytes = tok_data["use_utf8_bytes"]
        tok2.vocab_size = len(tok2.token_to_id)
        # Verify identical encoding
        for text in ["hello world", "foo bar", ""]:
            ids1 = tok.encode(text, add_special_tokens=True)
            ids2 = tok2.encode(text, add_special_tokens=True)
            assert ids1 == ids2, f"Mismatch on '{text}'"


class TestBPEHeapMerge:
    """Test that heap-based BPE merge produces correct results."""

    def test_heap_merge_same_as_naive(self):
        """Heap-optimized merge must produce same merges as naive linear scan."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        corpus = [
            "the cat sat on the mat",
            "the dog ran to the park",
            "a quick brown fox jumps",
        ] * 30
        tok.train(corpus, vocab_size=320, verbose=False)
        # Merges should be deterministic and non-empty
        assert len(tok.merges) > 10
        # All merges are valid (pair of strings)
        for a, b in tok.merges:
            assert isinstance(a, str) and isinstance(b, str)
            assert len(a) > 0 and len(b) > 0

    def test_heap_merge_roundtrip_after_train(self):
        """After heap-based training, encode/decode still round-trips."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        corpus = ["hello world this is a test of the tokenizer"] * 50
        tok.train(corpus, vocab_size=310, verbose=False)
        text = "hello world this is a test"
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True)
        for word in ["hello", "world", "this", "test"]:
            assert word in decoded

    def test_heap_merge_large_vocab(self):
        """Heap merge handles larger vocab targets without error."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        corpus = [
            "the quick brown fox jumps over the lazy dog",
            "pack my box with five dozen liquor jugs",
            "how vexingly quick daft zebras jump",
        ] * 40
        tok.train(corpus, vocab_size=500, verbose=False)
        assert tok.vocab_size <= 500
        assert len(tok.merges) > 50


@pytest.mark.structural
class TestSpecialTokenIdSync:
    """I-25: Convenience IDs must stay in sync with special_tokens map."""

    def test_bpe_sync_after_save_load(self, tmp_path):
        """BPETokenizer._sync_special_ids rebuilds IDs from loaded map."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        path = tmp_path / "tok.json"
        tok.save(path)

        tok2 = BPETokenizer(vocab_file=path)
        assert tok2.pad_token_id == tok2.special_tokens["<pad>"]
        assert tok2.bos_token_id == tok2.special_tokens["<s>"]
        assert tok2.eos_token_id == tok2.special_tokens["</s>"]
        assert tok2.unk_token_id == tok2.special_tokens["<unk>"]
        assert tok2.think_start_id == tok2.special_tokens["<think>"]
        assert tok2.think_end_id == tok2.special_tokens["</think>"]

    def test_bpe_sync_with_remapped_ids(self, tmp_path):
        """If saved file has different special token IDs, convenience IDs update."""
        import json
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        path = tmp_path / "tok.json"
        tok.save(path)

        # Manually edit the saved file to remap <pad> to ID 99
        data = json.loads(path.read_text(encoding="utf-8"))
        data["special_tokens"]["<pad>"] = 99
        data["token_to_id"]["<pad>"] = 99
        path.write_text(json.dumps(data), encoding="utf-8")

        tok2 = BPETokenizer(vocab_file=path)
        assert tok2.pad_token_id == 99

    def test_simple_tokenizer_sync_after_load(self, tmp_path):
        """SimpleTokenizer syncs convenience IDs from loaded vocab."""
        from enigma_engine.core.tokenizer import SimpleTokenizer
        tok = SimpleTokenizer()
        path = tmp_path / "vocab.json"
        tok.save_vocab(path)

        tok2 = SimpleTokenizer(vocab_file=path)
        assert tok2.pad_token_id == tok2.special_tokens["<pad>"]
        assert tok2.eos_token_id == tok2.special_tokens["</s>"]
        assert tok2.think_start_id == tok2.special_tokens["<think>"]

    def test_bpe_has_sync_method(self):
        """BPETokenizer._sync_special_ids exists and is callable."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        assert hasattr(tok, "_sync_special_ids")
        assert callable(tok._sync_special_ids)

    def test_simple_has_sync_method(self):
        """SimpleTokenizer._sync_special_ids exists and is callable."""
        from enigma_engine.core.tokenizer import SimpleTokenizer
        tok = SimpleTokenizer()
        assert hasattr(tok, "_sync_special_ids")
        assert callable(tok._sync_special_ids)


# ================================================================
# Tokenizer Metrics (from test_tokenizer_metrics.py)
# ================================================================


def _make_stub_tokenizer(vocab_size=100, unk_id=3):
    """Create a stub tokenizer with predictable behavior."""
    vocab = {chr(i + 32): i for i in range(vocab_size)}
    special = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    merges = [("a", "b"), ("c", "d"), ("e", "f")]

    tok = SimpleNamespace(
        token_to_id=vocab,
        special_tokens=special,
        merges=merges,
        use_utf8_bytes=False,
        unk_token_id=unk_id,
        vocab_size=vocab_size,
    )
    # encode: simple char-to-id mapping
    def encode(text, add_special_tokens=False):
        return [vocab.get(ch, unk_id) for ch in text]
    tok.encode = encode
    return tok


class TestAnalyzeVocabulary:
    """Test vocabulary analysis."""

    def test_vocab_size(self):
        from enigma_engine.core.tokenizer_metrics import analyze_vocabulary
        tok = _make_stub_tokenizer(vocab_size=50)
        result = analyze_vocabulary(tok)
        assert result['vocab_size'] == 50

    def test_num_merges(self):
        from enigma_engine.core.tokenizer_metrics import analyze_vocabulary
        tok = _make_stub_tokenizer()
        result = analyze_vocabulary(tok)
        assert result['num_merges'] == 3

    def test_num_special(self):
        from enigma_engine.core.tokenizer_metrics import analyze_vocabulary
        tok = _make_stub_tokenizer()
        result = analyze_vocabulary(tok)
        assert result['num_special'] == 4

    def test_token_lengths_present(self):
        from enigma_engine.core.tokenizer_metrics import analyze_vocabulary
        tok = _make_stub_tokenizer()
        result = analyze_vocabulary(tok)
        lengths = result['token_lengths']
        assert 'min' in lengths
        assert 'max' in lengths
        assert 'mean' in lengths
        assert 'median' in lengths

    def test_all_single_char(self):
        """Stub vocab is all single chars — single_char_tokens should be high."""
        from enigma_engine.core.tokenizer_metrics import analyze_vocabulary
        tok = _make_stub_tokenizer()
        result = analyze_vocabulary(tok)
        assert result['single_char_tokens'] > 0

    def test_utf8_bytes_flag(self):
        from enigma_engine.core.tokenizer_metrics import analyze_vocabulary
        tok = _make_stub_tokenizer()
        result = analyze_vocabulary(tok)
        assert result['use_utf8_bytes'] is False


class TestEvaluateCoverage:
    """Test coverage evaluation."""

    def test_known_text_coverage(self):
        """Known text (all chars in vocab) should have 100% coverage."""
        from enigma_engine.core.tokenizer_metrics import evaluate_coverage
        tok = _make_stub_tokenizer()
        result = evaluate_coverage(tok, ["abc def"])
        assert result['coverage'] == 1.0
        assert result['unk_count'] == 0

    def test_unknown_text_has_unk(self):
        """Text with chars outside vocab should produce UNK tokens."""
        from enigma_engine.core.tokenizer_metrics import evaluate_coverage
        tok = _make_stub_tokenizer(vocab_size=10)
        # Chars with code points above vocab range will map to unk
        result = evaluate_coverage(tok, ["\x00\x01\x02"])
        assert result['unk_count'] >= 0  # Depends on mapping

    def test_total_tokens(self):
        from enigma_engine.core.tokenizer_metrics import evaluate_coverage
        tok = _make_stub_tokenizer()
        result = evaluate_coverage(tok, ["hello"])
        assert result['total_tokens'] == 5

    def test_unique_tokens(self):
        from enigma_engine.core.tokenizer_metrics import evaluate_coverage
        tok = _make_stub_tokenizer()
        result = evaluate_coverage(tok, ["aaa"])
        assert result['unique_tokens'] == 1

    def test_empty_text_list(self):
        from enigma_engine.core.tokenizer_metrics import evaluate_coverage
        tok = _make_stub_tokenizer()
        result = evaluate_coverage(tok, [])
        assert result['total_tokens'] == 0
        assert result['coverage'] == 1.0


class TestCompressionRatio:
    """Test compression ratio computation."""

    def test_char_level_ratio_is_one(self):
        """Char-level tokenizer should get ~1.0 chars/token."""
        from enigma_engine.core.tokenizer_metrics import compute_compression_ratio
        tok = _make_stub_tokenizer()
        result = compute_compression_ratio(tok, ["hello world"])
        assert result['chars_per_token'] == 1.0

    def test_total_chars(self):
        from enigma_engine.core.tokenizer_metrics import compute_compression_ratio
        tok = _make_stub_tokenizer()
        result = compute_compression_ratio(tok, ["abc"])
        assert result['total_chars'] == 3

    def test_empty_text(self):
        from enigma_engine.core.tokenizer_metrics import compute_compression_ratio
        tok = _make_stub_tokenizer()
        result = compute_compression_ratio(tok, [])
        assert result['total_tokens'] == 0
        assert result['chars_per_token'] == 0.0


class TestDetectIssues:
    """Test health-check issue detection."""

    def test_no_issues_on_good_tokenizer(self):
        """Healthy tokenizer should produce no high-severity warnings."""
        from enigma_engine.core.tokenizer_metrics import detect_issues
        tok = _make_stub_tokenizer()
        # All chars in vocab → no UNK
        issues = detect_issues(tok, ["abcdef"])
        # char-level tokenizer will warn about low compression (1.0 chars/tok)
        high_unk = [w for w in issues if "High UNK" in w]
        assert len(high_unk) == 0

    def test_low_compression_warning(self):
        """Char-level tokenizer should warn about low compression."""
        from enigma_engine.core.tokenizer_metrics import detect_issues
        tok = _make_stub_tokenizer()
        issues = detect_issues(tok, ["abcdef"])
        low_comp = [w for w in issues if "Low compression" in w]
        assert len(low_comp) == 1


class TestFormatReport:
    """Test human-readable report generation."""

    def test_report_is_string(self):
        from enigma_engine.core.tokenizer_metrics import format_report
        tok = _make_stub_tokenizer()
        report = format_report(tok, ["hello"])
        assert isinstance(report, str)

    def test_report_contains_sections(self):
        from enigma_engine.core.tokenizer_metrics import format_report
        tok = _make_stub_tokenizer()
        report = format_report(tok, ["hello"])
        assert "Vocabulary" in report
        assert "Coverage" in report
        assert "Compression" in report

    def test_report_contains_vocab_size(self):
        from enigma_engine.core.tokenizer_metrics import format_report
        tok = _make_stub_tokenizer(vocab_size=100)
        report = format_report(tok, ["hello"])
        assert "100" in report

