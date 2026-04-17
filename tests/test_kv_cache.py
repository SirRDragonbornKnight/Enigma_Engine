"""Tests for KV cache and JSON schema constraint (TC-4, TC-5)."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# TC-4: KVCache — allocation, update, get, clear, clone
# ─────────────────────────────────────────────────────────────────────────────

class TestKVCacheBasic:
    """Core KVCache operations on CPU (TC-4)."""

    def test_allocate_and_memory(self):
        """Cache allocates with expected shape and reports memory."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"))
        assert cache.current_pos == 0
        assert cache.memory_usage_mb() > 0

    def test_update_advances_position(self):
        """update() advances current_pos by seq_len."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"))
        k = torch.randn(1, 5, 4, 32)
        v = torch.randn(1, 5, 4, 32)
        new_pos = cache.update(k, v)
        assert new_pos == 5
        assert cache.current_pos == 5

    def test_get_returns_cached_values(self):
        """get() returns previously cached K, V tensors."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"))
        k = torch.randn(1, 5, 4, 32)
        v = torch.randn(1, 5, 4, 32)
        cache.update(k, v)
        k_out, v_out = cache.get()
        assert k_out.shape == (1, 5, 4, 32)
        assert v_out.shape == (1, 5, 4, 32)
        assert torch.allclose(k_out, k, atol=1e-6)
        assert torch.allclose(v_out, v, atol=1e-6)

    def test_sequential_updates(self):
        """Multiple update() calls append correctly."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"))
        k1 = torch.randn(1, 3, 4, 32)
        v1 = torch.randn(1, 3, 4, 32)
        k2 = torch.randn(1, 2, 4, 32)
        v2 = torch.randn(1, 2, 4, 32)
        cache.update(k1, v1)
        cache.update(k2, v2)
        assert cache.current_pos == 5
        k_out, v_out = cache.get()
        assert k_out.shape == (1, 5, 4, 32)

    def test_clear_resets_position(self):
        """clear() resets position to 0."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"))
        k = torch.randn(1, 5, 4, 32)
        v = torch.randn(1, 5, 4, 32)
        cache.update(k, v)
        cache.clear()
        assert cache.current_pos == 0

    def test_get_up_to_position(self):
        """get(up_to_position=N) returns only first N positions."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"))
        k = torch.randn(1, 10, 4, 32)
        v = torch.randn(1, 10, 4, 32)
        cache.update(k, v)
        k_out, v_out = cache.get(up_to_position=3)
        assert k_out.shape == (1, 3, 4, 32)

    def test_batch_mismatch_raises(self):
        """Mismatched batch size raises ValueError."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"))
        k = torch.randn(2, 5, 4, 32)  # batch=2 vs cache batch=1
        v = torch.randn(2, 5, 4, 32)
        with pytest.raises(ValueError, match="batch"):
            cache.update(k, v)


class TestKVCacheClone:
    """KVCache.clone() creates independent copy (TC-4)."""

    def test_clone_has_same_position(self):
        """Cloned cache has same current_pos."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"))
        k = torch.randn(1, 5, 4, 32)
        v = torch.randn(1, 5, 4, 32)
        cache.update(k, v)
        clone = cache.clone()
        assert clone.current_pos == cache.current_pos

    def test_clone_is_independent(self):
        """Modifying clone doesn't affect original."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"))
        k = torch.randn(1, 5, 4, 32)
        v = torch.randn(1, 5, 4, 32)
        cache.update(k, v)
        clone = cache.clone()
        clone.clear()
        assert cache.current_pos == 5
        assert clone.current_pos == 0

    def test_clone_data_matches(self):
        """Cloned cache has identical K,V data."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"))
        k = torch.randn(1, 5, 4, 32)
        v = torch.randn(1, 5, 4, 32)
        cache.update(k, v)
        clone = cache.clone()
        k_orig, v_orig = cache.get()
        k_copy, v_copy = clone.get()
        assert torch.allclose(k_orig, k_copy)
        assert torch.allclose(v_orig, v_copy)


class TestKVCacheQuantized:
    """INT8 quantized KV cache (TC-4)."""

    def test_quantized_allocates(self):
        """INT8 cache allocates without error."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"),
            quantize_to_int8=True)
        assert cache.quantize is True
        assert cache._cache_k.dtype == torch.int8

    def test_quantized_roundtrip_approximate(self):
        """INT8 quantization is lossy but approximately correct."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"),
            quantize_to_int8=True)
        k = torch.randn(1, 5, 4, 32)
        v = torch.randn(1, 5, 4, 32)
        cache.update(k, v)
        k_out, v_out = cache.get()
        assert k_out.shape == (1, 5, 4, 32)
        # INT8 quantization error should be small (within ~5% of range)
        assert torch.allclose(k_out, k, atol=0.1)

    def test_quantized_uses_less_memory(self):
        """INT8 cache uses less memory than float32."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        fp32 = KVCache(
            batch_size=1, max_seq_len=256, n_kv_heads=8,
            head_dim=64, device=torch.device("cpu"))
        int8 = KVCache(
            batch_size=1, max_seq_len=256, n_kv_heads=8,
            head_dim=64, device=torch.device("cpu"),
            quantize_to_int8=True)
        # INT8 cache bytes for KV should be ~4x smaller than FP32,
        # but scale/zp tensors add overhead → still significantly less
        assert int8.memory_usage_mb() < fp32.memory_usage_mb()


class TestKVCacheRestorePrefix:
    """KVCache.restore_prefix for speculative decoding (TC-4)."""

    def test_restore_prefix_sets_position(self):
        """restore_prefix sets current_pos to prefix length."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"))
        k = torch.randn(1, 8, 4, 32)
        v = torch.randn(1, 8, 4, 32)
        cache.restore_prefix(k, v)
        assert cache.current_pos == 8

    def test_restore_prefix_data_retrievable(self):
        """After restore_prefix, get() returns the prefix data."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=64, n_kv_heads=4,
            head_dim=32, device=torch.device("cpu"))
        k = torch.randn(1, 8, 4, 32)
        v = torch.randn(1, 8, 4, 32)
        cache.restore_prefix(k, v)
        k_out, v_out = cache.get()
        assert torch.allclose(k_out, k, atol=1e-6)


class TestKVCacheSlidingWindow:
    """KV cache handles overflow with sliding window shift (TC-4)."""

    def test_overflow_shifts_cache(self):
        """Writing past max_seq_len triggers window shift."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=10, n_kv_heads=2,
            head_dim=8, device=torch.device("cpu"))
        # Fill to capacity
        k1 = torch.randn(1, 10, 2, 8)
        v1 = torch.randn(1, 10, 2, 8)
        cache.update(k1, v1)
        assert cache.current_pos == 10
        # Overflow with 3 more tokens
        k2 = torch.randn(1, 3, 2, 8)
        v2 = torch.randn(1, 3, 2, 8)
        new_pos = cache.update(k2, v2)
        assert new_pos == 10  # Window shifted, position stays at max


# ─────────────────────────────────────────────────────────────────────────────
# TC-5: JsonSchemaConstraint — FSM-based JSON decoding
# ─────────────────────────────────────────────────────────────────────────────

class _MockTokenizer:
    """Minimal tokenizer mock for JsonSchemaConstraint tests."""
    vocab_size = 128

    def decode(self, ids):
        """Map token IDs to ASCII chars (simple 1:1 mapping)."""
        return "".join(chr(i) if 0 <= i < 128 else "?" for i in ids)


class TestJsonSchemaConstraintInit:
    """JsonSchemaConstraint initialisation (TC-5)."""

    def test_creates_from_schema(self):
        """Constraint can be created from a simple schema."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            }
        }
        c = JsonSchemaConstraint(schema, _MockTokenizer())
        assert not c.is_done

    def test_reset_returns_to_initial(self):
        """reset() puts FSM back to EXPECT_OPEN state."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        c = JsonSchemaConstraint(schema, _MockTokenizer())
        # Advance past the opening brace
        c.advance(ord("{"))
        c.reset()
        assert c._state == "EXPECT_OPEN"


class TestJsonSchemaConstraintFSM:
    """JsonSchemaConstraint FSM state transitions (TC-5)."""

    def test_advance_through_simple_object(self):
        """FSM transitions through a complete {"name": "val"} object."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            }
        }
        c = JsonSchemaConstraint(schema, _MockTokenizer())

        # Feed: {"name": "val"}
        for ch in '{"name": "val"}':
            c.advance(ord(ch))
        assert c.is_done

    def test_advance_multi_field(self):
        """FSM handles multiple key-value pairs."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
            }
        }
        c = JsonSchemaConstraint(schema, _MockTokenizer())
        for ch in '{"a": "x", "b": "y"}':
            c.advance(ord(ch))
        assert c.is_done

    def test_empty_object_schema(self):
        """Schema with no properties: {} should complete immediately."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        schema = {"type": "object", "properties": {}}
        c = JsonSchemaConstraint(schema, _MockTokenizer())
        for ch in "{}":
            c.advance(ord(ch))
        assert c.is_done


# ─────────────────────────────────────────────────────────────────────────────
# S736-S738: Nested values, array depth, key-ordered type lookup
# ─────────────────────────────────────────────────────────────────────────────

class TestJsonSchemaConstraintNested:
    """S736-S738: FSM handles nested objects, arrays, and key-based types."""

    def _make(self, schema):
        torch = pytest.importorskip("torch")
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        return JsonSchemaConstraint(schema, _MockTokenizer())

    def test_nested_object_value(self):
        """S736: {"key": {"inner": "val"}} parses to DONE."""
        schema = {"type": "object", "properties": {"key": {"type": "object"}}}
        c = self._make(schema)
        for ch in '{"key": {"inner": "val"}}':
            c.advance(ord(ch))
        assert c.is_done

    def test_array_value(self):
        """S737: {"key": [1, 2, 3]} parses to DONE."""
        schema = {"type": "object", "properties": {"key": {"type": "array"}}}
        c = self._make(schema)
        for ch in '{"key": [1, 2, 3]}':
            c.advance(ord(ch))
        assert c.is_done

    def test_nested_array_of_objects(self):
        """S737: {"key": [{"a": 1}]} parses to DONE."""
        schema = {"type": "object", "properties": {"key": {"type": "array"}}}
        c = self._make(schema)
        for ch in '{"key": [{"a": 1}]}':
            c.advance(ord(ch))
        assert c.is_done

    def test_mixed_nesting(self):
        """S736+S737: {"key": {"nested": [1, 2]}} parses to DONE."""
        schema = {"type": "object", "properties": {"key": {"type": "object"}}}
        c = self._make(schema)
        for ch in '{"key": {"nested": [1, 2]}}':
            c.advance(ord(ch))
        assert c.is_done

    def test_key_type_lookup_by_name(self):
        """S738: type constraint follows actual key name, not index."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            }
        }
        c = self._make(schema)
        # Feed keys in reverse order: age first, then name
        for ch in '{"age": 42, "name": "Jo"}':
            c.advance(ord(ch))
        assert c.is_done

    def test_flat_objects_still_work(self):
        """Regression: flat objects unchanged after nesting fixes."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "boolean"},
            }
        }
        c = self._make(schema)
        for ch in '{"a": "hi", "b": true}':
            c.advance(ord(ch))
        assert c.is_done


# ─────────────────────────────────────────────────────────────────────────────
# S739: H2O eviction must compact zero-point tensors
# ─────────────────────────────────────────────────────────────────────────────

class TestH2OEvictionQuantized:
    """S739: H2O evict_if_needed must reindex _zp_k/_zp_v alongside scales."""

    def test_eviction_preserves_quantized_values(self):
        """After eviction, dequantized KV values match the kept originals."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import H2OKVCache

        cache = H2OKVCache(
            heavy_hitter_count=2, recent_window=2,
            batch_size=1, max_seq_len=32, n_kv_heads=2,
            head_dim=8, device=torch.device("cpu"),
            dtype=torch.float32, quantize_to_int8=True)

        # Store 8 tokens with distinguishable values
        originals_k = []
        originals_v = []
        for i in range(8):
            k = torch.randn(1, 1, 2, 8) * (i + 1)
            v = torch.randn(1, 1, 2, 8) * (i + 1)
            originals_k.append(k.clone())
            originals_v.append(v.clone())
            cache.update(k, v)

        assert cache.current_pos == 8

        # Make tokens 0 and 1 heavy hitters (high attention)
        attn = torch.zeros(1, 8)
        attn[0, 0] = 10.0
        attn[0, 1] = 8.0
        cache.accumulate_attention(attn)

        # Snapshot pre-eviction values for kept positions
        # Kept: HH=[0,1], recent=[6,7] → indices [0,1,6,7]
        k_before, v_before = cache.get()
        kept_indices = [0, 1, 6, 7]
        k_kept_before = k_before[:, kept_indices, :, :]
        v_kept_before = v_before[:, kept_indices, :, :]

        cache.evict_if_needed()
        assert cache.current_pos == 4

        # Dequantized values after eviction should match pre-eviction
        k_after, v_after = cache.get()
        assert k_after.shape == (1, 4, 2, 8)
        assert torch.allclose(k_after, k_kept_before, atol=1e-5), \
            "K values corrupted after H2O eviction (zero-point mismatch)"
        assert torch.allclose(v_after, v_kept_before, atol=1e-5), \
            "V values corrupted after H2O eviction (zero-point mismatch)"

    def test_not_done_mid_object(self):
        """FSM is not done while mid-object."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        c = JsonSchemaConstraint(schema, _MockTokenizer())
        c.advance(ord("{"))
        assert not c.is_done


# ─────────────────────────────────────────────────────────────────────────────
# S741: KVCache.rewind_to — partial cache truncation
# ─────────────────────────────────────────────────────────────────────────────

class TestKVCacheRewind:
    """rewind_to() truncates cache to a given position."""

    def test_rewind_basic(self):
        """rewind_to(pos) sets current_pos and zeros invalidated slots."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=32, n_kv_heads=2,
            head_dim=8, device=torch.device("cpu"))
        k = torch.randn(1, 10, 2, 8)
        v = torch.randn(1, 10, 2, 8)
        cache.update(k, v)
        assert cache.current_pos == 10

        # Snapshot values at positions 0..4
        k_before, v_before = cache.get()
        k_kept = k_before[:, :5].clone()
        v_kept = v_before[:, :5].clone()

        cache.rewind_to(5)
        assert cache.current_pos == 5

        k_after, v_after = cache.get()
        assert k_after.shape[1] == 5
        assert torch.allclose(k_after, k_kept)
        assert torch.allclose(v_after, v_kept)

    def test_rewind_noop_when_at_or_past(self):
        """rewind_to(pos >= current_pos) is a no-op."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=32, n_kv_heads=2,
            head_dim=8, device=torch.device("cpu"))
        cache.update(torch.randn(1, 5, 2, 8), torch.randn(1, 5, 2, 8))
        cache.rewind_to(5)
        assert cache.current_pos == 5
        cache.rewind_to(10)
        assert cache.current_pos == 5

    def test_rewind_clamps_negative(self):
        """rewind_to(negative) clamps to 0."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=32, n_kv_heads=2,
            head_dim=8, device=torch.device("cpu"))
        cache.update(torch.randn(1, 5, 2, 8), torch.randn(1, 5, 2, 8))
        cache.rewind_to(-1)
        assert cache.current_pos == 0

    def test_rewind_quantized(self):
        """rewind_to resets scales and zero-points for invalidated slots."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=32, n_kv_heads=2,
            head_dim=8, device=torch.device("cpu"),
            dtype=torch.float32, quantize_to_int8=True)
        k = torch.randn(1, 8, 2, 8)
        v = torch.randn(1, 8, 2, 8)
        cache.update(k, v)

        # Snapshot kept values
        k_before, v_before = cache.get()
        k_kept = k_before[:, :4].clone()
        v_kept = v_before[:, :4].clone()

        cache.rewind_to(4)
        assert cache.current_pos == 4

        k_after, v_after = cache.get()
        assert torch.allclose(k_after, k_kept, atol=1e-5)
        assert torch.allclose(v_after, v_kept, atol=1e-5)


class TestH2OCacheRewind:
    """H2OKVCache.rewind_to also zeros attention scores."""

    def test_rewind_zeros_attn_scores(self):
        """Attention scores for invalidated positions are zeroed."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import H2OKVCache
        cache = H2OKVCache(
            heavy_hitter_count=4, recent_window=4,
            batch_size=1, max_seq_len=32, n_kv_heads=2,
            head_dim=8, device=torch.device("cpu"))
        cache.update(torch.randn(1, 8, 2, 8), torch.randn(1, 8, 2, 8))
        # accumulate_attention needs 3D+ input (batch, q_len, kv_len)
        cache.accumulate_attention(torch.ones(1, 1, 8))

        cache.rewind_to(5)
        assert cache.current_pos == 5
        # Positions 5..7 should have zero attention
        assert cache._attn_scores[:, 5:8].abs().max().item() == 0.0
        # Positions 0..4 should still have scores
        assert cache._attn_scores[:, :5].sum().item() > 0


class TestStreamingLLMCacheRewind:
    """StreamingLLMCache.rewind_to adjusts logical position."""

    def test_rewind_adjusts_logical_pos(self):
        """Logical position is decremented by the same delta."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import StreamingLLMCache
        cache = StreamingLLMCache(
            n_sink=2, window_size=16,
            batch_size=1, max_seq_len=32, n_kv_heads=2,
            head_dim=8, device=torch.device("cpu"))
        cache.update(torch.randn(1, 10, 2, 8), torch.randn(1, 10, 2, 8))
        logical_before = cache._logical_pos
        physical_before = cache.current_pos

        cache.rewind_to(7)
        assert cache.current_pos == 7
        assert cache._logical_pos == logical_before - (physical_before - 7)


class TestKVCacheManagerRewind:
    """KVCacheManager.rewind_to rewinds all layers."""

    def test_rewind_all_layers(self):
        """All layer caches are rewound to the same position."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.kv_cache import KVCacheManager
        mgr = KVCacheManager(
            n_layers=4, n_kv_heads=2, head_dim=8,
            max_seq_len=32, device=torch.device("cpu"))
        mgr.allocate(batch_size=1)
        for i in range(4):
            mgr.update(i, torch.randn(1, 10, 2, 8), torch.randn(1, 10, 2, 8))
        mgr.rewind_to(6)
        for i in range(4):
            k, v = mgr.get(i)
            assert k.shape[1] == 6


class TestModelRewindCacheStructural:
    """Structural: rewind_cache exists on model hierarchy."""

    def test_model_has_rewind_cache(self):
        """Enigma model exposes rewind_cache method."""
        import inspect
        from enigma_engine.core.model import Enigma
        assert hasattr(Enigma, 'rewind_cache')
        sig = inspect.signature(Enigma.rewind_cache)
        assert 'position' in sig.parameters

    def test_transformer_block_has_rewind_cache(self):
        """TransformerBlock exposes rewind_cache method."""
        from enigma_engine.core.model_components import TransformerBlock
        assert hasattr(TransformerBlock, 'rewind_cache')

    def test_attention_has_rewind_cache(self):
        """Attention exposes rewind_cache method."""
        from enigma_engine.core.model_components import Attention
        assert hasattr(Attention, 'rewind_cache')


class TestGenerationRewindWiring:
    """Structural: generation methods use rewind_cache."""

    def test_speculative_uses_rewind(self):
        """speculative_generate references rewind_cache."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin.speculative_generate)
        assert 'rewind_cache' in src

    def test_medusa_uses_rewind(self):
        """medusa_generate references rewind_cache."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin.medusa_generate)
        assert 'rewind_cache' in src

    def test_lookahead_uses_rewind(self):
        """lookahead_generate references rewind_cache."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin.lookahead_generate)
        assert 'rewind_cache' in src
    """JsonSchemaConstraint.mask_logits masks invalid tokens (TC-5)."""

    def test_mask_logits_shape_preserved(self):
        """mask_logits returns same shape as input."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        c = JsonSchemaConstraint(schema, _MockTokenizer())
        logits = torch.zeros(1, 128)
        masked = c.mask_logits(logits)
        assert masked.shape == logits.shape

    def test_mask_logits_blocks_invalid(self):
        """At EXPECT_OPEN, only '{' and whitespace tokens are allowed."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        c = JsonSchemaConstraint(schema, _MockTokenizer())
        logits = torch.zeros(1, 128)
        masked = c.mask_logits(logits)
        # Token for '{' = ord('{') = 123
        assert masked[0, 123].item() == 0.0  # allowed
        # Token for 'a' = 97 should be blocked
        assert masked[0, 97].item() == float("-inf")

    def test_mask_logits_after_done_blocks_all(self):
        """After FSM is done, all tokens are blocked."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        schema = {"type": "object", "properties": {}}
        c = JsonSchemaConstraint(schema, _MockTokenizer())
        for ch in "{}":
            c.advance(ord(ch))
        assert c.is_done
        logits = torch.zeros(1, 128)
        masked = c.mask_logits(logits)
        # All should be -inf
        assert (masked == float("-inf")).all()
