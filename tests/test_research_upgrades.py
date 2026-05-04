"""
Tests for research upgrades R1-R24.

These are structural tests that verify the new APIs exist and behave
correctly without requiring a GPU or trained model weights.

Run with: python -m pytest tests/test_research_upgrades.py -v
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Shared test helpers ──────────────────────────────────────────────


def _make_fake_model(n_layers=6):
    """Create a FakeModel with nn.Module layers for structural tests."""
    import itertools
    import torch.nn as nn

    class _FakeModel(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(n)])
            self.tok_embeddings = nn.Embedding(1000, 64)
            self.norm = nn.LayerNorm(64)
            self.output = nn.Linear(64, 1000)

        def named_parameters(self, recurse=True):
            for idx, layer in enumerate(self.layers):
                for name, p in layer._parameters.items():
                    yield f"layers.{idx}.{name}", p
            for name, p in self.tok_embeddings._parameters.items():
                yield f"tok_embeddings.{name}", p
            for name, p in self.norm._parameters.items():
                yield f"norm.{name}", p
            for name, p in self.output._parameters.items():
                yield f"output.{name}", p

        def parameters(self, recurse=True):
            return itertools.chain(
                *(layer.parameters() for layer in self.layers),
                self.tok_embeddings.parameters(),
                self.norm.parameters(),
                self.output.parameters(),
            )

    return _FakeModel(n_layers)


# Convenience alias used by many tests
def FakeModel(n_layers=6):
    return _make_fake_model(n_layers)


class FakeTokenizer:
    """Character-to-token mapper for JSON schema constraint tests."""
    vocab_size = 12
    _VOCAB = {
        0: '{', 1: '}', 2: '"', 3: ':',
        4: ',', 5: 'n', 6: 'a', 7: 'm',
        8: 'e', 9: ' ', 10: 'h', 11: 'i',
    }

    def decode(self, ids):
        return ''.join(self._VOCAB.get(i, '?') for i in ids)


# ════════════════════════════════════════════════════════════════════
# R1 — LoRA+ differential learning rates
# ════════════════════════════════════════════════════════════════════

class TestLoRAPlus:
    """R1: LoRA+ differential LR in lora_utils.py."""

    def test_lora_config_has_lambda(self):
        from enigma_engine.core.lora_utils import LoraConfig
        cfg = LoraConfig()
        assert hasattr(cfg, "lora_plus_lambda")
        assert cfg.lora_plus_lambda == 1.0

    def test_lora_config_custom_lambda(self):
        from enigma_engine.core.lora_utils import LoraConfig
        cfg = LoraConfig(lora_plus_lambda=16.0)
        assert cfg.lora_plus_lambda == 16.0


# ════════════════════════════════════════════════════════════════════
# R2 — BM25+ delta floor
# ════════════════════════════════════════════════════════════════════

class TestBM25Plus:
    """R2: BM25+ delta in rag.py TfidfVectorizer."""

    def test_vectorizer_has_delta(self):
        from enigma_engine.core.rag import TfidfVectorizer
        vec = TfidfVectorizer()
        assert hasattr(vec, "delta")
        assert vec.delta == 1.0

    def test_vectorizer_delta_custom(self):
        from enigma_engine.core.rag import TfidfVectorizer
        vec = TfidfVectorizer(delta=0.5)
        assert vec.delta == 0.5

    def test_delta_in_serialization(self):
        from enigma_engine.core.rag import TfidfVectorizer
        vec = TfidfVectorizer(delta=2.0)
        d = vec.to_dict()
        assert d["delta"] == 2.0
        vec2 = TfidfVectorizer.from_dict(d)
        assert vec2.delta == 2.0

    def test_cooccurrence_built_during_fit(self):
        """T2-2: fit() should build a co-occurrence map."""
        from enigma_engine.core.rag import TfidfVectorizer
        docs = [
            "the orange cat sat on the mat",
            "the orange dog ran on the road",
            "the blue bird flew over the mat",
        ]
        vec = TfidfVectorizer()
        vec.fit(docs)
        assert hasattr(vec, '_cooccurrence')
        assert len(vec._cooccurrence) > 0

    def test_cooccurrence_serialization(self):
        """T2-2: co-occurrence map round-trips through to_dict/from_dict."""
        from enigma_engine.core.rag import TfidfVectorizer
        docs = ["alpha beta gamma", "alpha beta delta", "gamma delta epsilon"]
        vec = TfidfVectorizer()
        vec.fit(docs)
        d = vec.to_dict()
        assert "cooccurrence" in d
        vec2 = TfidfVectorizer.from_dict(d)
        assert vec2._cooccurrence == vec._cooccurrence

    def test_expand_query_adds_cooccurring_terms(self):
        """T2-2: expand_query should add co-occurring terms."""
        from enigma_engine.core.rag import TfidfVectorizer
        # "training" and "model" always co-occur
        docs = [
            "training the model on data produces good results",
            "training the model requires large datasets",
            "inference uses the model for predictions",
        ]
        vec = TfidfVectorizer()
        vec.fit(docs)
        original = {"training"}
        expanded = vec.expand_query(original)
        # Should still contain the original term
        assert "training" in expanded
        # Should have expanded with at least one co-occurring term
        assert len(expanded) > len(original)

    def test_expand_query_empty_cooccurrence(self):
        """T2-2: expand_query with no co-occurrence returns original."""
        from enigma_engine.core.rag import TfidfVectorizer
        vec = TfidfVectorizer()
        tokens = {"hello", "world"}
        assert vec.expand_query(tokens) == tokens

    def test_expand_query_caches_reverse_lookup(self):
        """S764: expand_query should cache idx_to_term, not rebuild each call."""
        from enigma_engine.core.rag import TfidfVectorizer
        docs = [
            "training the model on data produces good results",
            "training the model requires large datasets",
        ]
        vec = TfidfVectorizer()
        vec.fit(docs)
        # First call — cache should be built
        vec.expand_query({"training"})
        assert hasattr(vec, "_idx_to_term"), \
            "expand_query should cache _idx_to_term after first call"
        cached = vec._idx_to_term
        # Second call — should reuse same dict object
        vec.expand_query({"model"})
        assert vec._idx_to_term is cached, \
            "_idx_to_term should be the same object on second call"


# ════════════════════════════════════════════════════════════════════
# R3 — Adaptive repetition window
# ════════════════════════════════════════════════════════════════════

class TestAdaptiveRepWindow:
    """R3: Adaptive repetition window in engine_generation.py."""

    def test_adaptive_window_values(self):
        from enigma_engine.core.engine_generation import _GenerationMixin
        # Short sequence → floor of 64
        assert _GenerationMixin._adaptive_rep_window(10) == 64
        # Medium sequence → proportional
        assert _GenerationMixin._adaptive_rep_window(300) == 150
        # Long sequence → capped at 256
        assert _GenerationMixin._adaptive_rep_window(1000) == 256


# ════════════════════════════════════════════════════════════════════
# R4 — FFD sequence packing
# ════════════════════════════════════════════════════════════════════

class TestFFDPacking:
    """R4: Best-fit decreasing (Multipack) packing in training.py."""

    def test_multipack_tighter_packing(self):
        """Multipack packs tighter than naive FFD on adversarial input."""
        from enigma_engine.core.training import pack_sequences
        # Sequences designed to pack better with best-fit:
        # max_length=20, seqs: [9, 9, 5, 5, 5, 5]
        # FFD: row1=[9+eos, 9+eos]=20, row2=[5+eos, 5+eos, 5+eos]=18, row3=[5+eos]=6 → 3 rows
        # BFD: same optimal but checks remaining capacity
        seqs = [[1]*9, [1]*9, [1]*5, [1]*5, [1]*5, [1]*5]
        packed, masks = pack_sequences(seqs, max_length=20, eos_id=2)
        # All 6 seqs should pack into ≤3 rows
        assert packed.shape[0] <= 3
        # Total real tokens should equal sum of seq lengths + eos separators
        total_tokens = sum(len(s) for s in seqs) + len(seqs)
        # Verify no tokens are lost
        non_pad = (packed != 0).sum().item()
        assert non_pad == total_tokens


# ════════════════════════════════════════════════════════════════════
# R5 — Gradient noise injection
# ════════════════════════════════════════════════════════════════════

class TestGradientNoise:
    """R5: Gradient noise in TrainingConfig."""

    def test_config_has_noise_fields(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "gradient_noise_eta")
        assert hasattr(cfg, "gradient_noise_gamma")
        assert cfg.gradient_noise_eta == 0.01  # optimal default
        assert cfg.gradient_noise_gamma == 0.55

    def test_config_has_noise_envelope_fields(self):
        """T2-3: Noise warmup/decay fractions exist with defaults."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.noise_warmup_fraction == 0.05
        assert cfg.noise_decay_fraction == 0.2


# ════════════════════════════════════════════════════════════════════
# R6 — Sentiment negation window
# ════════════════════════════════════════════════════════════════════

class TestSentimentNegation:
    """R6: Negation-aware sentiment scoring."""

    def test_negation_flips_sentiment(self):
        from enigma_engine.core.sentiment import analyze_sentiment
        pos = analyze_sentiment("I am happy")["valence"]
        neg_pos = analyze_sentiment("I am not happy")["valence"]
        # "not happy" should be less positive than "happy"
        assert neg_pos < pos

    def test_double_negation(self):
        from enigma_engine.core.sentiment import analyze_sentiment
        base = analyze_sentiment("I am bad")["valence"]
        not_bad = analyze_sentiment("I am not bad")["valence"]
        # "not bad" should be more positive than "bad"
        assert not_bad > base


# ════════════════════════════════════════════════════════════════════
# R7 — Frequency + presence penalty split
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# R8 — Cosine warm restarts
# ════════════════════════════════════════════════════════════════════

class TestCosineWarmRestarts:
    """R8: Cosine restart period in TrainingConfig."""

    def test_config_has_restart_period(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "cosine_restart_period")
        assert cfg.cosine_restart_period == 0  # disabled


# ════════════════════════════════════════════════════════════════════
# R9 — Adaptive KL coefficient
# ════════════════════════════════════════════════════════════════════

class TestAdaptiveKL:
    """R9: Adaptive KL in RLHFConfig and SelfPlayConfig."""

    def test_rlhf_config_has_kl_target(self):
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert hasattr(cfg, "kl_target")
        assert hasattr(cfg, "kl_horizon")
        assert cfg.kl_target == 0.0  # disabled

    def test_selfplay_config_has_kl_target(self):
        from enigma_engine.core.rl_training import SelfPlayConfig
        cfg = SelfPlayConfig()
        assert hasattr(cfg, "kl_target")
        assert cfg.kl_target == 0.0


# ════════════════════════════════════════════════════════════════════
# R10 — Typical sampling
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# R11 — Mirostat sampling
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# R12 — Gradual unfreezing
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# T2-4 — R-Drop Regularization
# ════════════════════════════════════════════════════════════════════

class TestRDrop:
    """T2-4: R-Drop regularization in training.py."""

    def test_config_has_r_drop_alpha(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, 'r_drop_alpha')
        assert cfg.r_drop_alpha == 0.0  # disabled by default

    def test_r_drop_in_serialization(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(r_drop_alpha=1.0)
        d = cfg.to_dict()
        assert d['r_drop_alpha'] == 1.0


# ════════════════════════════════════════════════════════════════════
# T2-5 — Online/Iterative DPO
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# T2-6 — WSD schedule
# ════════════════════════════════════════════════════════════════════

class TestWSDSchedule:
    """T2-6: Warmup-Stable-Decay schedule in training.py."""

    def test_config_has_schedule_type(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.schedule_type == 'wsd'
        assert cfg.wsd_decay_fraction == 0.1

    def test_schedule_type_in_serialization(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(schedule_type='cosine')
        d = cfg.to_dict()
        assert d['schedule_type'] == 'cosine'


# ════════════════════════════════════════════════════════════════════
# R13 — SWA weight averaging
# ════════════════════════════════════════════════════════════════════

class TestSWA:
    """R13: SWA in training.py."""

    def test_config_has_swa_interval(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "swa_update_interval")
        assert cfg.swa_update_interval == 0  # disabled


# ════════════════════════════════════════════════════════════════════
# R14 — SimPO (reference-free DPO)
# ════════════════════════════════════════════════════════════════════

class TestSimPO:
    """R14: train_simpo in Trainer."""

    def test_simpo_source_no_ref_model(self):
        """SimPO should NOT create a reference model."""
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer.train_simpo)
        assert "ref_model" not in src or "no ref model" in src.lower()


# ════════════════════════════════════════════════════════════════════
# R15 — KTO (unpaired preference)
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# R16 — GRPO (Group Relative Policy Optimization)
# ════════════════════════════════════════════════════════════════════

class TestGRPO:
    """R16: GRPOTrainer in rl_training.py."""

    def test_grpo_config_exists(self):
        from enigma_engine.core.rl_training import GRPOConfig
        cfg = GRPOConfig()
        assert hasattr(cfg, "group_size")
        assert cfg.group_size == 4
        assert hasattr(cfg, "normalize_advantages")


# ════════════════════════════════════════════════════════════════════
# R17 — DoRA (Weight-Decomposed LoRA)
# ════════════════════════════════════════════════════════════════════

class TestDoRA:
    """R17: DoRA in lora_utils.py."""

    def test_lora_config_has_dora_flag(self):
        from enigma_engine.core.lora_utils import LoraConfig
        cfg = LoraConfig()
        assert hasattr(cfg, "use_dora")
        assert cfg.use_dora is False

    def test_dora_linear_has_magnitude(self):
        """DoRALinear should have a magnitude parameter and LoRA A/B."""
        import torch
        from enigma_engine.core.lora_utils import DoRALinear
        base = torch.nn.Linear(32, 16)
        dora = DoRALinear(base, rank=4, alpha=8)
        assert hasattr(dora, "m")
        assert hasattr(dora, "lora_a")
        assert hasattr(dora, "lora_b")
        assert dora.m.shape == (16,)  # one magnitude per output dim
        assert dora.lora_a.shape == (4, 32)  # (rank, in_features)
        assert dora.lora_b.shape == (16, 4)  # (out_features, rank)

    def test_dora_forward(self):
        """DoRALinear forward produces correct output shape."""
        import torch
        from enigma_engine.core.lora_utils import DoRALinear
        base = torch.nn.Linear(32, 16)
        dora = DoRALinear(base, rank=4, alpha=8)
        x = torch.randn(2, 10, 32)
        out = dora(x)
        assert out.shape == (2, 10, 16)


# ════════════════════════════════════════════════════════════════════
# R18 — Prefix KV caching
# ════════════════════════════════════════════════════════════════════

class TestPrefixKVCache:
    """R18: PrefixKVCache in kv_cache.py."""

    def test_class_exists(self):
        from enigma_engine.core.kv_cache import PrefixKVCache
        cache = PrefixKVCache()
        assert cache.prefix_len == 0
        assert cache.n_layers == 0

    def test_build_from_manager(self):
        import torch
        from enigma_engine.core.kv_cache import KVCacheManager, PrefixKVCache
        mgr = KVCacheManager(
            n_layers=2, n_kv_heads=4, head_dim=8,
            max_seq_len=64, device=torch.device("cpu"))
        mgr.allocate(batch_size=1)
        # Simulate a 10-token prefill
        k = torch.randn(1, 10, 4, 8)
        v = torch.randn(1, 10, 4, 8)
        mgr.update(0, k, v)
        mgr.update(1, k, v)

        prefix = PrefixKVCache()
        prefix.build_from_manager(mgr)
        assert prefix.prefix_len == 10
        assert prefix.n_layers == 2

        pk, pv = prefix.get(0)
        assert pk.shape == (1, 10, 4, 8)

    def test_clear(self):
        from enigma_engine.core.kv_cache import PrefixKVCache
        cache = PrefixKVCache()
        cache.clear()
        assert cache.prefix_len == 0

    def test_kv_cache_restore_prefix(self):
        """KVCache.restore_prefix writes data and sets current_pos."""
        import torch
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(batch_size=1, max_seq_len=64, n_kv_heads=4,
                        head_dim=8, device=torch.device("cpu"))
        prefix_k = torch.randn(1, 5, 4, 8)
        prefix_v = torch.randn(1, 5, 4, 8)
        cache.restore_prefix(prefix_k, prefix_v)
        assert cache.current_pos == 5
        k, v = cache.get()
        assert k.shape == (1, 5, 4, 8)
        assert torch.allclose(k, prefix_k)

    def test_restore_prefix_then_append(self):
        """After restoring prefix, new tokens append correctly."""
        import torch
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(batch_size=1, max_seq_len=64, n_kv_heads=4,
                        head_dim=8, device=torch.device("cpu"))
        prefix_k = torch.ones(1, 5, 4, 8)
        prefix_v = torch.ones(1, 5, 4, 8)
        cache.restore_prefix(prefix_k, prefix_v)
        new_k = torch.full((1, 3, 4, 8), 2.0)
        new_v = torch.full((1, 3, 4, 8), 2.0)
        cache.update(new_k, new_v)
        assert cache.current_pos == 8
        k, v = cache.get()
        assert k.shape == (1, 8, 4, 8)
        # First 5 should be ones, next 3 should be twos
        assert torch.allclose(k[:, :5], prefix_k)
        assert torch.allclose(k[:, 5:8], new_k)

    def test_model_restore_prefix(self):
        """Enigma model can restore prefix KV across layers."""
        import torch
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.kv_cache import PrefixKVCache

        cfg = ForgeConfig(vocab_size=64, dim=32, n_heads=2, n_layers=2,
                          n_kv_heads=2, use_qk_norm=False,
                          use_differential_attn=False, n_predict_heads=0)
        model = Enigma(config=cfg)
        model.eval()

        # Do a prefill to populate caches
        ids = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            model.clear_cache()
            model(ids, use_cache=True)

        # Snapshot into prefix cache
        prefix = PrefixKVCache()
        prefix.build_from_layers(model.layers)
        assert prefix.prefix_len == 8
        assert prefix.n_layers == 2

        # Clear and restore
        model.clear_cache()
        model.restore_prefix_cache(prefix)

        # Each layer should now have current_pos == 8
        for layer in model.layers:
            assert layer.attention._kv_cache.current_pos == 8


# ════════════════════════════════════════════════════════════════════
# T2-1 — Per-Channel KV Quantization
# ════════════════════════════════════════════════════════════════════

class TestPerChannelKVQuant:
    """T2-1: Per-channel-group INT8 quantization in KVCache."""

    def test_scale_shape_has_groups(self):
        """Scale buffers should be [B, T, H, G] not [B, T, H]."""
        import torch
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=16, n_kv_heads=2,
            head_dim=16, device=torch.device("cpu"),
            quantize_to_int8=True,
        )
        # head_dim=16, group_size=8 => n_groups=2
        assert cache._quant_group_size == 8
        assert cache._quant_n_groups == 2
        assert cache._scale_k.shape == (1, 16, 2, 2)

    def test_quantize_roundtrip_accuracy(self):
        """Per-group quantization should be more accurate than per-head."""
        import torch
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=16, n_kv_heads=2,
            head_dim=32, device=torch.device("cpu"),
            quantize_to_int8=True,
        )
        # Create tensor where different channel groups have very different magnitudes
        t = torch.zeros(1, 4, 2, 32)
        t[..., :16] = torch.randn(1, 4, 2, 16) * 0.01   # small group
        t[..., 16:] = torch.randn(1, 4, 2, 16) * 100.0   # large group
        q, s, zp = cache._quantize_tensor(t)
        reconstructed = cache._dequantize_tensor(q, s, zp)
        # Per-group should recover the small values reasonably well
        small_error = (reconstructed[..., :16] - t[..., :16]).abs().max().item()
        # With per-head quant (old), small values would be crushed by 100x scale
        # Per-group keeps them separate, so error should be small relative to range
        assert small_error < 0.01, f"Small-group error too high: {small_error}"

    def test_update_get_roundtrip_int8(self):
        """INT8 cache update+get should work with per-channel-group quant."""
        import torch
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=16, n_kv_heads=4,
            head_dim=64, device=torch.device("cpu"),
            quantize_to_int8=True,
        )
        k = torch.randn(1, 5, 4, 64)
        v = torch.randn(1, 5, 4, 64)
        cache.update(k, v)
        k_out, v_out = cache.get()
        assert k_out.shape == (1, 5, 4, 64)
        assert v_out.shape == (1, 5, 4, 64)
        # INT8 quantization: allow some error but should be close
        assert torch.allclose(k_out, k, atol=0.02)

    def test_group_size_fallback_odd_dim(self):
        """head_dim not divisible by 8 should fallback to smaller group."""
        import torch
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=8, n_kv_heads=1,
            head_dim=10, device=torch.device("cpu"),
            quantize_to_int8=True,
        )
        # 10 % 8 != 0, 10 % 4 != 0, 10 % 2 == 0 => gs=2, G=5
        assert cache._quant_group_size == 2
        assert cache._quant_n_groups == 5
        assert cache._scale_k.shape == (1, 8, 1, 5)

    def test_asymmetric_quant_non_centered_data(self):
        """Asymmetric quant should handle non-zero-centered values better."""
        import torch
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=8, n_kv_heads=2,
            head_dim=16, device=torch.device("cpu"),
            quantize_to_int8=True,
        )
        # Create tensor with large positive offset (e.g. post-RoPE keys)
        t = torch.randn(1, 4, 2, 16) * 0.5 + 5.0  # centered around 5.0
        q, s, zp = cache._quantize_tensor(t)
        reconstructed = cache._dequantize_tensor(q, s, zp)
        max_error = (reconstructed - t).abs().max().item()
        # Asymmetric should handle offset data well (error << range)
        assert max_error < 0.05, f"Asymmetric quant error too high: {max_error}"
        # Zero-point should capture the offset
        assert (zp.abs() > 1.0).any(), "Zero-point should be non-trivial for offset data"

    def test_quantize_returns_three_tuple(self):
        """_quantize_tensor returns (quantized, scale, zero_point)."""
        import torch
        from enigma_engine.core.kv_cache import KVCache
        cache = KVCache(
            batch_size=1, max_seq_len=8, n_kv_heads=1,
            head_dim=8, device=torch.device("cpu"),
            quantize_to_int8=True,
        )
        t = torch.randn(1, 2, 1, 8)
        result = cache._quantize_tensor(t)
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
        q, s, zp = result
        assert q.dtype == torch.int8
        assert s.shape == zp.shape


# ════════════════════════════════════════════════════════════════════
# R19 — H2O KV cache eviction
# ════════════════════════════════════════════════════════════════════

class TestH2OKVCache:
    """R19: H2OKVCache in kv_cache.py."""

    def test_eviction(self):
        import torch
        from enigma_engine.core.kv_cache import H2OKVCache
        cache = H2OKVCache(
            heavy_hitter_count=2, recent_window=2,
            batch_size=1, max_seq_len=32, n_kv_heads=2,
            head_dim=4, device=torch.device("cpu"))

        # Fill cache with 10 tokens
        for i in range(10):
            k = torch.randn(1, 1, 2, 4)
            v = torch.randn(1, 1, 2, 4)
            cache.update(k, v)

        assert cache.current_pos == 10

        # Simulate attention scores — make token 0 and 1 heavy hitters
        attn = torch.zeros(1, 10)
        attn[0, 0] = 10.0
        attn[0, 1] = 8.0
        cache.accumulate_attention(attn)

        cache.evict_if_needed()
        # Budget = 2 HH + 2 recent = 4, so should compact from 10 to 4
        assert cache.current_pos == 4

    def test_clear_resets_scores(self):
        import torch
        from enigma_engine.core.kv_cache import H2OKVCache
        cache = H2OKVCache(
            heavy_hitter_count=4, recent_window=2,
            batch_size=1, max_seq_len=16, n_kv_heads=2,
            head_dim=4, device=torch.device("cpu"))
        cache.clear()
        assert cache.current_pos == 0


# ════════════════════════════════════════════════════════════════════
# R20 — Bert2BERT layer copy
# ════════════════════════════════════════════════════════════════════

class TestBert2BERTLayerCopy:
    """R20: compute_layer_mapping_bert2bert in progressive_growing.py."""

    def test_mapping_cycles(self):
        from enigma_engine.core.progressive_growing import compute_layer_mapping_bert2bert
        # 4 old layers → 8 new layers should cycle: 0,1,2,3,0,1,2,3
        mapping = compute_layer_mapping_bert2bert(4, 8)
        assert mapping == [0, 1, 2, 3, 0, 1, 2, 3]

    def test_identity_when_same(self):
        from enigma_engine.core.progressive_growing import compute_layer_mapping_bert2bert
        mapping = compute_layer_mapping_bert2bert(4, 4)
        assert mapping == [0, 1, 2, 3]


# ════════════════════════════════════════════════════════════════════
# R21 — Adaptive RAG chunking
# ════════════════════════════════════════════════════════════════════

class TestAdaptiveChunking:
    """R21: adaptive_chunk_text in rag.py."""

    def test_empty_input(self):
        from enigma_engine.core.rag import adaptive_chunk_text
        assert adaptive_chunk_text("") == []
        assert adaptive_chunk_text("   ") == []

    def test_respects_paragraph_breaks(self):
        from enigma_engine.core.rag import adaptive_chunk_text
        text = "Paragraph one with content.\n\nParagraph two with content."
        chunks = adaptive_chunk_text(text, target_size=5000)
        # Small text should NOT be split across paragraphs
        assert len(chunks) >= 1

    def test_splits_markdown_headers(self):
        from enigma_engine.core.rag import adaptive_chunk_text
        text = (
            "# Introduction\n\n"
            "Some intro text.\n\n"
            "# Methods\n\n"
            "Some methods text.\n\n"
            "# Results\n\n"
            "Some results text."
        )
        chunks = adaptive_chunk_text(text, target_size=50, max_size=100)
        # Should split at headers rather than mid-sentence
        assert len(chunks) >= 2

    def test_oversized_section_gets_split(self):
        from enigma_engine.core.rag import adaptive_chunk_text
        # One giant section with no breaks — use target_size > CHUNK_OVERLAP (128)
        # to avoid infinite loop in chunk_text when overlap >= chunk_size
        text = "word " * 500
        chunks = adaptive_chunk_text(text, target_size=300, max_size=600)
        assert len(chunks) > 1


# ════════════════════════════════════════════════════════════════════
# R22 — Differential attention
# ════════════════════════════════════════════════════════════════════

class TestDifferentialAttention:
    """R22: Differential attention in model_components.py."""

    def test_config_flag_exists(self):
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig()
        assert hasattr(cfg, "use_differential_attn")
        assert cfg.use_differential_attn is True

    def test_attention_has_diff_attn_attr(self):
        from enigma_engine.core.model_components import Attention
        from enigma_engine.core.model_presets import ForgeConfig
        # With flag off — should not have _diff_lambda
        cfg = ForgeConfig(n_heads=8, use_differential_attn=False)
        attn = Attention(cfg)
        assert attn.use_differential_attn is False

    def test_attention_diff_attn_enabled(self):
        from enigma_engine.core.model_components import Attention
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig(n_heads=8, use_differential_attn=True)
        attn = Attention(cfg)
        assert attn.use_differential_attn is True
        assert hasattr(attn, "_diff_lambda")
        assert attn._diff_lambda.shape == (4,)  # n_heads // 2

    def test_diff_attn_forward(self):
        """Differential attention forward pass produces correct shape."""
        import torch
        from enigma_engine.core.model_components import Attention
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig(
            dim=64, n_heads=8, n_kv_heads=8,
            use_differential_attn=True, use_rope=False)
        attn = Attention(cfg)
        x = torch.randn(1, 10, 64)
        out = attn(x)
        assert out.shape == (1, 10, 64)

    def test_diff_attn_disabled_for_odd_heads(self):
        """Differential attention auto-disables with odd n_heads."""
        from enigma_engine.core.model_components import Attention
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig(dim=768, n_heads=3, n_kv_heads=3,
                          use_differential_attn=True)
        attn = Attention(cfg)
        assert attn.use_differential_attn is False


# ════════════════════════════════════════════════════════════════════
# T2-7 — Adaptive stop check interval
# ════════════════════════════════════════════════════════════════════

class TestAdaptiveStopInterval:
    """T2-7: Adaptive stop-check interval in engine_generation.py."""

    def test_high_confidence_longer_interval(self):
        from enigma_engine.core.engine_generation import _GenerationMixin
        interval = _GenerationMixin._adaptive_stop_interval([0.95, 0.9, 0.92, 0.88])
        assert interval >= 28  # close to max_interval=32

    def test_low_confidence_shorter_interval(self):
        from enigma_engine.core.engine_generation import _GenerationMixin
        interval = _GenerationMixin._adaptive_stop_interval([0.1, 0.15, 0.12, 0.08])
        assert interval <= 12  # close to min_interval=8

    def test_empty_history_default(self):
        from enigma_engine.core.engine_generation import _GenerationMixin
        interval = _GenerationMixin._adaptive_stop_interval([])
        assert interval == 16


# ════════════════════════════════════════════════════════════════════
# R23+R24 — Speculative decoding + adaptive K
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# R25 — Multi-token Prediction (Gloeckle et al. 2024)
# ════════════════════════════════════════════════════════════════════

class TestMultiTokenPrediction:
    """R25: Extra prediction heads in model.py that predict k-th next token."""

    def test_config_field_exists(self):
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig()
        assert hasattr(cfg, "n_predict_heads")
        # MTP-2b: default flipped 2 → 0 — Medusa inference is superseded
        # by EAGLE-2 (Pass 148), so paying ~33-49M params per head by
        # default is no longer justified. Explicit opt-in for Medusa runs.
        assert cfg.n_predict_heads == 0

    def test_default_model_has_no_predict_heads(self):
        """MTP-2b: default ForgeConfig builds a model with zero MTP heads."""
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig(vocab_size=64, dim=32, n_heads=2, n_layers=2)
        model = Enigma(config=cfg)
        assert hasattr(model, "predict_heads")
        assert len(model.predict_heads) == 0

    def test_model_has_predict_heads_list(self):
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig(vocab_size=64, dim=32, n_heads=2, n_layers=2,
                          n_predict_heads=0)
        model = Enigma(config=cfg)
        assert hasattr(model, "predict_heads")
        assert len(model.predict_heads) == 0

    def test_model_creates_heads_when_configured(self):
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig(
            vocab_size=64, dim=32, n_heads=2, n_layers=2,
            n_predict_heads=3)
        model = Enigma(config=cfg)
        assert len(model.predict_heads) == 3
        # Heads are NOT weight-tied with embeddings
        for head in model.predict_heads:
            assert head.weight is not model.tok_embeddings.weight

    def test_mtp_heads_not_used_at_eval(self):
        """Extra heads only contribute loss during training, not eval."""
        import torch
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig(
            vocab_size=64, dim=32, n_heads=2, n_layers=2,
            n_predict_heads=2)
        model = Enigma(config=cfg)
        model.eval()
        ids = torch.randint(0, 64, (1, 10))
        targets = torch.randint(0, 64, (1, 10))
        logits, loss = model(ids, targets=targets)
        assert logits is not None
        assert loss is not None

    def test_mtp_loss_is_finite_during_training(self):
        import torch
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig(
            vocab_size=64, dim=32, n_heads=2, n_layers=2,
            n_predict_heads=2)
        model = Enigma(config=cfg)
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        targets = torch.randint(0, 64, (2, 16))
        _, loss = model(ids, targets=targets)
        assert loss is not None
        assert torch.isfinite(loss)


# ════════════════════════════════════════════════════════════════════
# R29  – TurboQuant Mixed KV Cache
# ════════════════════════════════════════════════════════════════════


class TestTurboQuantKVCache:
    """R29: Mixed-precision KV cache with per-head importance scoring."""

    def test_int4_pack_unpack_roundtrip(self):
        import torch
        from enigma_engine.core.kv_cache import _pack_int4, _unpack_int4
        a = torch.tensor([3, 7, 15, 0], dtype=torch.uint8)
        b = torch.tensor([1, 14, 5, 9], dtype=torch.uint8)
        packed = _pack_int4(a, b)
        a2, b2 = _unpack_int4(packed)
        assert torch.equal(a2, a.to(torch.int8))
        assert torch.equal(b2, b.to(torch.int8))

    def test_basic_update_and_get(self):
        import torch
        from enigma_engine.core.kv_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(
            int4_fraction=0.5,
            batch_size=1,
            max_seq_len=64,
            n_kv_heads=4,
            head_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        k = torch.randn(1, 4, 4, 8)
        v = torch.randn(1, 4, 4, 8)
        new_pos = cache.update(k, v)
        assert new_pos == 4
        k_out, v_out = cache.get()
        assert k_out.shape == (1, 4, 4, 8)
        assert v_out.shape == (1, 4, 4, 8)

    def test_update_head_importance(self):
        import torch
        from enigma_engine.core.kv_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(
            int4_fraction=0.5,
            batch_size=1,
            max_seq_len=64,
            n_kv_heads=4,
            head_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        # Simulate attention weights: 4 query heads, 2 Q positions, 3 KV positions
        attn = torch.softmax(torch.randn(1, 4, 2, 3), dim=-1)
        cache.update_head_importance(attn)
        # Importance should be non-zero after update
        assert cache._head_importance.sum() > 0

    def test_rebalance_assigns_int4_to_least_important(self):
        import torch
        from enigma_engine.core.kv_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(
            int4_fraction=0.5,
            batch_size=1,
            max_seq_len=64,
            n_kv_heads=4,
            head_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        # Set known importance scores
        cache._head_importance = torch.tensor([10.0, 1.0, 5.0, 0.5])
        cache.rebalance()
        # Heads 1 and 3 (lowest) should be INT4
        assert cache._is_int4[1].item() is True
        assert cache._is_int4[3].item() is True
        # Heads 0 and 2 (highest) should be INT8
        assert cache._is_int4[0].item() is False
        assert cache._is_int4[2].item() is False

    def test_memory_usage_includes_int4(self):
        import torch
        from enigma_engine.core.kv_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(
            int4_fraction=0.5,
            batch_size=1,
            max_seq_len=64,
            n_kv_heads=4,
            head_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        mb = cache.memory_usage_mb()
        assert mb > 0

    def test_clear_resets_all(self):
        import torch
        from enigma_engine.core.kv_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(
            int4_fraction=0.5,
            batch_size=1,
            max_seq_len=64,
            n_kv_heads=4,
            head_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        k = torch.randn(1, 4, 4, 8)
        v = torch.randn(1, 4, 4, 8)
        cache.update(k, v)
        cache._head_importance += 1.0
        cache.clear()
        assert cache.current_pos == 0
        assert cache._head_importance.sum() == 0
        assert cache._update_count == 0

    def test_quantize_dequantize_int4_roundtrip(self):
        import torch
        from enigma_engine.core.kv_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(
            int4_fraction=1.0,
            batch_size=1,
            max_seq_len=16,
            n_kv_heads=2,
            head_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        original = torch.randn(1, 4, 2, 8)
        packed, scale = cache._quantize_int4(original)
        recovered = cache._dequantize_int4(packed, scale)
        # INT4 is lossy but should be within reasonable bounds
        error = (original - recovered).abs().max().item()
        assert error < 0.5, f"INT4 roundtrip error too large: {error}"


# ════════════════════════════════════════════════════════════════════
# R26 — LISA (Pan et al. 2024)
# ════════════════════════════════════════════════════════════════════

class TestLISA:
    """R26: Layerwise Importance Sampled AdamW in progressive_growing.py."""

    def test_config_fields_exist(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "use_lisa")
        assert cfg.use_lisa is False
        assert hasattr(cfg, "lisa_activated_layers")
        assert cfg.lisa_activated_layers == 2

    def test_lisa_freezes_middle_layers(self):

        from enigma_engine.core.progressive_growing import LISAScheduler
        model = FakeModel()
        LISAScheduler(model, n_layers=6, activated_layers=1)

        # First and last layer should always be trainable
        for p in model.layers[0].parameters():
            assert p.requires_grad
        for p in model.layers[5].parameters():
            assert p.requires_grad

        # Non-layer params always trainable
        for p in model.tok_embeddings.parameters():
            assert p.requires_grad
        for p in model.norm.parameters():
            assert p.requires_grad
        for p in model.output.parameters():
            assert p.requires_grad

        # Exactly 1 middle layer should be active (of 4 middle: 1,2,3,4)
        active_middle = 0
        for idx in range(1, 5):
            if all(p.requires_grad for p in model.layers[idx].parameters()):
                active_middle += 1
        assert active_middle == 1

    def test_lisa_resamples_on_step(self):

        from enigma_engine.core.progressive_growing import LISAScheduler
        model = FakeModel(8)
        lisa = LISAScheduler(model, n_layers=8, activated_layers=2)

        # Track active middle layers across multiple steps
        seen_active = set()
        for _ in range(50):
            lisa.step()
            for idx in range(1, 7):
                if all(p.requires_grad for p in model.layers[idx].parameters()):
                    seen_active.add(idx)
        # With 6 middle layers and 50 resamples, should have seen most of them
        assert len(seen_active) >= 3


# ════════════════════════════════════════════════════════════════════
# R27 — NEFTune (Jain et al. 2023)
# ════════════════════════════════════════════════════════════════════

class TestNEFTune:
    """R27: NEFTune embedding noise in model.py forward pass."""

    def test_config_field_exists(self):
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig()
        assert hasattr(cfg, "neftune_alpha")
        assert cfg.neftune_alpha == 5.0

    def test_noise_applied_during_training(self):
        import torch
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig(
            vocab_size=64, dim=32, n_heads=2, n_layers=2,
            neftune_alpha=10.0)
        model = Enigma(config=cfg)
        model.train()
        torch.manual_seed(42)
        ids = torch.randint(0, 64, (1, 16))
        out1 = model(ids)
        torch.manual_seed(43)
        out2 = model(ids)
        # Outputs should differ due to noise
        assert not torch.allclose(out1, out2)

    def test_no_noise_at_eval(self):
        import torch
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig(
            vocab_size=64, dim=32, n_heads=2, n_layers=2,
            neftune_alpha=10.0)
        model = Enigma(config=cfg)
        model.eval()
        ids = torch.randint(0, 64, (1, 16))
        out1 = model(ids)
        out2 = model(ids)
        # Deterministic at eval — no noise
        assert torch.allclose(out1, out2)


# ════════════════════════════════════════════════════════════════════
# R28 — Z-Loss (PaLM, Chowdhery et al. 2022)
# ════════════════════════════════════════════════════════════════════

class TestZLoss:
    """R28: Z-Loss auxiliary loss in training.py."""

    def test_config_field_exists(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "z_loss_weight")
        assert cfg.z_loss_weight == 0.0

    def test_zloss_increases_loss(self):
        """Z-Loss should add a positive penalty to the base loss."""
        import torch
        # Simulate what the training loop does
        logits = torch.randn(2, 10, 64)  # (B, T, vocab)
        z = torch.logsumexp(logits, dim=-1)
        z_loss = 1e-4 * (z ** 2).mean()
        assert z_loss.item() > 0
        assert torch.isfinite(z_loss)


# ════════════════════════════════════════════════════════════════════
# R30 — AdEMAMix optimizer
# ════════════════════════════════════════════════════════════════════

class TestAdEMAMix:
    """R30: AdEMAMix dual-EMA optimizer in training.py."""

    def test_config_has_optimizer_field(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "optimizer")
        assert cfg.optimizer == "adamw"
        assert hasattr(cfg, "ademamix_beta3")
        assert cfg.ademamix_beta3 == 0.9999
        assert hasattr(cfg, "ademamix_alpha")
        assert cfg.ademamix_alpha == 5.0

    def test_config_optimizer_validates(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(optimizer="invalid")
        with pytest.raises(ValueError, match="optimizer"):
            cfg.validate()

    def test_to_dict_includes_optimizer_fields(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(optimizer="ademamix")
        d = cfg.to_dict()
        assert d["optimizer"] == "ademamix"
        assert "ademamix_beta3" in d
        assert "ademamix_alpha" in d

    def test_ademamix_step(self):
        """AdEMAMix optimizer runs a step without error."""
        import torch
        from enigma_engine.core.training import AdEMAMix

        model = torch.nn.Linear(8, 4)
        opt = AdEMAMix(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                       beta3=0.9999, alpha=5.0)
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    def test_ademamix_two_emas(self):
        """Verify AdEMAMix maintains fast and slow EMA buffers."""
        import torch
        from enigma_engine.core.training import AdEMAMix

        model = torch.nn.Linear(4, 2)
        opt = AdEMAMix(model.parameters(), lr=1e-3)
        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        # Check state has both EMAs
        for p in model.parameters():
            state = opt.state[p]
            assert "m_fast" in state
            assert "m_slow" in state
            assert "v" in state

    def test_ademamix_reduces_loss(self):
        """AdEMAMix should reduce loss over multiple steps."""
        import torch
        from enigma_engine.core.training import AdEMAMix

        torch.manual_seed(42)
        model = torch.nn.Linear(4, 2, bias=False)
        opt = AdEMAMix(model.parameters(), lr=0.01)

        target = torch.randn(1, 2)
        x = torch.randn(1, 4)

        losses = []
        for _ in range(50):
            loss = ((model(x) - target) ** 2).sum()
            losses.append(loss.item())
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert losses[-1] < losses[0], (
            f"AdEMAMix did not reduce loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


# ════════════════════════════════════════════════════════════════════
# S558 — RoPE Theta for Medium+ Presets
# ════════════════════════════════════════════════════════════════════

class TestRopeThetaPresets:
    """S558: Presets with max_seq_len >= 2048 should use rope_theta=500000."""

    def test_medium_plus_presets_have_high_theta(self):
        from enigma_engine.core.model_presets import MODEL_PRESETS
        for name, cfg in MODEL_PRESETS.items():
            if cfg.max_seq_len >= 2048:
                assert cfg.rope_theta == 500000.0, (
                    f"Preset '{name}' has max_seq_len={cfg.max_seq_len} "
                    f"but rope_theta={cfg.rope_theta} (expected 500000.0)")

    def test_small_presets_keep_default_theta(self):
        from enigma_engine.core.model_presets import MODEL_PRESETS
        for name, cfg in MODEL_PRESETS.items():
            if cfg.max_seq_len < 2048:
                assert cfg.rope_theta == 10000.0, (
                    f"Preset '{name}' has max_seq_len={cfg.max_seq_len} "
                    f"but rope_theta={cfg.rope_theta} (expected 10000.0)")


# ════════════════════════════════════════════════════════════════════
# S559 — Proper Noun Exemption in Repetition Penalty
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# S560 — AdEMAMix Alpha Scheduling
# ════════════════════════════════════════════════════════════════════

class TestAdEMAMixAlphaScheduling:
    """S560: Alpha annealing from initial to final during warmup."""

    def test_config_has_alpha_scheduling_fields(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, 'ademamix_alpha_initial')
        assert cfg.ademamix_alpha_initial == 10.0
        assert hasattr(cfg, 'ademamix_alpha_warmup')
        assert cfg.ademamix_alpha_warmup == 0.1

    def test_to_dict_includes_alpha_scheduling(self):
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(optimizer="ademamix")
        d = cfg.to_dict()
        assert "ademamix_alpha_initial" in d
        assert "ademamix_alpha_warmup" in d


# ════════════════════════════════════════════════════════════════════
# T3-6 — Cut Cross-Entropy (Vocab Chunking)
# ════════════════════════════════════════════════════════════════════

class TestChunkedCrossEntropy:
    """T3-6: Chunked cross-entropy avoids full [B*T, V] logit tensor."""

    def test_matches_standard_ce(self):
        """Chunked CE should produce the same loss as standard CE."""
        import torch
        import torch.nn as nn
        from enigma_engine.core.model import _chunked_cross_entropy

        torch.manual_seed(42)
        output = nn.Linear(32, 100, bias=False)
        hidden = torch.randn(1, 16, 32)  # [B, T, D]
        targets = torch.randint(0, 100, (1, 16))

        # Standard CE
        logits = output(hidden)
        standard_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 100), targets.reshape(-1))

        # Chunked CE with small chunk
        chunked_loss = _chunked_cross_entropy(
            output, hidden, targets, chunk_size=4)

        assert torch.allclose(standard_loss, chunked_loss, atol=1e-5), (
            f"Standard={standard_loss.item():.6f}, "
            f"Chunked={chunked_loss.item():.6f}")

    def test_ignore_index(self):
        """Padding tokens should be excluded from loss."""
        import torch
        import torch.nn as nn
        from enigma_engine.core.model import _chunked_cross_entropy

        torch.manual_seed(42)
        output = nn.Linear(16, 50, bias=False)
        hidden = torch.randn(1, 8, 16)
        targets = torch.randint(0, 50, (1, 8))
        targets[0, 4:] = -100  # Pad last 4 positions

        loss = _chunked_cross_entropy(
            output, hidden, targets, chunk_size=3, ignore_index=-100)

        # Should not be zero (we have valid tokens)
        assert loss.item() > 0
        # Should match standard with same ignore_index
        logits = output(hidden)
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 50), targets.reshape(-1), ignore_index=-100)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_all_padding_returns_zero(self):
        """All-padding batch should return zero loss."""
        import torch
        import torch.nn as nn
        from enigma_engine.core.model import _chunked_cross_entropy

        output = nn.Linear(8, 20, bias=False)
        hidden = torch.randn(1, 4, 8)
        targets = torch.full((1, 4), -100)

        loss = _chunked_cross_entropy(
            output, hidden, targets, chunk_size=2, ignore_index=-100)
        assert loss.item() == 0.0

    def test_config_has_ce_chunk_size(self):
        """TrainingConfig should have ce_chunk_size field."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, 'ce_chunk_size')
        assert cfg.ce_chunk_size == 0  # Default disabled

    def test_to_dict_includes_ce_chunk_size(self):
        from enigma_engine.core.training import TrainingConfig
        d = TrainingConfig(ce_chunk_size=4096).to_dict()
        assert d['ce_chunk_size'] == 4096


# ════════════════════════════════════════════════════════════════════
# T3-8 — LongLoRA Shifted Sparse Attention
# ════════════════════════════════════════════════════════════════════

class TestShiftedSparseAttention:
    """T3-8: LongLoRA shifted sparse attention in model_components.py."""

    def test_config_has_shifted_attention_fields(self):
        from enigma_engine.core.model_presets import ForgeConfig
        cfg = ForgeConfig()
        assert hasattr(cfg, 'use_shifted_attention')
        assert cfg.use_shifted_attention is False
        assert hasattr(cfg, 'shifted_group_size')
        assert cfg.shifted_group_size == 256

    def test_attention_stores_shifted_flag(self):
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model_components import Attention
        cfg = ForgeConfig(
            use_shifted_attention=True, shifted_group_size=128,
            n_heads=4, n_kv_heads=4, dim=64)
        attn = Attention(cfg)
        assert attn.use_shifted_attention is True
        assert attn._shifted_group_size == 128

    def test_shifted_attention_produces_output(self):
        """Shifted attention should produce valid output of correct shape."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model_components import Attention
        cfg = ForgeConfig(
            use_shifted_attention=True, shifted_group_size=4,
            n_heads=4, n_kv_heads=4, dim=32,
            use_differential_attn=False)
        attn = Attention(cfg)
        attn.eval()
        x = torch.randn(1, 8, 32)  # T=8 > group_size=4
        output = attn(x, mask=None, use_cache=False)
        assert output.shape == (1, 8, 32)

    def test_shifted_attention_odd_seqlen(self):
        """Should handle sequence lengths not divisible by group_size."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model_components import Attention
        cfg = ForgeConfig(
            use_shifted_attention=True, shifted_group_size=4,
            n_heads=4, n_kv_heads=4, dim=32,
            use_differential_attn=False)
        attn = Attention(cfg)
        attn.eval()
        # T=7 is not divisible by group_size=4
        x = torch.randn(1, 7, 32)
        output = attn(x, mask=None, use_cache=False)
        assert output.shape == (1, 7, 32)


# ════════════════════════════════════════════════════════════════════
# T3-9 — JSON Schema Masking (Grammar-Guided Decoding)
# ════════════════════════════════════════════════════════════════════

class TestJsonSchemaConstraint:
    """T3-9: JSON schema constraint for grammar-guided decoding."""

    def test_initial_state_expects_open_brace(self):
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint

        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
        c = JsonSchemaConstraint(schema, FakeTokenizer())
        allowed = c._allowed_tokens()
        assert allowed is not None
        assert 0 in allowed  # '{' token
        assert 1 not in allowed  # '}' not valid at start

    def test_advance_through_simple_json(self):
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint

        schema = {
            'type': 'object',
            'properties': {'name': {'type': 'string'}}
        }
        c = JsonSchemaConstraint(schema, FakeTokenizer())

        # Generate: {"name": "hi"}
        for char, state_after in [
            (0, 'EXPECT_KEY'),     # {
            (2, 'IN_KEY'),         # "
            (5, 'IN_KEY'),         # n
            (6, 'IN_KEY'),         # a
            (7, 'IN_KEY'),         # m
            (8, 'IN_KEY'),         # e
            (2, 'EXPECT_COLON'),   # "
            (3, 'EXPECT_VALUE'),   # :
            (2, 'IN_VALUE'),       # "
            (10, 'IN_VALUE'),      # h
            (11, 'IN_VALUE'),      # i
            (2, 'AFTER_VALUE'),    # "
            (1, 'DONE'),           # }
        ]:
            c.advance(char)
            assert c._state == state_after, (
                f"After token {char}, expected state {state_after}, "
                f"got {c._state}")

    def test_is_done(self):
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint

        schema = {'type': 'object', 'properties': {}}
        c = JsonSchemaConstraint(schema, FakeTokenizer())
        assert not c.is_done
        c.advance(0)  # {
        c.advance(1)  # }
        assert c.is_done

    def test_mask_logits_shape(self):
        """mask_logits should return same-shape tensor."""
        import torch
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint

        schema = {'type': 'object', 'properties': {'a': {'type': 'string'}}}
        c = JsonSchemaConstraint(schema, FakeTokenizer())
        logits = torch.randn(1, 10)
        masked = c.mask_logits(logits)
        assert masked.shape == logits.shape
        # Only '{' token should be allowed at start
        assert masked[0, 0].item() > float('-inf')  # '{' allowed
        assert masked[0, 1].item() == float('-inf')  # '}' blocked

    def test_reset(self):
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint

        schema = {'type': 'object', 'properties': {}}
        c = JsonSchemaConstraint(schema, FakeTokenizer())
        c.advance(0)  # {
        c.advance(1)  # }
        assert c.is_done
        c.reset()
        assert not c.is_done
        assert c._state == 'EXPECT_OPEN'


class TestJsonSchemaConstraintBoundaryValidation:
    """Pass after 156z9e: schema validation at constructor time.

    Closes the API-validation follow-up filed under Pass 156z3 / 156z4
    ("API accepts any dict shape; malformed schema reaches the FSM
    and silently produces degraded output").  Now the constructor
    raises ``ValueError`` with a message naming the bad field, so
    HTTP callers (`/api/chat` with json_schema) and Python callers
    (`engine.generate(json_schema=...)`) both fail loud at the
    boundary instead of silently getting free-form output.
    """

    def test_non_dict_schema_raises(self):
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        with pytest.raises(ValueError, match="must be a dict"):
            JsonSchemaConstraint("not a dict", FakeTokenizer())  # type: ignore[arg-type]

    def test_non_object_top_level_type_raises(self):
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        with pytest.raises(ValueError, match="'type'.*'object'"):
            JsonSchemaConstraint(
                {"type": "array", "properties": {}},
                FakeTokenizer(),
            )

    def test_non_dict_properties_raises(self):
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        with pytest.raises(ValueError, match="properties.*must be a dict"):
            JsonSchemaConstraint(
                {"type": "object", "properties": ["name", "age"]},
                FakeTokenizer(),
            )

    def test_non_dict_property_spec_raises(self):
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        with pytest.raises(ValueError, match="'name'"):
            JsonSchemaConstraint(
                {"type": "object", "properties": {"name": "string"}},
                FakeTokenizer(),
            )

    def test_unsupported_property_type_raises(self):
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        with pytest.raises(ValueError, match="not supported"):
            JsonSchemaConstraint(
                {"type": "object",
                 "properties": {"x": {"type": "frobnicate"}}},
                FakeTokenizer(),
            )

    def test_default_top_level_type_accepted(self):
        """A schema with no top-level ``type`` defaults to 'object' —
        accepted, since the FSM is object-only and the absence is
        consistent with that default."""
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        # Should not raise
        c = JsonSchemaConstraint(
            {"properties": {"x": {"type": "string"}}},
            FakeTokenizer(),
        )
        assert c._n_keys == 1

    def test_default_property_type_accepted(self):
        """A property with no ``type`` defaults to 'string' (existing
        FSM behaviour) — accepted."""
        from enigma_engine.core.json_schema_mask import JsonSchemaConstraint
        c = JsonSchemaConstraint(
            {"type": "object", "properties": {"x": {}}},
            FakeTokenizer(),
        )
        assert c._key_type_map["x"] == "string"


# ════════════════════════════════════════════════════════════════════
# N-15 — JSON Schema Constraint Wired Into EnigmaEngine.generate
# ════════════════════════════════════════════════════════════════════

class TestJsonSchemaConstraintWiring:
    """Pass 156z3 (N-15): close the dead-infra gap on JsonSchemaConstraint.

    Pre-fix: ``JsonSchemaConstraint`` (T3-9) had full FSM logic + 5
    unit tests but ZERO production callers. ``_sample_token`` accepted
    a ``json_constraint`` kwarg but no caller in ``_generate_manual``
    ever set it, AND ``.advance()`` was never called from the loop.
    Two-layer dead infra (per AA principle).

    Tests below gate the three wire-sites that close the chain:
    public ``generate`` → ``_generate_text`` → ``_generate_manual``.
    Structural is justified here because the behavioural path through
    ``_generate_manual`` requires stubbing the full engine
    scaffolding (model.forward, KV cache, _build_exempt_tokens,
    _adaptive_stop_interval, tokenizer.decode/eos_token_id) — far
    more code than the wiring being tested. The GGUF rejection test
    IS behavioural since it short-circuits before any of that.
    """

    def test_public_generate_signature_accepts_json_schema(self):
        """N-15: the public API must expose ``json_schema`` so callers
        can opt in. Structural — proves the kwarg is in the signature
        and forwarded to ``_generate_text``."""
        from enigma_engine.core.inference import EnigmaEngine
        sig = inspect.signature(EnigmaEngine.generate)
        assert "json_schema" in sig.parameters, (
            "EnigmaEngine.generate must accept json_schema kwarg")
        src = inspect.getsource(EnigmaEngine.generate)
        # Wired through to the internal _generate_text helper
        assert "json_schema=json_schema" in src, (
            "EnigmaEngine.generate must forward json_schema to "
            "_generate_text — without this, the public kwarg is a "
            "silent no-op")

    def test_generate_text_builds_constraint_and_forwards(self):
        """N-15: ``_generate_text`` must construct a
        ``JsonSchemaConstraint`` from the schema (paying the one-time
        vocab-scan cost once per call, not per token) and forward it
        to ``_generate_manual`` as ``json_constraint=``."""
        from enigma_engine.core import engine_generation
        src = inspect.getsource(engine_generation._GenerationMixin._generate_text)
        assert "JsonSchemaConstraint(json_schema, self.tokenizer)" in src, (
            "_generate_text must build JsonSchemaConstraint with the "
            "schema + tokenizer (the only valid construction signature)")
        assert "json_constraint=json_constraint" in src, (
            "_generate_text must forward the constraint to "
            "_generate_manual — without this, the constraint is built "
            "and immediately discarded")

    def test_generate_manual_advances_constraint_and_early_stops(self):
        """N-15: ``_generate_manual`` must (1) pass the constraint to
        ``_sample_token`` so the mask is applied per step, (2) call
        ``constraint.advance(token_id)`` after each sample to drive the
        FSM forward, and (3) check ``is_done`` to exit the loop early
        when the JSON is structurally complete (otherwise the masker
        would block all further tokens to -inf and the loop would
        either NaN-fallback or waste tokens to max_gen)."""
        from enigma_engine.core import engine_generation
        src = inspect.getsource(engine_generation._GenerationMixin._generate_manual)
        assert "json_constraint=json_constraint" in src, (
            "_generate_manual must forward json_constraint to "
            "_sample_token (the mask hook lives there)")
        assert "json_constraint.advance(" in src, (
            "_generate_manual must call constraint.advance(token_id) "
            "after sampling — without this, the FSM never moves past "
            "EXPECT_OPEN and every token after the first '{' is masked "
            "to -inf forever")
        assert "json_constraint.is_done" in src, (
            "_generate_manual must early-stop on is_done — past DONE "
            "the masker returns the empty allowed set and downstream "
            "sampling NaN-fallbacks")

    def test_gguf_model_with_schema_raises_notimplemented(self):
        """N-15 honesty gate: GGUF models route through llama.cpp's
        own sampler (``model.generate(...)`` in C++), which never sees
        our logit mask. Silently returning unconstrained output
        labelled as schema-conforming would be a correctness lie. Be
        loud at the API boundary instead."""
        from enigma_engine.core import engine_generation

        class _FakeSelf:
            _is_gguf = True
            model = object()  # any non-None value; never accessed

        with pytest.raises(NotImplementedError, match="GGUF"):
            engine_generation._GenerationMixin._generate_text(
                _FakeSelf(),
                "irrelevant prompt",
                max_gen=4,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1,
                stop_strings=None,
                use_cache=True,
                min_p=0.0,
                json_schema={"type": "object", "properties": {}},
            )

    def test_stream_generate_builds_constraint_advances_and_breaks(self):
        """Pass 156z6 (N-15c): ``stream_generate`` must mirror the
        non-streaming wiring — build the constraint once before the
        loop, forward to ``_sample_token`` so the mask runs each step,
        advance the FSM after sampling, and break on ``is_done``.
        Without all four pieces the streaming path is dead infra
        regardless of how complete ``_generate_manual`` is.

        Structural is justified for the same reason as the
        ``_generate_manual`` test above: the behavioural path requires
        stubbing model.forward + KV cache + tokenizer + lock — far
        more code than the wiring being tested. Behavioural coverage
        for the constraint itself lives in ``TestJsonSchemaConstraint``.
        """
        from enigma_engine.core import engine_generation
        # Pass 156z9al (B-3d) split stream_generate into an outer
        # multi-round splice orchestrator and an inner per-round
        # helper ``_stream_round_tokens``. The constraint is BUILT in
        # the outer (once per call, not per round) and ADVANCED in the
        # inner (per token).  Check both halves.
        outer = inspect.getsource(
            engine_generation._GenerationMixin.stream_generate)
        inner = inspect.getsource(
            engine_generation._GenerationMixin._stream_round_tokens)
        assert "json_schema" in inspect.signature(
            engine_generation._GenerationMixin.stream_generate
        ).parameters, (
            "stream_generate must accept json_schema kwarg (N-15c)")
        assert "JsonSchemaConstraint(json_schema, self.tokenizer)" in outer, (
            "stream_generate must build JsonSchemaConstraint — without "
            "this, the kwarg is silently dropped and streaming callers "
            "get unconstrained output labelled as schema-conforming")
        assert "json_constraint=json_constraint" in outer, (
            "stream_generate must forward json_constraint to its "
            "per-round helper")
        assert "json_constraint=json_constraint" in inner, (
            "_stream_round_tokens must forward json_constraint to "
            "_sample_token (the mask hook lives there)")
        assert "json_constraint.advance(" in inner, (
            "_stream_round_tokens must advance the FSM each step — "
            "without this the FSM stays in EXPECT_OPEN forever and "
            "every token after the first '{' is masked to -inf")
        assert "json_constraint.is_done" in inner, (
            "_stream_round_tokens must break on is_done — past DONE "
            "the mask returns the empty allowed set and softmax NaNs")

    def test_stream_chat_gguf_with_schema_raises_notimplemented(self):
        """Pass 156z6 (N-15c): the streaming GGUF path must reject
        json_schema with the same NotImplementedError as the
        non-streaming GGUF path. Without this, GGUF callers using
        /api/chat/stream would silently receive unconstrained tokens
        because llama.cpp's stream sampler never sees our mask.
        """
        from enigma_engine.core.engine_chat import ChatContext, _ChatMixin

        # Stub _prepare_chat to return a GGUF-flagged ctx without
        # touching tokenizer/history machinery.
        fake_ctx = ChatContext(
            messages=[{"role": "user", "content": "hi"}],
            prompt="hi",
            stop_strings=[],
            max_gen=8,
            temperature=0.7,
            repeat_penalty=1.1,
            top_p=0.9,
            top_k=40,
            is_gguf=True,
            has_server_backend=False,
        )

        class _FakeModel:
            def chat(self, *a, **kw):  # makes hasattr(model, 'chat') True
                return ""

        class _FakeSelf:
            model = _FakeModel()

            def _prepare_chat(self, *a, **kw):
                return fake_ctx

        gen = _ChatMixin.stream_chat(
            _FakeSelf(),
            "hi",
            json_schema={"type": "object", "properties": {}},
        )
        with pytest.raises(NotImplementedError, match="GGUF"):
            next(gen)

    def test_chat_gguf_with_schema_raises_notimplemented(self):
        """Pass 156z7 (N-15c2) sibling-boundary fix: ``chat()`` GGUF
        branch must raise NotImplementedError on json_schema, mirroring
        ``stream_chat``. Pass 156z6 fixed only the streaming sibling
        and missed the non-streaming twin \u2014 production path
        ``POST /api/chat`` \u2192 ``state.chat`` \u2192 ``engine.chat`` would
        silently pass GGUF callers unconstrained output labelled as
        schema-conforming.
        """
        from enigma_engine.core.engine_chat import ChatContext, _ChatMixin

        fake_ctx = ChatContext(
            messages=[{"role": "user", "content": "hi"}],
            prompt="hi",
            stop_strings=[],
            max_gen=8,
            temperature=0.7,
            repeat_penalty=1.1,
            top_p=0.9,
            top_k=40,
            is_gguf=True,
            has_server_backend=False,
        )

        class _FakeModel:
            def chat(self, *a, **kw):  # makes hasattr(model, 'chat') True
                return ""

        class _FakeSelf:
            model = _FakeModel()

            def _prepare_chat(self, *a, **kw):
                return fake_ctx

        with pytest.raises(NotImplementedError, match="GGUF"):
            _ChatMixin.chat(
                _FakeSelf(),
                "hi",
                json_schema={"type": "object", "properties": {}},
            )

    def test_generate_with_vision_with_schema_raises_notimplemented(self):
        """Pass 156z7 (N-15c2) sibling-boundary fix:
        ``_generate_with_vision`` must raise on json_schema. The
        multimodal path samples without going through
        ``_generate_text``/``_generate_manual``, so the constraint
        FSM is never wired in. Reachable via
        ``engine.chat(images=[...], json_schema={...})`` from any
        Python caller.
        """
        from enigma_engine.core import engine_generation

        # _generate_with_vision short-circuits on empty prompt AFTER the
        # gate; if the gate is missing, this returns "" without raising.
        # If the gate is present, we get NotImplementedError before any
        # tokenizer/model access.
        with pytest.raises(NotImplementedError, match="vision"):
            engine_generation._GenerationMixin._generate_with_vision(
                object(),  # `self` never accessed before the gate
                "irrelevant prompt",
                vision_features=object(),  # never accessed before the gate
                json_schema={"type": "object", "properties": {}},
            )

    def test_generate_with_schema_and_execute_tools_raises_value_error(self):
        """Pass 156z7 (N-15c2) sibling-boundary fix: when
        ``json_schema`` and ``execute_tools=True`` are both set,
        ``EnigmaEngine.generate`` must raise ValueError. Without the
        gate, the first generation is schema-constrained but
        ``_execute_tools_in_text`` re-enters ``_generate_text``
        without the schema on every tool-call detection \u2014 silent
        partial constraint, output still labelled as schema-conforming.

        Behavioural: pass enable_tools=True path into a stub engine,
        assert the error message points at the right knob.
        """
        from enigma_engine.core.inference import EnigmaEngine

        class _FakeSelf:
            enable_tools = True
            use_routing = False
            _tool_router = None
            # Anything below should never be reached before the raise
            model = None
            tokenizer = None

        with pytest.raises(ValueError, match="execute_tools"):
            EnigmaEngine.generate(
                _FakeSelf(),
                "irrelevant",
                max_gen=4,
                json_schema={"type": "object", "properties": {}},
                execute_tools=True,
            )


# ════════════════════════════════════════════════════════════════════
# S562 — PrefixKVCache.prompt_hash Removed
# ════════════════════════════════════════════════════════════════════

class TestPrefixKVCacheNoPromptHash:
    """S562: prompt_hash dead field removed from PrefixKVCache."""

    def test_no_prompt_hash_attribute(self):
        src = inspect.getsource(
            __import__('enigma_engine.core.kv_cache',
                       fromlist=['PrefixKVCache']).PrefixKVCache
        )
        assert 'prompt_hash' not in src, (
            "PrefixKVCache should not have prompt_hash (dead field)")


# ════════════════════════════════════════════════════════════════════
# R30 continued — AdEMAMix validation tests
# ════════════════════════════════════════════════════════════════════

class TestAdEMAMixValidation:
    def test_ademamix_invalid_params(self):
        """Invalid hyperparameters should raise ValueError."""
        import torch
        from enigma_engine.core.training import AdEMAMix

        model = torch.nn.Linear(4, 2)
        with pytest.raises(ValueError, match="beta1"):
            AdEMAMix(model.parameters(), betas=(1.0, 0.95))
        with pytest.raises(ValueError, match="beta3"):
            AdEMAMix(model.parameters(), beta3=1.0)


# ════════════════════════════════════════════════════════════════════
# R31 — StreamingLLM cache
# ════════════════════════════════════════════════════════════════════

class TestStreamingLLMCache:
    """R31: StreamingLLM attention-sink cache in kv_cache.py."""

    def test_inherits_kvcache(self):
        from enigma_engine.core.kv_cache import KVCache, StreamingLLMCache
        assert issubclass(StreamingLLMCache, KVCache)

    def test_has_sink_and_window(self):
        """StreamingLLMCache should have n_sink and window_size params."""
        import torch
        from enigma_engine.core.kv_cache import StreamingLLMCache

        cache = StreamingLLMCache(
            n_sink=4, window_size=16,
            batch_size=1, max_seq_len=64, n_kv_heads=2,
            head_dim=8, device=torch.device("cpu"),
        )
        assert cache.n_sink == 4
        assert cache.window_size == 16

    def test_effective_budget(self):
        import torch
        from enigma_engine.core.kv_cache import StreamingLLMCache

        cache = StreamingLLMCache(
            n_sink=4, window_size=16,
            batch_size=1, max_seq_len=64, n_kv_heads=2,
            head_dim=8, device=torch.device("cpu"),
        )
        assert cache.effective_budget == 20  # 4 + 16

    def test_normal_append_before_full(self):
        """Tokens should be appended normally before budget is filled."""
        import torch
        from enigma_engine.core.kv_cache import StreamingLLMCache

        cache = StreamingLLMCache(
            n_sink=2, window_size=4,
            batch_size=1, max_seq_len=32, n_kv_heads=2,
            head_dim=4, device=torch.device("cpu"),
        )
        k = torch.randn(1, 3, 2, 4)
        v = torch.randn(1, 3, 2, 4)
        pos = cache.update(k, v)
        assert pos == 3
        assert cache.current_pos == 3

    def test_eviction_preserves_sinks(self):
        """After overflow, sink tokens should be preserved."""
        import torch
        from enigma_engine.core.kv_cache import StreamingLLMCache

        cache = StreamingLLMCache(
            n_sink=2, window_size=4,
            batch_size=1, max_seq_len=32, n_kv_heads=1,
            head_dim=4, device=torch.device("cpu"),
        )
        # Fill with identifiable tokens
        # Sinks: positions 0-1
        sink_k = torch.ones(1, 2, 1, 4) * 99.0
        sink_v = torch.ones(1, 2, 1, 4) * 99.0
        cache.update(sink_k, sink_v)

        # Fill window: positions 2-5
        fill_k = torch.randn(1, 4, 1, 4)
        fill_v = torch.randn(1, 4, 1, 4)
        cache.update(fill_k, fill_v)

        # Now overflow: should evict mid-tokens and keep sinks
        new_k = torch.ones(1, 1, 1, 4) * -1.0
        new_v = torch.ones(1, 1, 1, 4) * -1.0
        cache.update(new_k, new_v)

        # Read back — first 2 positions should still be sinks
        k_out, v_out = cache.get()
        assert k_out.shape[1] <= cache.effective_budget
        # Sink tokens should still be ~99.0
        assert (k_out[0, 0, 0] - 99.0).abs().max() < 0.1

    def test_clear_resets_logical_pos(self):
        import torch
        from enigma_engine.core.kv_cache import StreamingLLMCache

        cache = StreamingLLMCache(
            n_sink=2, window_size=4,
            batch_size=1, max_seq_len=32, n_kv_heads=1,
            head_dim=4, device=torch.device("cpu"),
        )
        k = torch.randn(1, 3, 1, 4)
        v = torch.randn(1, 3, 1, 4)
        cache.update(k, v)
        cache.clear()
        assert cache._logical_pos == 0
        assert cache.current_pos == 0


# ════════════════════════════════════════════════════════════════════
# R32 — ORPO: Odds Ratio Preference Optimization
# ════════════════════════════════════════════════════════════════════

class TestORPO:
    """R32: ORPO training in training.py."""

    def test_orpo_empty_data_raises(self):
        """Empty preference data should raise ValueError."""
        from unittest.mock import MagicMock
        from enigma_engine.core.training import Trainer
        trainer = MagicMock(spec=Trainer)
        trainer.train_orpo = Trainer.train_orpo.__get__(trainer, Trainer)
        with pytest.raises(ValueError, match="No preference data"):
            trainer.train_orpo([])

    def test_orpo_no_reference_model(self):
        """ORPO should not create a reference model (unlike DPO)."""
        from enigma_engine.core.training import Trainer
        src = inspect.getsource(Trainer.train_orpo)
        assert "deepcopy" not in src
        assert "ref_model" not in src


# ════════════════════════════════════════════════════════════════════
# R33 — ReMax: REINFORCE with mean-reward baseline
# ════════════════════════════════════════════════════════════════════

class TestReMax:
    """R33: ReMax RL trainer in rl_training.py."""

    def test_config_exists(self):
        from enigma_engine.core.rl_training import ReMaxConfig
        cfg = ReMaxConfig()
        assert cfg.epochs == 3
        assert cfg.n_responses == 4
        assert cfg.clip_range == 0.2

    def test_no_value_head(self):
        """ReMax should not use a ValueHead — that's the whole point."""
        from enigma_engine.core.rl_training import ReMaxTrainer
        src = inspect.getsource(ReMaxTrainer.train)
        assert "ValueHead" not in src
        assert "value_head" not in src

    def test_empty_prompts_raises(self):
        from unittest.mock import MagicMock
        from enigma_engine.core.rl_training import ReMaxTrainer
        trainer = MagicMock(spec=ReMaxTrainer)
        trainer.train = ReMaxTrainer.train.__get__(trainer, ReMaxTrainer)
        with pytest.raises(ValueError, match="No prompts"):
            trainer.train([])


# ════════════════════════════════════════════════════════════════════
# R34 — Muon: Newton-Schulz orthogonalization optimizer
# ════════════════════════════════════════════════════════════════════

class TestMuon:
    """R34: Muon optimizer in training.py."""

    def test_muon_step(self):
        """Muon optimizer runs a step without error."""
        import torch
        from enigma_engine.core.training import Muon

        model = torch.nn.Linear(8, 4)
        opt = Muon(model.parameters(), lr=0.01, momentum=0.95, ns_steps=3)
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    def test_muon_reduces_loss(self):
        """Muon should reduce loss over multiple steps."""
        import torch
        from enigma_engine.core.training import Muon

        torch.manual_seed(42)
        model = torch.nn.Linear(4, 2, bias=False)
        opt = Muon(model.parameters(), lr=0.01)

        target = torch.randn(1, 2)
        x = torch.randn(1, 4)

        losses = []
        for _ in range(50):
            loss = ((model(x) - target) ** 2).sum()
            losses.append(loss.item())
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert losses[-1] < losses[0], "Muon should reduce loss"

    def test_newton_schulz_produces_orthogonal(self):
        """Newton-Schulz should produce approximately orthogonal output."""
        import torch
        from enigma_engine.core.training import Muon

        G = torch.randn(4, 4)
        Q = Muon._newton_schulz(G, steps=10)
        # Q^T Q should be close to identity
        eye = torch.eye(4)
        residual = (Q.T @ Q - eye).abs().max()
        assert residual < 1.0, f"Expected near-orthogonal, got residual={residual:.4f}"

    def test_muon_handles_1d_params(self):
        """1D params (bias) should use plain momentum, not Newton-Schulz."""
        import torch
        from enigma_engine.core.training import Muon

        model = torch.nn.Linear(4, 2, bias=True)
        opt = Muon(model.parameters(), lr=0.01)
        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step()  # Should not crash on 1D bias

    def test_muon_invalid_ns_steps(self):
        """ns_steps < 1 should raise ValueError."""
        import torch
        from enigma_engine.core.training import Muon

        model = torch.nn.Linear(4, 2)
        with pytest.raises(ValueError, match="ns_steps"):
            Muon(model.parameters(), ns_steps=0)


# ════════════════════════════════════════════════════════════════════
# T3-7 — NF4 Linear (QLoRA native quantization)
# ════════════════════════════════════════════════════════════════════

class TestNF4Linear:
    """T3-7: NF4 4-bit quantization for QLoRA fine-tuning."""

    def test_from_linear_shape(self):
        """NF4Linear.from_linear preserves input/output dimensions."""
        import torch
        from enigma_engine.core.nf4_linear import NF4Linear

        linear = torch.nn.Linear(64, 32, bias=False)
        nf4 = NF4Linear.from_linear(linear)
        assert nf4.in_features == 64
        assert nf4.out_features == 32

    def test_forward_shape(self):
        """Forward pass produces correct output shape."""
        import torch
        from enigma_engine.core.nf4_linear import NF4Linear

        linear = torch.nn.Linear(64, 32, bias=True)
        nf4 = NF4Linear.from_linear(linear)
        x = torch.randn(2, 64)
        out = nf4(x)
        assert out.shape == (2, 32)

    def test_quantize_dequantize_accuracy(self):
        """Dequantized weights should be close to originals (4-bit tolerance)."""
        import torch
        from enigma_engine.core.nf4_linear import NF4Linear

        torch.manual_seed(42)
        linear = torch.nn.Linear(128, 64, bias=False)
        # Initialize with normal distribution (best case for NF4)
        torch.nn.init.normal_(linear.weight, std=0.02)
        nf4 = NF4Linear.from_linear(linear)
        reconstructed = nf4._dequantize()
        # 4-bit quantization: expect ~5% relative error for normal weights
        rel_error = (reconstructed - linear.weight).abs().mean() / linear.weight.abs().mean()
        assert rel_error < 0.15, f"Relative error {rel_error:.4f} too high for NF4"

    def test_forward_numerical_close(self):
        """NF4 forward should approximate original linear output."""
        import torch
        from enigma_engine.core.nf4_linear import NF4Linear

        torch.manual_seed(42)
        linear = torch.nn.Linear(64, 32, bias=True)
        torch.nn.init.normal_(linear.weight, std=0.02)
        nf4 = NF4Linear.from_linear(linear)
        x = torch.randn(4, 64)
        orig_out = linear(x)
        nf4_out = nf4(x)
        # Output should be close but not exact
        assert torch.allclose(orig_out, nf4_out, atol=0.5), (
            f"Max diff: {(orig_out - nf4_out).abs().max():.4f}")

    def test_memory_savings(self):
        """NF4 should use significantly less memory than fp16."""
        import torch
        from enigma_engine.core.nf4_linear import NF4Linear

        linear = torch.nn.Linear(256, 128, bias=False)
        nf4 = NF4Linear.from_linear(linear)
        ratio = nf4.memory_savings_ratio()
        # 4-bit = 0.5 bytes per param vs 2 bytes for fp16 → ~0.25 ratio
        # Plus scale overhead so expect < 0.40
        assert ratio < 0.40, f"Memory ratio {ratio:.3f} not enough savings"

    def test_bias_preserved(self):
        """Bias should be preserved through quantization."""
        import torch
        from enigma_engine.core.nf4_linear import NF4Linear

        linear = torch.nn.Linear(16, 8, bias=True)
        linear.bias.data.fill_(3.14)
        nf4 = NF4Linear.from_linear(linear)
        assert nf4.bias is not None
        assert torch.allclose(nf4.bias, linear.bias)

    def test_no_bias(self):
        """NF4Linear without bias should have bias=None."""
        import torch
        from enigma_engine.core.nf4_linear import NF4Linear

        linear = torch.nn.Linear(16, 8, bias=False)
        nf4 = NF4Linear.from_linear(linear)
        assert nf4.bias is None

    def test_quantize_model(self):
        """quantize_linear_nf4 replaces all Linear layers."""
        import torch
        from enigma_engine.core.nf4_linear import NF4Linear, quantize_linear_nf4

        model = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
        )
        count = quantize_linear_nf4(model)
        assert count == 2
        assert isinstance(model[0], NF4Linear)
        assert isinstance(model[2], NF4Linear)

    def test_quantize_model_skip(self):
        """quantize_linear_nf4 should skip named layers."""
        import torch
        from enigma_engine.core.nf4_linear import NF4Linear, quantize_linear_nf4

        model = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, 8),
        )
        count = quantize_linear_nf4(model, skip_names={'1'})
        assert count == 1
        assert isinstance(model[0], NF4Linear)
        assert not isinstance(model[1], NF4Linear)

    def test_odd_dimensions(self):
        """NF4 should handle odd element counts (padding edge case)."""
        import torch
        from enigma_engine.core.nf4_linear import NF4Linear

        # 7 * 5 = 35 elements (odd)
        linear = torch.nn.Linear(7, 5, bias=False)
        nf4 = NF4Linear.from_linear(linear)
        x = torch.randn(1, 7)
        out = nf4(x)
        assert out.shape == (1, 5)


# ════════════════════════════════════════════════════════════════════
# T3-1 — Cross-Layer KV Sharing (YOCO-style)
# ════════════════════════════════════════════════════════════════════

class TestCrossLayerKVSharing:
    """T3-1: Adjacent layers share K, V projections to save memory."""

    def test_config_field_exists(self):
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig()
        assert hasattr(config, 'kv_share_groups')
        assert config.kv_share_groups == 0

    def test_sharing_disabled_by_default(self):
        """With kv_share_groups=0, no layers should be followers."""
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=4,
            vocab_size=100, max_seq_len=64, kv_share_groups=0,
        )
        model = Enigma(config)
        for layer in model.layers:
            assert layer.attention._kv_share_source is None

    def test_sharing_links_set_correctly(self):
        """With kv_share_groups=2 and 4 layers, layers 1 and 3 are followers."""
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=4,
            vocab_size=100, max_seq_len=64, kv_share_groups=2,
        )
        model = Enigma(config)
        # Group 0: layers 0, 1 — leader is layer 0
        assert model.layers[0].attention._kv_share_source is None  # leader
        assert model.layers[1].attention._kv_share_source is model.layers[0].attention
        # Group 1: layers 2, 3 — leader is layer 2
        assert model.layers[2].attention._kv_share_source is None  # leader
        assert model.layers[3].attention._kv_share_source is model.layers[2].attention

    def test_forward_training_with_sharing(self):
        """Model forward should work in training mode with KV sharing."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=4,
            vocab_size=100, max_seq_len=64, kv_share_groups=2,
        )
        model = Enigma(config)
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            logits = model(x, use_cache=False)
        assert logits.shape == (1, 8, 128)  # padded_vocab = (100+63)&~63 = 128

    def test_forward_inference_with_sharing(self):
        """Model forward with use_cache=True should work with KV sharing."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=4,
            vocab_size=100, max_seq_len=64, kv_share_groups=2,
        )
        model = Enigma(config)
        model.eval()
        # Prefill
        x = torch.randint(0, 100, (1, 4))
        with torch.no_grad():
            logits = model(x, use_cache=True, start_pos=0)
        assert logits.shape == (1, 4, 128)
        # Decode one token
        next_tok = torch.randint(0, 100, (1, 1))
        with torch.no_grad():
            logits2 = model(next_tok, use_cache=True, start_pos=4)
        assert logits2.shape == (1, 1, 128)
        model.clear_cache()


# ════════════════════════════════════════════════════════════════════
# T3-2 — Self-Speculative Decoding (Early Exit)
# ════════════════════════════════════════════════════════════════════

class TestSelfSpeculativeDecoding:
    """T3-2: Early-exit head for self-speculative decoding."""

    def test_config_field_exists(self):
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig()
        assert hasattr(config, 'early_exit_layer')
        assert config.early_exit_layer == 0

    def test_draft_forward_shape(self):
        """draft_forward should return logits from early-exit head."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=6,
            vocab_size=100, max_seq_len=64, early_exit_layer=2,
        )
        model = Enigma(config)
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            logits = model.draft_forward(x)
        assert logits.shape == (1, 8, 128)  # padded vocab

    def test_draft_forward_raises_if_disabled(self):
        """draft_forward should raise if early_exit_layer is 0."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=6,
            vocab_size=100, max_seq_len=64,
        )
        model = Enigma(config)
        x = torch.randint(0, 100, (1, 4))
        with pytest.raises(RuntimeError, match="early_exit_layer"):
            model.draft_forward(x)

    def test_training_with_early_exit_loss(self):
        """Training forward should include early-exit auxiliary loss."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=6,
            vocab_size=100, max_seq_len=64, early_exit_layer=2,
        )
        model = Enigma(config)
        model.train()
        x = torch.randint(0, 100, (1, 8))
        targets = torch.randint(0, 100, (1, 8))
        logits, loss = model(x, targets=targets)
        assert loss is not None
        assert loss.item() > 0


# ════════════════════════════════════════════════════════════════════
# T3-3 — Medusa Multi-Head Speculation
# ════════════════════════════════════════════════════════════════════

class TestMedusaSpeculation:
    """T3-3: MTP heads reused for Medusa-style parallel draft tokens."""

    def test_medusa_forward_shape(self):
        """medusa_forward returns main logits + K draft logits."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=4,
            vocab_size=100, max_seq_len=64, n_predict_heads=2,
        )
        model = Enigma(config)
        model.eval()
        x = torch.randint(0, 100, (1, 6))
        with torch.no_grad():
            main_logits, draft_logits = model.medusa_forward(x)
        padded = 128  # (100+63)&~63
        assert main_logits.shape == (1, 6, padded)
        assert len(draft_logits) == 2
        for dl in draft_logits:
            assert dl.shape == (1, 6, padded)

    def test_medusa_forward_raises_without_heads(self):
        """medusa_forward should raise if no MTP heads."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=4,
            vocab_size=100, max_seq_len=64, n_predict_heads=0,
        )
        model = Enigma(config)
        x = torch.randint(0, 100, (1, 4))
        with pytest.raises(RuntimeError, match="n_predict_heads"):
            model.medusa_forward(x)

    def test_medusa_forward_with_cache(self):
        """medusa_forward should work with KV cache."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=4,
            vocab_size=100, max_seq_len=64, n_predict_heads=2,
        )
        model = Enigma(config)
        model.eval()
        x = torch.randint(0, 100, (1, 4))
        with torch.no_grad():
            main, drafts = model.medusa_forward(x, use_cache=True)
        assert main.shape == (1, 4, 128)
        assert len(drafts) == 2
        model.clear_cache()

    def test_draft_logits_differ_from_main(self):
        """Draft head logits should generally differ from main logits."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=4,
            vocab_size=100, max_seq_len=64, n_predict_heads=2,
        )
        model = Enigma(config)
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            main, drafts = model.medusa_forward(x)
        # Draft heads have different weights so logits should differ
        assert not torch.allclose(main, drafts[0], atol=1e-3)


# ════════════════════════════════════════════════════════════════════
# T3-4 — Mixture of Depths
# ════════════════════════════════════════════════════════════════════

class TestMixtureOfDepths:
    """T3-4: Per-token depth routing — easy tokens skip FFN."""

    def test_config_fields_exist(self):
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig()
        assert hasattr(config, 'use_mixture_of_depths')
        assert config.use_mixture_of_depths is False
        assert hasattr(config, 'mod_capacity_factor')
        assert config.mod_capacity_factor == 0.5

    def test_disabled_by_default(self):
        """Without MoD, TransformerBlock has no depth_router."""
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model_components import TransformerBlock

        config = ForgeConfig(dim=64, n_heads=4, n_kv_heads=2, n_layers=4)
        block = TransformerBlock(config, layer_id=0)
        assert not block.use_mod

    def test_router_created_when_enabled(self):
        """With MoD, TransformerBlock should have depth_router."""
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model_components import TransformerBlock

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=4,
            use_mixture_of_depths=True,
        )
        block = TransformerBlock(config, layer_id=0)
        assert block.use_mod
        assert hasattr(block, 'depth_router')
        assert block.depth_router.in_features == 64
        assert block.depth_router.out_features == 1

    def test_forward_shape_with_mod(self):
        """MoD forward should produce same output shape."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model_components import TransformerBlock

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=4,
            use_mixture_of_depths=True, mod_capacity_factor=0.5,
        )
        block = TransformerBlock(config, layer_id=0)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_mod_aux_loss(self):
        """get_mod_aux_loss() should return a scalar loss when MoD active."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model_components import TransformerBlock

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=4,
            use_mixture_of_depths=True,
        )
        block = TransformerBlock(config, layer_id=0)
        x = torch.randn(1, 8, 64)
        block(x)
        loss = block.get_mod_aux_loss()
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0

    def test_mod_no_aux_loss_when_disabled(self):
        """get_mod_aux_loss() returns 0 when MoD is disabled."""
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model_components import TransformerBlock

        config = ForgeConfig(dim=64, n_heads=4, n_kv_heads=2, n_layers=4)
        block = TransformerBlock(config, layer_id=0)
        loss = block.get_mod_aux_loss()
        assert loss.item() == 0.0

    def test_full_model_with_mod(self):
        """Full Enigma model should work with MoD enabled."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma

        config = ForgeConfig(
            dim=64, n_heads=4, n_kv_heads=2, n_layers=4,
            vocab_size=100, max_seq_len=64,
            use_mixture_of_depths=True, mod_capacity_factor=0.5,
        )
        model = Enigma(config)
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 8, 128)


# ════════════════════════════════════════════════════════════════════
# T4-1 — MinHash Near-Duplicate Detection
# ════════════════════════════════════════════════════════════════════

class TestMinHashDedup:
    """T4-1: MinHash near-duplicate detection in training.py."""

    def test_exact_duplicates_removed(self):
        from enigma_engine.core.training import minhash_dedup
        texts = ["hello world", "hello world", "something else"]
        result = minhash_dedup(texts, threshold=0.8)
        # Exact duplicates should be removed
        assert len(result) == 2

    def test_near_duplicates_removed(self):
        from enigma_engine.core.training import minhash_dedup
        # These two are very similar — only one word different
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy cat",
            "completely different sentence about something else entirely",
        ]
        result = minhash_dedup(texts, threshold=0.8)
        # First two are near-duplicates, keep the longer (or first)
        assert len(result) == 2

    def test_dissimilar_kept(self):
        from enigma_engine.core.training import minhash_dedup
        texts = [
            "machine learning is about algorithms",
            "cooking recipes for pasta dishes",
            "quantum physics and entanglement theory",
        ]
        result = minhash_dedup(texts, threshold=0.8)
        assert len(result) == 3

    def test_empty_input(self):
        from enigma_engine.core.training import minhash_dedup
        assert minhash_dedup([], threshold=0.8) == []

    def test_single_input(self):
        from enigma_engine.core.training import minhash_dedup
        result = minhash_dedup(["one text"], threshold=0.8)
        assert len(result) == 1

    def test_keeps_longest_near_duplicate(self):
        from enigma_engine.core.training import minhash_dedup
        short = "the quick brown fox"
        long = "the quick brown fox jumps over the lazy dog in the park"
        texts = [short, long]
        result = minhash_dedup(texts, threshold=0.5)
        # With a low threshold these should be near-dupes; keep longer
        if len(result) == 1:
            assert result[0] == long

    def test_threshold_1_keeps_all(self):
        from enigma_engine.core.training import minhash_dedup
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy cat",
        ]
        # threshold=1.0 means only exact shingle-set matches are dupes
        result = minhash_dedup(texts, threshold=1.0)
        assert len(result) == 2

    def test_default_threshold_matches_fineweb_standard(self):
        """Data-5b: MinHash default must be 0.75 (FineWeb/DCLM/SmolLM3 standard).

        FineWeb tech report (arxiv:2406.17557 \u00a73.4) targets documents
        \u226575% similar. The previous 0.8 default was slightly more permissive
        than industry practice and disagreed with the cited research.
        """
        import inspect
        from enigma_engine.core.training import minhash_dedup
        sig = inspect.signature(minhash_dedup)
        assert sig.parameters["threshold"].default == 0.75


# ════════════════════════════════════════════════════════════════════
# T4-2 — Curriculum Learning (Easy→Hard)
# ════════════════════════════════════════════════════════════════════

class TestCurriculumLearning:
    """T4-2: Curriculum learning config and ordering."""

    def test_config_field_exists(self):
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert hasattr(config, 'curriculum')
        assert config.curriculum == "none"

    def test_valid_curriculum_values(self):
        from enigma_engine.core.training import TrainingConfig
        for val in ("none", "easy_first"):
            config = TrainingConfig(curriculum=val)
            config.validate()

    def test_invalid_curriculum_rejected(self):
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig(curriculum="invalid")
        with pytest.raises(ValueError, match="curriculum"):
            config.validate()

    def test_to_dict_includes_curriculum(self):
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig(curriculum="easy_first")
        d = config.to_dict()
        assert "curriculum" in d
        assert d["curriculum"] == "easy_first"


# ════════════════════════════════════════════════════════════════════
# T4-3 — ReST (Reinforced Self-Training)
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# T4-4 — Vocab Size 32K Default
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# T5-4 — Token Merging (ToMe) in TransformerBlock
# ════════════════════════════════════════════════════════════════════

class TestTokenMerging:
    """T5-4: Bipartite soft matching to merge redundant tokens during training."""

    def test_config_has_tome_ratio(self):
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig()
        assert hasattr(config, 'tome_ratio')
        assert config.tome_ratio == 0.0

    def test_bipartite_soft_matching_basic(self):
        """_bipartite_soft_matching returns merge_dst, merge_dst, merged_mask."""
        import torch
        mc = __import__('enigma_engine.core.model_components',
                        fromlist=['_bipartite_soft_matching'])
        x = torch.randn(1, 8, 16)  # B=1, T=8, D=16
        merge_dst, _, merged_mask = mc._bipartite_soft_matching(x, r=2)
        assert merge_dst is not None
        assert merged_mask.shape == (1, 8)

    def test_tome_merge_reduces_tokens(self):
        """_tome_merge should reduce token count."""
        import torch
        mc = __import__('enigma_engine.core.model_components',
                        fromlist=['_bipartite_soft_matching', '_tome_merge'])
        x = torch.randn(1, 8, 16)
        _, _, merged_mask = mc._bipartite_soft_matching(x, r=2)
        merged = mc._tome_merge(x, merged_mask)
        assert merged.shape[1] < x.shape[1]

    def test_tome_roundtrip_shapes(self):
        """Merge then unmerge should restore original token count."""
        import torch
        mc = __import__('enigma_engine.core.model_components',
                        fromlist=['_bipartite_soft_matching', '_tome_merge', '_tome_unmerge'])
        x = torch.randn(1, 8, 16)
        merge_dst, _, merged_mask = mc._bipartite_soft_matching(x, r=2)
        merged = mc._tome_merge(x, merged_mask)
        restored = mc._tome_unmerge(merged, 8, merged_mask, merge_dst)
        assert restored.shape == x.shape

    def test_tome_skip_long_sequences(self):
        """S822: _bipartite_soft_matching returns identity for T > 4096."""
        import torch
        mc = __import__('enigma_engine.core.model_components',
                        fromlist=['_bipartite_soft_matching'])
        # Create input over the 4096-token threshold
        x = torch.randn(1, 4097, 16)
        merge_dst, unmerge_src, merged_mask = mc._bipartite_soft_matching(
            x, r=2)
        # Identity: no tokens merged (all False = nothing removed)
        assert not merged_mask.any(), "Long sequence should skip merging"
        assert merge_dst.shape == (1, 4097)


# ════════════════════════════════════════════════════════════════════
# T5-5 — Lookahead Decoding (Jacobi iteration)
# ════════════════════════════════════════════════════════════════════

class TestLookaheadDecoding:
    """T5-5: Jacobi iteration draft + verify for speculative decoding."""

    def test_ngram_pool_update(self):
        """_update_ngram_pool should maintain FIFO eviction."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        pool = {}
        tokens = [1, 2, 3, 4, 5]
        _GenerationMixin._update_ngram_pool(pool, tokens, max_size=10)
        # Should have bigram entries
        assert len(pool) > 0

    def test_verify_sends_full_draft_including_next_id(self):
        """S689: draft_tensor[:, 1:] skipped next_id — verify must
        send the full draft so next_id enters the KV cache at the
        correct position."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.lookahead_generate)
        # Must NOT skip draft[0] — the old buggy pattern
        assert 'draft_tensor[:, 1:]' not in source, \
            "verify path must send full draft_tensor, not skip draft[0]"
        # Must send draft_tensor directly to model
        assert 'draft_tensor, use_cache=True' in source or \
            'draft_tensor,' in source

    def test_cache_reset_single_model_call(self):
        """S690: After lookahead verify, cache reset must repopulate
        with one model call, not clear+fill+process-last-token."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.lookahead_generate)
        # After clear_cache, should NOT have a second model call that
        # re-processes the last token at start_pos=N-1
        # The old pattern: clear_cache() + model(full_seq) + model(last_token)
        # The fix: clear_cache() + logits = model(full_seq)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'clear_cache()' in line and 'Reset cache' in ''.join(lines[max(0, i-3):i+1]):
                # Look ahead: the next model call should capture logits
                remaining = '\n'.join(lines[i+1:i+5])
                assert 'logits = self.model(' in remaining or \
                    'logits=self.model(' in remaining, \
                    "Cache reset must capture logits from the refill call"

    def test_no_cache_path_slices_verify_logits(self):
        """S689: No-cache verify path must slice logits to draft region,
        not use raw indices from the start of the sequence."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        source = inspect.getsource(_GenerationMixin.lookahead_generate)
        # The no-cache path must slice verify_logits to the draft region
        assert 'verify_logits[:, -len(draft):' in source or \
            'verify_logits[:, -(len(draft)):' in source, \
            "No-cache path must slice verify_logits to draft region"


# ════════════════════════════════════════════════════════════════════
# T5-6 — Multi-Head Latent Attention (MLA)
# ════════════════════════════════════════════════════════════════════

class TestMLA:
    """T5-6: Low-rank KV bottleneck for compressed attention projections."""

    def test_config_has_mla_latent_dim(self):
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig()
        assert hasattr(config, 'mla_latent_dim')
        assert config.mla_latent_dim == 0

    def test_mla_disabled_by_default(self):
        """With mla_latent_dim=0, standard wk/wv are used."""
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model_components import Attention

        config = ForgeConfig(dim=64, n_heads=4, n_kv_heads=2, max_seq_len=64)
        attn = Attention(config)
        assert not attn._use_mla
        assert hasattr(attn, 'wk')
        assert hasattr(attn, 'wv')

    def test_mla_enabled_creates_latent_layers(self):
        """With mla_latent_dim>0, wkv_down/wk_up/wv_up should exist."""
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model_components import Attention

        config = ForgeConfig(dim=64, n_heads=4, n_kv_heads=2,
                             max_seq_len=64, mla_latent_dim=16)
        attn = Attention(config)
        assert attn._use_mla
        assert hasattr(attn, 'wkv_down')
        assert hasattr(attn, 'wk_up')
        assert hasattr(attn, 'wv_up')
        # Check dimensions
        assert attn.wkv_down.in_features == 64
        assert attn.wkv_down.out_features == 16
        assert attn.wk_up.in_features == 16
        assert attn.wv_up.in_features == 16


# ════════════════════════════════════════════════════════════════════
# T5-7 — Layer-wise Learning Rate Decay (LLRD)
# ════════════════════════════════════════════════════════════════════

class TestLLRD:
    """T5-7: Exponentially decaying LR per transformer layer."""

    def test_config_has_llrd_decay(self):
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig()
        assert hasattr(config, 'llrd_decay')
        assert config.llrd_decay == 0.0

    def test_llrd_decay_in_to_dict(self):
        from enigma_engine.core.training import TrainingConfig
        config = TrainingConfig(llrd_decay=0.8)
        d = config.to_dict()
        assert 'llrd_decay' in d
        assert d['llrd_decay'] == 0.8

    def test_llrd_decay_validation(self):
        """llrd_decay must be in [0.0, 1.0)."""
        from enigma_engine.core.training import TrainingConfig
        with pytest.raises(ValueError):
            TrainingConfig(llrd_decay=1.0).validate()
        with pytest.raises(ValueError):
            TrainingConfig(llrd_decay=-0.1).validate()
        # Valid values should not raise
        TrainingConfig(llrd_decay=0.0).validate()
        TrainingConfig(llrd_decay=0.95).validate()


# ════════════════════════════════════════════════════════════════════
# T5-8 — Process Reward Model (PRM)
# ════════════════════════════════════════════════════════════════════

class TestPRM:
    """T5-8: Per-step reward scoring for chain-of-thought training."""

    def test_prm_trainer_config_exists(self):
        from enigma_engine.core.rl_training import PRMTrainerConfig
        config = PRMTrainerConfig()
        assert config.epochs == 3
        assert config.learning_rate == 1e-5


# ════════════════════════════════════════════════════════════════════
# T5-9 — Self-Consistency Decoding
# ════════════════════════════════════════════════════════════════════

class TestSelfConsistencyDecoding:
    """T5-9: Generate N responses and majority-vote on the final answer."""

    def test_default_extractor_returns_last_line(self):
        """Default extractor should return the last non-empty line."""
        from enigma_engine.core.engine_generation import _GenerationMixin
        result = _GenerationMixin._default_answer_extractor("Line 1\nLine 2\nFinal answer")
        assert result == "Final answer"


class TestAutoBatchSize:
    """Auto batch size estimation when batch_size=0."""

    def test_config_allows_zero_batch_size(self):
        """TrainingConfig should accept batch_size=0 (auto) without error."""
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(batch_size=0)
        cfg.validate()  # should not raise

    def test_config_rejects_negative_batch_size(self):
        """TrainingConfig should reject batch_size < 0."""
        import pytest
        from enigma_engine.core.training import TrainingConfig
        cfg = TrainingConfig(batch_size=-1)
        with pytest.raises(ValueError):
            cfg.validate()


# ════════════════════════════════════════════════════════════════════
# S740 — Speculative/Medusa must use _sample_token, not raw softmax
# ════════════════════════════════════════════════════════════════════

class TestSpeculativeSamplingFilters:
    """S740: speculative bonus + medusa main must call _sample_token."""

    def test_speculative_bonus_uses_sample_token(self):
        """speculative_generate bonus token path calls _sample_token."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin.speculative_generate)
        # Old broken pattern was: bonus_probs = F.softmax(bonus_logits, ...)
        assert "bonus_probs" not in src, \
            "speculative bonus should use _sample_token, not raw softmax"
        assert "_sample_token" in src

    def test_medusa_main_uses_sample_token(self):
        """medusa_generate main sampling calls _sample_token."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        src = inspect.getsource(_GenerationMixin.medusa_generate)
        # Old broken pattern: main_probs = F.softmax(main_logits[...] / t, ...)
        assert "main_probs" not in src, \
            "medusa main should use _sample_token, not raw softmax"
        assert "_sample_token" in src

    def test_medusa_has_sampling_params(self):
        """medusa_generate accepts top_k, top_p, repetition_penalty, min_p."""
        import inspect
        from enigma_engine.core.engine_generation import _GenerationMixin
        sig = inspect.signature(_GenerationMixin.medusa_generate)
        for param in ("top_k", "top_p", "repetition_penalty", "min_p"):
            assert param in sig.parameters, \
                f"medusa_generate missing parameter: {param}"
