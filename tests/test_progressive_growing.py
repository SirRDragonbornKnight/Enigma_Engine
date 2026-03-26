"""Tests for progressive model growing (Phase 7).

Tests the weight expansion logic that lets a trained smaller model
grow into a larger architecture while preserving learned weights.
"""
import inspect
import math
import unittest
from unittest.mock import patch

import torch

from enigma_engine.core.model_presets import ForgeConfig


class TestExpandModelWeights(unittest.TestCase):
    """Test expand_model_weights() core logic."""

    def _make_config(self, dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
                     max_seq_len=128, vocab_size=256, use_swiglu=True,
                     use_bias=False, use_moe=False):
        return ForgeConfig(
            dim=dim, n_layers=n_layers, n_heads=n_heads,
            n_kv_heads=n_kv_heads, max_seq_len=max_seq_len,
            vocab_size=vocab_size, use_swiglu=use_swiglu,
            use_bias=use_bias, use_moe=use_moe,
        )

    def _make_state_dict(self, config):
        """Create a realistic state dict matching Enigma model structure."""
        sd = {}
        padded_vocab = (config.vocab_size + 63) & ~63
        head_dim = config.dim // config.n_heads

        # Embeddings (padded vocab)
        sd['tok_embeddings.weight'] = torch.randn(padded_vocab, config.dim)
        # Output is tied to embeddings, but appears in state dict
        sd['output.weight'] = sd['tok_embeddings.weight']

        # Final norm
        sd['norm.weight'] = torch.ones(config.dim)

        # Per-layer weights
        for i in range(config.n_layers):
            p = f'layers.{i}'
            sd[f'{p}.attention_norm.weight'] = torch.ones(config.dim)
            sd[f'{p}.ffn_norm.weight'] = torch.ones(config.dim)

            # Attention projections
            sd[f'{p}.attention.wq.weight'] = torch.randn(
                config.n_heads * head_dim, config.dim)
            sd[f'{p}.attention.wk.weight'] = torch.randn(
                config.n_kv_heads * head_dim, config.dim)
            sd[f'{p}.attention.wv.weight'] = torch.randn(
                config.n_kv_heads * head_dim, config.dim)
            sd[f'{p}.attention.wo.weight'] = torch.randn(
                config.dim, config.n_heads * head_dim)

            if config.use_bias:
                sd[f'{p}.attention.wq.bias'] = torch.randn(
                    config.n_heads * head_dim)
                sd[f'{p}.attention.wk.bias'] = torch.randn(
                    config.n_kv_heads * head_dim)
                sd[f'{p}.attention.wv.bias'] = torch.randn(
                    config.n_kv_heads * head_dim)
                sd[f'{p}.attention.wo.bias'] = torch.randn(config.dim)

            # FFN
            if config.use_swiglu:
                sd[f'{p}.feed_forward.w1.weight'] = torch.randn(
                    config.hidden_dim, config.dim)
                sd[f'{p}.feed_forward.w2.weight'] = torch.randn(
                    config.dim, config.hidden_dim)
                sd[f'{p}.feed_forward.w3.weight'] = torch.randn(
                    config.hidden_dim, config.dim)
            else:
                sd[f'{p}.feed_forward.up.weight'] = torch.randn(
                    config.hidden_dim, config.dim)
                sd[f'{p}.feed_forward.down.weight'] = torch.randn(
                    config.dim, config.hidden_dim)

        return sd

    def test_width_expansion_shapes(self):
        """Expanding dim produces correct tensor shapes."""
        from enigma_engine.core.progressive_growing import expand_model_weights

        src_cfg = self._make_config(dim=64, n_layers=2, n_heads=4, n_kv_heads=2)
        tgt_cfg = self._make_config(dim=128, n_layers=2, n_heads=8, n_kv_heads=4)
        src_sd = self._make_state_dict(src_cfg)

        new_sd = expand_model_weights(src_sd, src_cfg, tgt_cfg)

        padded_vocab = (tgt_cfg.vocab_size + 63) & ~63
        tgt_head_dim = tgt_cfg.dim // tgt_cfg.n_heads

        # Check global tensors
        self.assertEqual(new_sd['tok_embeddings.weight'].shape,
                         (padded_vocab, tgt_cfg.dim))
        self.assertEqual(new_sd['norm.weight'].shape,
                         (tgt_cfg.dim,))

        # Check per-layer tensors
        for i in range(tgt_cfg.n_layers):
            p = f'layers.{i}'
            self.assertEqual(
                new_sd[f'{p}.attention.wq.weight'].shape,
                (tgt_cfg.n_heads * tgt_head_dim, tgt_cfg.dim))
            self.assertEqual(
                new_sd[f'{p}.attention.wk.weight'].shape,
                (tgt_cfg.n_kv_heads * tgt_head_dim, tgt_cfg.dim))
            self.assertEqual(
                new_sd[f'{p}.attention.wo.weight'].shape,
                (tgt_cfg.dim, tgt_cfg.n_heads * tgt_head_dim))
            self.assertEqual(
                new_sd[f'{p}.feed_forward.w1.weight'].shape,
                (tgt_cfg.hidden_dim, tgt_cfg.dim))
            self.assertEqual(
                new_sd[f'{p}.feed_forward.w2.weight'].shape,
                (tgt_cfg.dim, tgt_cfg.hidden_dim))
            self.assertEqual(
                new_sd[f'{p}.feed_forward.w3.weight'].shape,
                (tgt_cfg.hidden_dim, tgt_cfg.dim))

    def test_width_expansion_preserves_old_weights(self):
        """Old weight values are preserved in expanded tensors."""
        from enigma_engine.core.progressive_growing import expand_model_weights

        src_cfg = self._make_config(dim=64, n_layers=2, n_heads=4, n_kv_heads=2)
        tgt_cfg = self._make_config(dim=128, n_layers=2, n_heads=4, n_kv_heads=2)
        src_sd = self._make_state_dict(src_cfg)

        new_sd = expand_model_weights(src_sd, src_cfg, tgt_cfg)

        # Embedding: old rows and columns preserved
        old_emb = src_sd['tok_embeddings.weight']
        new_emb = new_sd['tok_embeddings.weight']
        src_vocab = (src_cfg.vocab_size + 63) & ~63
        torch.testing.assert_close(
            new_emb[:src_vocab, :src_cfg.dim], old_emb)

        # Norm: old dimensions preserved
        torch.testing.assert_close(
            new_sd['norm.weight'][:src_cfg.dim],
            src_sd['norm.weight'])

    def test_depth_expansion_shapes(self):
        """Adding layers produces correct number of layer entries."""
        from enigma_engine.core.progressive_growing import expand_model_weights

        src_cfg = self._make_config(dim=64, n_layers=2, n_heads=4, n_kv_heads=2)
        tgt_cfg = self._make_config(dim=64, n_layers=4, n_heads=4, n_kv_heads=2)
        src_sd = self._make_state_dict(src_cfg)

        new_sd = expand_model_weights(src_sd, src_cfg, tgt_cfg)

        # Should have all 4 layers
        for i in range(4):
            self.assertIn(f'layers.{i}.attention.wq.weight', new_sd)
            self.assertIn(f'layers.{i}.feed_forward.w1.weight', new_sd)

        # Should NOT have layer 4
        self.assertNotIn('layers.4.attention.wq.weight', new_sd)

    def test_depth_expansion_layer_mapping(self):
        """Old layers are spread evenly into new positions."""
        from enigma_engine.core.progressive_growing import (
            expand_model_weights, compute_layer_mapping)

        # 2 old layers → 4 new: mapping should spread them
        mapping = compute_layer_mapping(2, 4)
        self.assertEqual(len(mapping), 4)
        # Each new layer maps to a source (-1 means identity)
        for src_idx in mapping:
            self.assertIn(src_idx, {0, 1, -1})

    def test_combined_expansion(self):
        """Both width + depth expansion together."""
        from enigma_engine.core.progressive_growing import expand_model_weights

        src_cfg = self._make_config(dim=64, n_layers=2, n_heads=4, n_kv_heads=2)
        tgt_cfg = self._make_config(dim=128, n_layers=4, n_heads=8, n_kv_heads=4)
        src_sd = self._make_state_dict(src_cfg)

        new_sd = expand_model_weights(src_sd, src_cfg, tgt_cfg)

        tgt_head_dim = tgt_cfg.dim // tgt_cfg.n_heads
        # Verify shapes for all 4 layers
        for i in range(4):
            p = f'layers.{i}'
            self.assertEqual(
                new_sd[f'{p}.attention.wq.weight'].shape,
                (tgt_cfg.n_heads * tgt_head_dim, tgt_cfg.dim))

    def test_reject_smaller_target(self):
        """Cannot shrink a model — target must be >= source."""
        from enigma_engine.core.progressive_growing import expand_model_weights

        src_cfg = self._make_config(dim=128, n_layers=4)
        tgt_cfg = self._make_config(dim=64, n_layers=2)
        src_sd = self._make_state_dict(src_cfg)

        with self.assertRaises(ValueError):
            expand_model_weights(src_sd, src_cfg, tgt_cfg)

    def test_reject_mismatched_architecture(self):
        """Cannot expand across different architecture types."""
        from enigma_engine.core.progressive_growing import expand_model_weights

        src_cfg = self._make_config(use_swiglu=True)
        tgt_cfg = self._make_config(dim=128, n_layers=4, use_swiglu=False)
        src_sd = self._make_state_dict(src_cfg)

        with self.assertRaises(ValueError):
            expand_model_weights(src_sd, src_cfg, tgt_cfg)

    def test_reject_moe_mismatch(self):
        """Cannot grow from non-MoE to MoE or vice versa."""
        from enigma_engine.core.progressive_growing import expand_model_weights

        src_cfg = self._make_config(use_moe=False)
        tgt_cfg = self._make_config(dim=128, n_layers=4, use_moe=True)
        src_sd = self._make_state_dict(src_cfg)

        with self.assertRaises(ValueError):
            expand_model_weights(src_sd, src_cfg, tgt_cfg)

    def test_same_size_is_noop(self):
        """Expanding to the same size returns equivalent weights."""
        from enigma_engine.core.progressive_growing import expand_model_weights

        cfg = self._make_config(dim=64, n_layers=2, n_heads=4, n_kv_heads=2)
        src_sd = self._make_state_dict(cfg)

        new_sd = expand_model_weights(src_sd, cfg, cfg)

        for key in src_sd:
            torch.testing.assert_close(new_sd[key], src_sd[key])

    def test_non_swiglu_expansion(self):
        """Expanding a non-SwiGLU model (standard FFN with up/down)."""
        from enigma_engine.core.progressive_growing import expand_model_weights

        src_cfg = self._make_config(dim=64, n_layers=2, use_swiglu=False)
        tgt_cfg = self._make_config(dim=128, n_layers=2, use_swiglu=False)
        src_sd = self._make_state_dict(src_cfg)

        new_sd = expand_model_weights(src_sd, src_cfg, tgt_cfg)

        # Standard FFN uses up/down instead of w1/w2/w3
        self.assertEqual(
            new_sd['layers.0.feed_forward.up.weight'].shape,
            (tgt_cfg.hidden_dim, tgt_cfg.dim))
        self.assertEqual(
            new_sd['layers.0.feed_forward.down.weight'].shape,
            (tgt_cfg.dim, tgt_cfg.hidden_dim))

    def test_bias_expansion(self):
        """Bias vectors are expanded correctly."""
        from enigma_engine.core.progressive_growing import expand_model_weights

        src_cfg = self._make_config(dim=64, n_layers=2, use_bias=True)
        tgt_cfg = self._make_config(dim=128, n_layers=2, use_bias=True)
        src_sd = self._make_state_dict(src_cfg)

        new_sd = expand_model_weights(src_sd, src_cfg, tgt_cfg)

        tgt_head_dim = tgt_cfg.dim // tgt_cfg.n_heads
        self.assertEqual(
            new_sd['layers.0.attention.wq.bias'].shape,
            (tgt_cfg.n_heads * tgt_head_dim,))

    def test_identity_layers_have_zero_residual_output(self):
        """New identity layers have wo and w2 weights zeroed."""
        from enigma_engine.core.progressive_growing import (
            expand_model_weights, compute_layer_mapping)

        src_cfg = self._make_config(dim=64, n_layers=2, n_heads=4, n_kv_heads=2)
        tgt_cfg = self._make_config(dim=64, n_layers=6, n_heads=4, n_kv_heads=2)
        src_sd = self._make_state_dict(src_cfg)

        new_sd = expand_model_weights(src_sd, src_cfg, tgt_cfg)
        mapping = compute_layer_mapping(2, 6)

        for new_idx, src_idx in enumerate(mapping):
            if src_idx == -1:
                p = f'layers.{new_idx}'
                # Residual output weights should be zero for identity layers
                self.assertTrue(
                    torch.all(new_sd[f'{p}.attention.wo.weight'] == 0),
                    f"Layer {new_idx} wo should be zero (identity)")
                self.assertTrue(
                    torch.all(new_sd[f'{p}.feed_forward.w2.weight'] == 0),
                    f"Layer {new_idx} w2 should be zero (identity)")

    def test_vocab_size_preserved(self):
        """Target vocab_size override is used for embedding."""
        from enigma_engine.core.progressive_growing import expand_model_weights

        src_cfg = self._make_config(dim=64, vocab_size=256)
        tgt_cfg = self._make_config(dim=128, vocab_size=512)
        src_sd = self._make_state_dict(src_cfg)

        new_sd = expand_model_weights(src_sd, src_cfg, tgt_cfg)

        tgt_padded = (512 + 63) & ~63
        self.assertEqual(new_sd['tok_embeddings.weight'].shape[0], tgt_padded)

    def test_output_weight_tied_to_embedding(self):
        """output.weight should reference tok_embeddings.weight."""
        from enigma_engine.core.progressive_growing import expand_model_weights

        src_cfg = self._make_config(dim=64)
        tgt_cfg = self._make_config(dim=128)
        src_sd = self._make_state_dict(src_cfg)

        new_sd = expand_model_weights(src_sd, src_cfg, tgt_cfg)

        torch.testing.assert_close(
            new_sd['tok_embeddings.weight'],
            new_sd['output.weight'])


class TestComputeLayerMapping(unittest.TestCase):
    """Test the layer mapping strategy for depth expansion."""

    def test_same_depth(self):
        from enigma_engine.core.progressive_growing import compute_layer_mapping
        mapping = compute_layer_mapping(4, 4)
        self.assertEqual(mapping, [0, 1, 2, 3])

    def test_double_depth(self):
        from enigma_engine.core.progressive_growing import compute_layer_mapping
        mapping = compute_layer_mapping(2, 4)
        # Each old layer should appear, with identity fills
        # Must have exactly 4 entries
        self.assertEqual(len(mapping), 4)
        # All source indices must appear at least once
        present = {m for m in mapping if m >= 0}
        self.assertEqual(present, {0, 1})

    def test_triple_depth(self):
        from enigma_engine.core.progressive_growing import compute_layer_mapping
        mapping = compute_layer_mapping(2, 6)
        self.assertEqual(len(mapping), 6)
        present = {m for m in mapping if m >= 0}
        self.assertEqual(present, {0, 1})


class TestValidateGrowthCompatibility(unittest.TestCase):
    """Test the validation function."""

    def test_valid_growth(self):
        from enigma_engine.core.progressive_growing import validate_growth

        src = ForgeConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2)
        tgt = ForgeConfig(dim=128, n_layers=4, n_heads=8, n_kv_heads=4)
        # Should not raise
        validate_growth(src, tgt)

    def test_dim_shrink_rejected(self):
        from enigma_engine.core.progressive_growing import validate_growth

        src = ForgeConfig(dim=128, n_layers=2, n_heads=4, n_kv_heads=2)
        tgt = ForgeConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2)
        with self.assertRaises(ValueError):
            validate_growth(src, tgt)

    def test_layers_shrink_rejected(self):
        from enigma_engine.core.progressive_growing import validate_growth

        src = ForgeConfig(dim=64, n_layers=4, n_heads=4, n_kv_heads=2)
        tgt = ForgeConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2)
        with self.assertRaises(ValueError):
            validate_growth(src, tgt)


class TestProgressiveGrowingStructure(unittest.TestCase):
    """Structural tests — verify code paths without GPU."""

    def test_expand_model_weights_exists(self):
        from enigma_engine.core.progressive_growing import expand_model_weights
        self.assertTrue(callable(expand_model_weights))

    def test_compute_layer_mapping_exists(self):
        from enigma_engine.core.progressive_growing import compute_layer_mapping
        self.assertTrue(callable(compute_layer_mapping))

    def test_validate_growth_exists(self):
        from enigma_engine.core.progressive_growing import validate_growth
        self.assertTrue(callable(validate_growth))

    def test_expand_function_signature(self):
        """expand_model_weights takes source_sd, source_config, target_config."""
        from enigma_engine.core.progressive_growing import expand_model_weights
        sig = inspect.signature(expand_model_weights)
        params = list(sig.parameters.keys())
        self.assertIn('source_sd', params)
        self.assertIn('source_config', params)
        self.assertIn('target_config', params)


if __name__ == '__main__':
    unittest.main()
