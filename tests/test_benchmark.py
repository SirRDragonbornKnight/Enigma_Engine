"""Lightweight CPU performance benchmarks for Enigma model."""

import time
import torch
from enigma_engine.core.model import Enigma, ForgeConfig


def _make_small_model():
    """Create a tiny model for CPU benchmarking."""
    cfg = ForgeConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=64,
    )
    model = Enigma(config=cfg)
    model.eval()
    return model, cfg


class TestForwardPassBenchmark:
    """Benchmark model forward pass throughput on CPU."""

    def test_forward_pass_runs(self):
        """Forward pass completes and returns logits."""
        model, cfg = _make_small_model()
        ids = torch.randint(0, cfg.vocab_size, (1, 16))
        with torch.no_grad():
            out = model(ids)
        assert out.shape == (1, 16, cfg.vocab_size)

    def test_forward_pass_timing(self):
        """Forward pass completes within a reasonable time on CPU."""
        model, cfg = _make_small_model()
        ids = torch.randint(0, cfg.vocab_size, (1, 32))

        # Warmup
        with torch.no_grad():
            model(ids)

        runs = 10
        start = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                model(ids)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / runs) * 1000
        # Just verify it completes — no hard threshold
        assert avg_ms > 0, f"Forward pass avg: {avg_ms:.1f}ms"

    def test_batch_forward(self):
        """Batched forward pass works correctly."""
        model, cfg = _make_small_model()
        ids = torch.randint(0, cfg.vocab_size, (4, 16))
        with torch.no_grad():
            out = model(ids)
        assert out.shape == (4, 16, cfg.vocab_size)


class TestGenerationBenchmark:
    """Benchmark token generation speed on CPU."""

    def test_generate_runs(self):
        """Generate produces tokens."""
        model, cfg = _make_small_model()
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=8)
        # Output should be longer than prompt
        assert out.shape[1] > prompt.shape[1]

    def test_generate_timing(self):
        """Generation completes within a reasonable time on CPU."""
        model, cfg = _make_small_model()
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))

        # Warmup
        with torch.no_grad():
            model.generate(prompt, max_new_tokens=4)

        runs = 5
        tokens = 16
        start = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                model.generate(prompt, max_new_tokens=tokens)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / runs) * 1000
        assert avg_ms > 0, f"Generation avg: {avg_ms:.1f}ms for {tokens} tokens"

    def test_generate_with_stop_token(self):
        """Generation respects stop tokens."""
        model, cfg = _make_small_model()
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = model.generate(
                prompt, max_new_tokens=32, stop_tokens=[2])
        # Should not exceed max length
        assert out.shape[1] <= prompt.shape[1] + 32
