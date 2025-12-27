#!/usr/bin/env python3
"""
Tests for the Enigma core model.

Run with: pytest tests/test_model.py -v
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRMSNorm:
    """Tests for RMSNorm layer."""
    
    def test_creation(self):
        """Test RMSNorm can be created."""
        from enigma.core.model import RMSNorm
        norm = RMSNorm(64)
        assert norm.weight.shape == (64,)
    
    def test_forward(self):
        """Test RMSNorm forward pass."""
        from enigma.core.model import RMSNorm
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape
    
    def test_normalization(self):
        """Test that RMSNorm actually normalizes."""
        from enigma.core.model import RMSNorm
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64) * 100  # Large values
        out = norm(x)
        # Output should have bounded magnitude
        assert out.abs().max() < 100


class TestRotaryEmbedding:
    """Tests for Rotary Position Embeddings."""
    
    def test_creation(self):
        """Test RotaryEmbedding can be created."""
        from enigma.core.model import RotaryEmbedding
        rope = RotaryEmbedding(64, max_seq_len=1024)
        assert rope.max_seq_len == 1024
    
    def test_forward(self):
        """Test RotaryEmbedding returns cos and sin."""
        from enigma.core.model import RotaryEmbedding
        rope = RotaryEmbedding(64, max_seq_len=1024)
        x = torch.randn(2, 10, 64)
        cos, sin = rope(x, 10)
        assert cos.shape == (1, 1, 10, 64)
        assert sin.shape == (1, 1, 10, 64)
    
    def test_extends_cache(self):
        """Test that cache extends for longer sequences."""
        from enigma.core.model import RotaryEmbedding
        rope = RotaryEmbedding(64, max_seq_len=10)
        x = torch.randn(2, 20, 64)
        cos, sin = rope(x, 20)
        assert cos.shape == (1, 1, 20, 64)


class TestSwiGLU:
    """Tests for SwiGLU activation."""
    
    def test_creation(self):
        """Test SwiGLU can be created."""
        from enigma.core.model import SwiGLU
        swiglu = SwiGLU(64, 256)
        assert swiglu.w1.in_features == 64
    
    def test_forward(self):
        """Test SwiGLU forward pass."""
        from enigma.core.model import SwiGLU
        swiglu = SwiGLU(64, 256)
        x = torch.randn(2, 10, 64)
        out = swiglu(x)
        assert out.shape == x.shape
    
    def test_activation(self):
        """Test SwiGLU produces reasonable output."""
        from enigma.core.model import SwiGLU
        swiglu = SwiGLU(64, 256)
        x = torch.zeros(2, 10, 64)
        out = swiglu(x)
        # With zero input, output should be small
        assert out.abs().mean() < 1.0


class TestEnigmaAttention:
    """Tests for Enigma attention mechanism."""
    
    def test_creation(self):
        """Test EnigmaAttention can be created."""
        from enigma.core.model import EnigmaAttention
        attn = EnigmaAttention(64, 4, max_seq_len=1024)
        assert attn.n_heads == 4
        assert attn.head_dim == 16
    
    def test_forward_no_cache(self):
        """Test attention without cache."""
        from enigma.core.model import EnigmaAttention
        attn = EnigmaAttention(64, 4, max_seq_len=1024)
        x = torch.randn(2, 10, 64)
        out, cache = attn(x, use_cache=False)
        assert out.shape == x.shape
        assert cache is None
    
    def test_forward_with_cache(self):
        """Test attention with cache."""
        from enigma.core.model import EnigmaAttention
        attn = EnigmaAttention(64, 4, max_seq_len=1024)
        x = torch.randn(2, 10, 64)
        out, cache = attn(x, use_cache=True)
        assert out.shape == x.shape
        assert cache is not None
        assert len(cache) == 2  # k and v
    
    def test_cache_continuation(self):
        """Test using cache for continuation."""
        from enigma.core.model import EnigmaAttention
        attn = EnigmaAttention(64, 4, max_seq_len=1024)
        
        # First pass
        x1 = torch.randn(2, 10, 64)
        out1, cache1 = attn(x1, use_cache=True)
        
        # Continue with cache
        x2 = torch.randn(2, 1, 64)
        out2, cache2 = attn(x2, kv_cache=cache1, use_cache=True)
        
        assert out2.shape == (2, 1, 64)
        assert cache2[0].shape[2] == 11  # 10 + 1 cached keys


class TestEnigmaBlock:
    """Tests for Enigma transformer block."""
    
    def test_creation(self):
        """Test EnigmaBlock can be created."""
        from enigma.core.model import EnigmaBlock
        block = EnigmaBlock(64, 4, ff_mult=4.0, max_seq_len=1024)
        assert block is not None
    
    def test_forward(self):
        """Test EnigmaBlock forward pass."""
        from enigma.core.model import EnigmaBlock
        block = EnigmaBlock(64, 4, ff_mult=4.0, max_seq_len=1024)
        x = torch.randn(2, 10, 64)
        out, cache = block(x)
        assert out.shape == x.shape


class TestEnigmaModel:
    """Tests for the full Enigma model."""
    
    def test_creation(self):
        """Test Enigma model can be created."""
        from enigma.core.model import Enigma
        model = Enigma(vocab_size=1000, dim=64, depth=2, heads=4)
        assert model.vocab_size == 1000
        assert model.dim == 64
        assert model.depth == 2
    
    def test_num_parameters(self):
        """Test parameter counting."""
        from enigma.core.model import Enigma
        model = Enigma(vocab_size=1000, dim=64, depth=2, heads=4)
        params = model.num_parameters
        assert params > 0
        # Manual check: embedding + blocks + head
        # Note: weight tying means embedding and head share weights
    
    def test_forward_no_cache(self):
        """Test forward without cache."""
        from enigma.core.model import Enigma
        model = Enigma(vocab_size=1000, dim=64, depth=2, heads=4)
        x = torch.randint(0, 1000, (2, 10))
        out = model(x, use_cache=False)
        assert out.shape == (2, 10, 1000)
    
    def test_forward_with_cache(self):
        """Test forward with cache."""
        from enigma.core.model import Enigma
        model = Enigma(vocab_size=1000, dim=64, depth=2, heads=4)
        x = torch.randint(0, 1000, (2, 10))
        out, cache = model(x, use_cache=True)
        assert out.shape == (2, 10, 1000)
        assert len(cache) == 2  # 2 layers
    
    def test_generate(self):
        """Test generation."""
        from enigma.core.model import Enigma
        model = Enigma(vocab_size=1000, dim=64, depth=2, heads=4)
        model.eval()
        x = torch.randint(0, 1000, (1, 5))
        out = model.generate(x, max_new_tokens=10, use_cache=True)
        assert out.shape == (1, 15)  # 5 + 10
    
    def test_generate_deterministic(self):
        """Test generation is deterministic with same seed."""
        from enigma.core.model import Enigma
        model = Enigma(vocab_size=1000, dim=64, depth=2, heads=4)
        model.eval()
        x = torch.randint(0, 1000, (1, 5))
        
        torch.manual_seed(42)
        out1 = model.generate(x.clone(), max_new_tokens=5, temperature=0.5)
        
        torch.manual_seed(42)
        out2 = model.generate(x.clone(), max_new_tokens=5, temperature=0.5)
        
        assert torch.equal(out1, out2)
    
    def test_backwards_compat(self):
        """Test TinyEnigma alias."""
        from enigma.core.model import Enigma, TinyEnigma
        assert TinyEnigma is Enigma


class TestModelConfig:
    """Tests for model configuration."""
    
    def test_presets_exist(self):
        """Test model presets are defined."""
        from enigma.core.model_config import MODEL_PRESETS
        assert "tiny" in MODEL_PRESETS
        assert "small" in MODEL_PRESETS
        assert "medium" in MODEL_PRESETS
        assert "large" in MODEL_PRESETS
    
    def test_get_config(self):
        """Test getting model config."""
        from enigma.core.model_config import get_model_config
        config = get_model_config("tiny")
        assert "dim" in config
        assert "depth" in config
        assert "heads" in config
    
    def test_estimate_parameters(self):
        """Test parameter estimation."""
        from enigma.core.model_config import estimate_parameters, get_model_config
        config = get_model_config("tiny")
        params = estimate_parameters(vocab_size=1000, **config)
        assert params > 0
    
    def test_create_model_from_config(self):
        """Test creating model from preset."""
        from enigma.core.model_config import get_model_config
        from enigma.core.model import Enigma
        
        config = get_model_config("tiny")
        model = Enigma(
            vocab_size=1000,
            dim=config["dim"],
            depth=config["depth"],
            heads=config["heads"],
        )
        assert model is not None


class TestModelScaling:
    """Tests for model scaling."""
    
    def test_grow_model(self):
        """Test growing model dimensions."""
        from enigma.core.model import Enigma
        from enigma.core.model_scaling import grow_model
        
        small = Enigma(vocab_size=1000, dim=32, depth=2, heads=2)
        large = grow_model(small, "small", vocab_size=1000)
        
        assert large.dim > small.dim
        assert large.depth >= small.depth
    
    def test_shrink_model(self):
        """Test shrinking model dimensions."""
        from enigma.core.model import Enigma
        from enigma.core.model_scaling import shrink_model
        
        large = Enigma(vocab_size=1000, dim=256, depth=6, heads=8)
        small = shrink_model(large, "tiny", vocab_size=1000)
        
        assert small.dim < large.dim
        assert small.depth <= large.depth


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
