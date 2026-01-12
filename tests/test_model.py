#!/usr/bin/env python3
"""
Tests for the AI Tester core model.

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


class TestAttention:
    """Tests for Attention mechanism."""
    
    def test_creation(self):
        """Test Attention can be created."""
        from enigma.core.model import Attention, AITesterConfig
        config = AITesterConfig(dim=64, n_heads=4, n_kv_heads=2)
        attn = Attention(config)
        assert attn.n_heads == 4
        assert attn.n_kv_heads == 2
    
    def test_forward(self):
        """Test attention forward pass."""
        from enigma.core.model import Attention, AITesterConfig
        config = AITesterConfig(dim=64, n_heads=4, n_kv_heads=2)
        attn = Attention(config)
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert out.shape == x.shape
    
    def test_cache(self):
        """Test attention with cache."""
        from enigma.core.model import Attention, AITesterConfig
        config = AITesterConfig(dim=64, n_heads=4, n_kv_heads=2)
        attn = Attention(config)
        
        # First pass with cache
        x = torch.randn(2, 10, 64)
        out = attn(x, use_cache=True)
        assert attn.cache_k is not None
        assert attn.cache_v is not None
        
        # Clear cache
        attn.clear_cache()
        assert attn.cache_k is None


class TestFeedForward:
    """Tests for FeedForward network."""
    
    def test_creation_swiglu(self):
        """Test SwiGLU FeedForward can be created."""
        from enigma.core.model import FeedForward, AITesterConfig
        config = AITesterConfig(dim=64, use_swiglu=True)
        ff = FeedForward(config)
        assert ff.use_swiglu == True
    
    def test_forward(self):
        """Test FeedForward forward pass."""
        from enigma.core.model import FeedForward, AITesterConfig
        config = AITesterConfig(dim=64, use_swiglu=True)
        ff = FeedForward(config)
        x = torch.randn(2, 10, 64)
        out = ff(x)
        assert out.shape == x.shape


class TestTransformerBlock:
    """Tests for TransformerBlock."""
    
    def test_creation(self):
        """Test TransformerBlock can be created."""
        from enigma.core.model import TransformerBlock, AITesterConfig
        config = AITesterConfig(dim=64, n_heads=4)
        block = TransformerBlock(config, layer_id=0)
        assert block is not None
    
    def test_forward(self):
        """Test TransformerBlock forward pass."""
        from enigma.core.model import TransformerBlock, AITesterConfig
        config = AITesterConfig(dim=64, n_heads=4)
        block = TransformerBlock(config, layer_id=0)
        x = torch.randn(2, 10, 64)
        out = block(x)
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
    
    def test_forward(self):
        """Test forward pass."""
        from enigma.core.model import Enigma
        model = Enigma(vocab_size=1000, dim=64, depth=2, heads=4)
        x = torch.randint(0, 1000, (2, 10))
        out = model(x)
        # Output should be (batch, seq_len, vocab_size)
        assert out.shape[-1] == 1000
    
    def test_generate(self):
        """Test generation."""
        from enigma.core.model import Enigma
        model = Enigma(vocab_size=1000, dim=64, depth=2, heads=4)
        model.eval()
        x = torch.randint(0, 1000, (1, 5))
        out = model.generate(x, max_new_tokens=10)
        assert out.shape[1] == 15  # 5 + 10
    
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
        """Test TinyAITester alias."""
        from enigma.core.model import Enigma, TinyAITester
        assert TinyAITester is Enigma


class TestModelPresets:
    """Tests for model presets."""
    
    def test_presets_exist(self):
        """Test model presets are defined."""
        from enigma.core.model import MODEL_PRESETS
        assert "tiny" in MODEL_PRESETS
        assert "small" in MODEL_PRESETS
        assert "medium" in MODEL_PRESETS
        assert "large" in MODEL_PRESETS
    
    def test_create_from_preset(self):
        """Test creating model from preset."""
        from enigma.core.model import create_model
        model = create_model("tiny", vocab_size=1000)
        assert model is not None


class TestModelConfig:
    """Tests for model configuration."""
    
    def test_config_creation(self):
        """Test AITesterConfig can be created."""
        from enigma.core.model import AITesterConfig
        config = AITesterConfig(dim=256, n_layers=6, n_heads=8)
        assert config.dim == 256
        assert config.n_layers == 6
        assert config.n_heads == 8
    
    def test_config_to_dict(self):
        """Test config serialization."""
        from enigma.core.model import AITesterConfig
        config = AITesterConfig(dim=256)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d['dim'] == 256
    
    def test_config_from_dict(self):
        """Test config deserialization."""
        from enigma.core.model import AITesterConfig
        d = {'dim': 128, 'n_layers': 4, 'n_heads': 4}
        config = AITesterConfig.from_dict(d)
        assert config.dim == 128
        assert config.n_layers == 4


class TestModelScaling:
    """Tests for model scaling (if available)."""
    
    @pytest.mark.skip(reason="Model scaling module may not be complete")
    def test_grow_model(self):
        """Test growing model dimensions."""
        from enigma.core.model import Enigma
        from enigma.core.model_scaling import grow_model
        
        small = Enigma(vocab_size=1000, dim=32, depth=2, heads=2)
        large = grow_model(small, "small", vocab_size=1000)
        
        assert large.dim > small.dim
    
    @pytest.mark.skip(reason="Model scaling module may not be complete")
    def test_shrink_model(self):
        """Test shrinking model dimensions."""
        from enigma.core.model import Enigma
        from enigma.core.model_scaling import shrink_model
        
        large = Enigma(vocab_size=1000, dim=256, depth=6, heads=8)
        small = shrink_model(large, "tiny", vocab_size=1000)
        
        assert small.dim < large.dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
