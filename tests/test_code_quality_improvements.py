#!/usr/bin/env python3
"""
Tests for code quality improvements.

Run with: pytest tests/test_code_quality_improvements.py -v
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVersionExport:
    """Tests for version string export."""
    
    def test_version_exists(self):
        """Test that __version__ can be imported."""
        from forge_ai import __version__
        assert __version__ is not None
    
    def test_version_value(self):
        """Test that __version__ has the correct value."""
        from forge_ai import __version__
        assert __version__ == "0.1.0"
    
    def test_version_in_all(self):
        """Test that __version__ is in __all__ exports."""
        from forge_ai import __all__
        assert '__version__' in __all__


class TestForgeConfigValidation:
    """Tests for ForgeConfig parameter validation."""
    
    def test_valid_config(self):
        """Test that a valid config can be created."""
        from forge_ai.core.model import ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=64,
            n_layers=4,
            n_heads=4,
            dropout=0.1,
            max_seq_len=512
        )
        assert config.vocab_size == 1000
        assert config.dim == 64
    
    def test_negative_vocab_size(self):
        """Test that negative vocab_size raises ValueError."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            ForgeConfig(vocab_size=-1)
    
    def test_zero_vocab_size(self):
        """Test that zero vocab_size raises ValueError."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            ForgeConfig(vocab_size=0)
    
    def test_negative_dim(self):
        """Test that negative dim raises ValueError."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="dim must be positive"):
            ForgeConfig(dim=-1)
    
    def test_zero_dim(self):
        """Test that zero dim raises ValueError."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="dim must be positive"):
            ForgeConfig(dim=0)
    
    def test_negative_n_layers(self):
        """Test that negative n_layers raises ValueError."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="n_layers must be positive"):
            ForgeConfig(n_layers=-1)
    
    def test_zero_n_layers(self):
        """Test that zero n_layers raises ValueError."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="n_layers must be positive"):
            ForgeConfig(n_layers=0)
    
    def test_negative_n_heads(self):
        """Test that negative n_heads raises ValueError."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="n_heads must be positive"):
            ForgeConfig(n_heads=-1)
    
    def test_zero_n_heads(self):
        """Test that zero n_heads raises ValueError."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="n_heads must be positive"):
            ForgeConfig(n_heads=0)
    
    def test_dropout_below_zero(self):
        """Test that dropout < 0 raises ValueError."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            ForgeConfig(dropout=-0.1)
    
    def test_dropout_above_one(self):
        """Test that dropout > 1 raises ValueError."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            ForgeConfig(dropout=1.5)
    
    def test_dropout_zero_valid(self):
        """Test that dropout = 0 is valid."""
        from forge_ai.core.model import ForgeConfig
        config = ForgeConfig(dropout=0.0)
        assert config.dropout == 0.0
    
    def test_dropout_one_valid(self):
        """Test that dropout = 1 is valid."""
        from forge_ai.core.model import ForgeConfig
        config = ForgeConfig(dropout=1.0)
        assert config.dropout == 1.0
    
    def test_negative_max_seq_len(self):
        """Test that negative max_seq_len raises ValueError."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            ForgeConfig(max_seq_len=-1)
    
    def test_zero_max_seq_len(self):
        """Test that zero max_seq_len raises ValueError."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            ForgeConfig(max_seq_len=0)
    
    def test_n_heads_not_divide_dim(self):
        """Test that n_heads must divide evenly into dim."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="n_heads.*must divide evenly into dim"):
            ForgeConfig(dim=100, n_heads=7)  # 100 % 7 != 0
    
    def test_n_heads_divide_dim_valid(self):
        """Test that valid dim/n_heads combination works."""
        from forge_ai.core.model import ForgeConfig
        config = ForgeConfig(dim=64, n_heads=8)  # 64 % 8 == 0
        assert config.dim == 64
        assert config.n_heads == 8
    
    def test_n_kv_heads_not_divide_n_heads(self):
        """Test that n_kv_heads must divide evenly into n_heads."""
        from forge_ai.core.model import ForgeConfig
        with pytest.raises(ValueError, match="n_kv_heads.*must divide evenly into n_heads"):
            ForgeConfig(n_heads=8, n_kv_heads=3)  # 8 % 3 != 0
    
    def test_n_kv_heads_divide_n_heads_valid(self):
        """Test that valid n_heads/n_kv_heads combination works."""
        from forge_ai.core.model import ForgeConfig
        config = ForgeConfig(n_heads=8, n_kv_heads=4)  # 8 % 4 == 0
        assert config.n_heads == 8
        assert config.n_kv_heads == 4
    
    def test_n_kv_heads_defaults_to_n_heads(self):
        """Test that n_kv_heads defaults to n_heads if not specified."""
        from forge_ai.core.model import ForgeConfig
        config = ForgeConfig(n_heads=8)
        assert config.n_kv_heads == 8


class TestPyTypedMarker:
    """Tests for py.typed marker file."""
    
    def test_py_typed_exists(self):
        """Test that py.typed marker file exists."""
        py_typed_path = Path(__file__).parent.parent / "forge_ai" / "py.typed"
        assert py_typed_path.exists(), "py.typed marker file should exist"
    
    def test_py_typed_is_file(self):
        """Test that py.typed is a file (not a directory)."""
        py_typed_path = Path(__file__).parent.parent / "forge_ai" / "py.typed"
        assert py_typed_path.is_file(), "py.typed should be a file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
