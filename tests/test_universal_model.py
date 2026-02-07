#!/usr/bin/env python3
"""
Tests for Universal Model enhancements in enigma_engine/core/model.py

Run with: pytest tests/test_universal_model.py -v
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBackwardCompatibility:
    """Ensure all existing code still works unchanged."""
    
    def test_create_model_basic(self):
        """Test basic model creation still works."""
        from enigma_engine.core.model import create_model
        model = create_model('small')
        assert model is not None
        assert model.num_parameters > 0
    
    def test_forge_config_basic(self):
        """Test basic ForgeConfig creation."""
        from enigma_engine.core.model import ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=256,
            n_layers=4,
            n_heads=4
        )
        assert config.vocab_size == 1000
        assert config.dim == 256
        assert config.n_layers == 4
        assert config.n_heads == 4
    
    def test_forge_init_basic(self):
        """Test Forge class can be initialized."""
        from enigma_engine.core.model import Forge, ForgeConfig
        config = ForgeConfig(vocab_size=1000, dim=128, n_layers=2, n_heads=4)
        model = Forge(config=config)
        assert model.vocab_size == 1000
    
    def test_forward_pass(self):
        """Test model forward pass works."""
        from enigma_engine.core.model import create_model
        model = create_model('tiny')
        input_ids = torch.randint(0, 1000, (1, 10))
        logits = model(input_ids)
        assert logits.shape == (1, 10, model.vocab_size)
    
    def test_generate(self):
        """Test generation works."""
        from enigma_engine.core.model import create_model
        model = create_model('nano')
        input_ids = torch.randint(0, 1000, (1, 5))
        output = model.generate(input_ids, max_new_tokens=5)
        assert output.shape[1] >= input_ids.shape[1]


class TestRoPEScaling:
    """Test RoPE scaling for extended context."""
    
    def test_rope_scaling_config_linear(self):
        """Test linear RoPE scaling configuration."""
        from enigma_engine.core.model import ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            rope_scaling_type='linear',
            rope_scaling_factor=2.0
        )
        assert config.rope_scaling_type == 'linear'
        assert config.rope_scaling_factor == 2.0
    
    def test_rope_scaling_config_dynamic(self):
        """Test dynamic NTK RoPE scaling configuration."""
        from enigma_engine.core.model import ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            rope_scaling_type='dynamic',
            rope_scaling_factor=2.0
        )
        assert config.rope_scaling_type == 'dynamic'
    
    def test_rope_scaling_config_yarn(self):
        """Test YaRN RoPE scaling configuration."""
        from enigma_engine.core.model import ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            rope_scaling_type='yarn',
            rope_scaling_factor=4.0
        )
        assert config.rope_scaling_type == 'yarn'
    
    def test_rope_scaling_model_creation(self):
        """Test model can be created with RoPE scaling."""
        from enigma_engine.core.model import Forge, ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            max_seq_len=1024,
            rope_scaling_type='dynamic',
            rope_scaling_factor=2.0
        )
        model = Forge(config=config)
        assert model.config.rope_scaling_type == 'dynamic'
        # Model should be created without errors
        assert model.freqs_cis is not None
    
    def test_invalid_rope_scaling_type(self):
        """Test that invalid RoPE scaling type raises error."""
        from enigma_engine.core.model import ForgeConfig
        with pytest.raises(ValueError, match="rope_scaling_type must be one of"):
            ForgeConfig(
                vocab_size=1000,
                dim=128,
                n_layers=2,
                n_heads=4,
                rope_scaling_type='invalid'
            )


class TestMultiModal:
    """Test multi-modal integration hooks."""
    
    def test_vision_projection_config(self):
        """Test vision projection configuration."""
        from enigma_engine.core.model import ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=256,
            n_layers=2,
            n_heads=4,
            vision_hidden_size=768
        )
        assert config.vision_hidden_size == 768
    
    def test_audio_projection_config(self):
        """Test audio projection configuration."""
        from enigma_engine.core.model import ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=256,
            n_layers=2,
            n_heads=4,
            audio_hidden_size=512
        )
        assert config.audio_hidden_size == 512
    
    def test_vision_projection_layer(self):
        """Test vision projection layer is created."""
        from enigma_engine.core.model import Forge, ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=256,
            n_layers=2,
            n_heads=4,
            vision_hidden_size=768
        )
        model = Forge(config=config)
        assert model.vision_projection is not None
        assert model.vision_projection.in_features == 768
        assert model.vision_projection.out_features == 256
    
    def test_audio_projection_layer(self):
        """Test audio projection layer is created."""
        from enigma_engine.core.model import Forge, ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=256,
            n_layers=2,
            n_heads=4,
            audio_hidden_size=512
        )
        model = Forge(config=config)
        assert model.audio_projection is not None
    
    def test_forward_multimodal_text_only(self):
        """Test multimodal forward with text only."""
        from enigma_engine.core.model import Forge, ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            vision_hidden_size=256
        )
        model = Forge(config=config)
        input_ids = torch.randint(0, 1000, (1, 10))
        logits = model.forward_multimodal(input_ids=input_ids)
        assert logits.shape == (1, 10, 1000)
    
    def test_forward_multimodal_vision(self):
        """Test multimodal forward with vision features."""
        from enigma_engine.core.model import Forge, ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            vision_hidden_size=256
        )
        model = Forge(config=config)
        vision_features = torch.randn(1, 5, 256)  # 5 vision tokens
        input_ids = torch.randint(0, 1000, (1, 10))
        logits = model.forward_multimodal(
            input_ids=input_ids,
            vision_features=vision_features
        )
        # Should have 5 vision tokens + 10 text tokens = 15 total
        assert logits.shape[1] == 15


class TestMoEConfig:
    """Test Mixture of Experts configuration."""
    
    def test_moe_config(self):
        """Test MoE configuration parameters."""
        from enigma_engine.core.model import ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=256,
            n_layers=4,
            n_heads=4,
            use_moe=True,
            num_experts=8,
            num_experts_per_token=2
        )
        assert config.use_moe is True
        assert config.num_experts == 8
        assert config.num_experts_per_token == 2
    
    def test_moe_config_validation(self):
        """Test MoE config validation."""
        from enigma_engine.core.model import ForgeConfig
        # Should raise error if num_experts_per_token > num_experts
        with pytest.raises(ValueError):
            ForgeConfig(
                vocab_size=1000,
                dim=256,
                n_layers=4,
                n_heads=4,
                use_moe=True,
                num_experts=4,
                num_experts_per_token=8  # Invalid: more than num_experts
            )


class TestEnhancedKVCache:
    """Test enhanced KV-cache features."""
    
    def test_sliding_window_config(self):
        """Test sliding window attention configuration."""
        from enigma_engine.core.model import ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=256,
            n_layers=4,
            n_heads=4,
            sliding_window=512
        )
        assert config.sliding_window == 512
    
    def test_paged_attention_config(self):
        """Test paged attention configuration."""
        from enigma_engine.core.model import ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=256,
            n_layers=4,
            n_heads=4,
            use_paged_attn=True
        )
        assert config.use_paged_attn is True
    
    def test_kv_cache_dtype_config(self):
        """Test KV-cache dtype configuration."""
        from enigma_engine.core.model import ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=256,
            n_layers=4,
            n_heads=4,
            kv_cache_dtype='int8'
        )
        assert config.kv_cache_dtype == 'int8'


class TestUniversalLoading:
    """Test universal model loading capabilities."""
    
    def test_from_any_method_exists(self):
        """Test from_any classmethod exists."""
        from enigma_engine.core.model import Forge
        assert hasattr(Forge, 'from_any')
        assert callable(Forge.from_any)
    
    def test_from_huggingface_method_exists(self):
        """Test from_huggingface classmethod exists."""
        from enigma_engine.core.model import Forge
        assert hasattr(Forge, 'from_huggingface')
        assert callable(Forge.from_huggingface)
    
    def test_from_safetensors_method_exists(self):
        """Test from_safetensors classmethod exists."""
        from enigma_engine.core.model import Forge
        assert hasattr(Forge, 'from_safetensors')
        assert callable(Forge.from_safetensors)
    
    def test_from_gguf_method_exists(self):
        """Test from_gguf classmethod exists."""
        from enigma_engine.core.model import Forge
        assert hasattr(Forge, 'from_gguf')
        assert callable(Forge.from_gguf)
    
    def test_from_onnx_method_exists(self):
        """Test from_onnx classmethod exists."""
        from enigma_engine.core.model import Forge
        assert hasattr(Forge, 'from_onnx')
        assert callable(Forge.from_onnx)


class TestLoRASupport:
    """Test LoRA adapter support."""
    
    def test_load_lora_method_exists(self):
        """Test load_lora method exists."""
        from enigma_engine.core.model import Forge
        assert hasattr(Forge, 'load_lora')
        assert callable(Forge.load_lora)
    
    def test_merge_lora_method_exists(self):
        """Test merge_lora method exists."""
        from enigma_engine.core.model import Forge
        assert hasattr(Forge, 'merge_lora')
        assert callable(Forge.merge_lora)


class TestSpeculativeDecoding:
    """Test speculative decoding support."""
    
    def test_enable_speculative_decoding(self):
        """Test enabling speculative decoding."""
        from enigma_engine.core.model import create_model
        draft_model = create_model('nano')
        main_model = create_model('small')
        
        main_model.enable_speculative_decoding(draft_model, num_speculative_tokens=4)
        assert hasattr(main_model, '_use_speculation')
        assert main_model._use_speculation is True
        assert main_model._num_speculative_tokens == 4
    
    def test_disable_speculative_decoding(self):
        """Test disabling speculative decoding."""
        from enigma_engine.core.model import create_model
        draft_model = create_model('nano')
        main_model = create_model('small')
        
        main_model.enable_speculative_decoding(draft_model)
        main_model.disable_speculative_decoding()
        assert main_model._use_speculation is False
    
    def test_generate_speculative_method_exists(self):
        """Test generate_speculative method exists."""
        from enigma_engine.core.model import Forge
        assert hasattr(Forge, 'generate_speculative')
        assert callable(Forge.generate_speculative)


class TestConfigSerialization:
    """Test config serialization with new parameters."""
    
    def test_config_to_dict(self):
        """Test config can be serialized to dict."""
        from enigma_engine.core.model import ForgeConfig
        config = ForgeConfig(
            vocab_size=1000,
            dim=256,
            n_layers=4,
            n_heads=4,
            rope_scaling_type='dynamic',
            rope_scaling_factor=2.0,
            use_moe=True,
            num_experts=8
        )
        config_dict = config.to_dict()
        assert 'rope_scaling_type' in config_dict
        assert 'rope_scaling_factor' in config_dict
        assert 'use_moe' in config_dict
        assert 'num_experts' in config_dict
    
    def test_config_from_dict(self):
        """Test config can be loaded from dict."""
        from enigma_engine.core.model import ForgeConfig
        config_dict = {
            'vocab_size': 1000,
            'dim': 256,
            'n_layers': 4,
            'n_heads': 4,
            'rope_scaling_type': 'yarn',
            'rope_scaling_factor': 4.0,
            'use_moe': True,
            'num_experts': 16
        }
        config = ForgeConfig.from_dict(config_dict)
        assert config.rope_scaling_type == 'yarn'
        assert config.rope_scaling_factor == 4.0
        assert config.use_moe is True
        assert config.num_experts == 16


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
