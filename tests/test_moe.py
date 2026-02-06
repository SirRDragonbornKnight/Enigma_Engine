"""
Unit tests for Mixture of Experts (MoE) module.

Tests the MoE architecture components: Router, Expert, MoELayer, MoETransformer.
"""

import pytest
import torch
import torch.nn as nn

from forge_ai.core.moe import (
    MoEConfig,
    Router,
    RouterType,
    Expert,
    MoELayer,
    MoETransformerBlock,
    MoETransformer,
)


class TestMoEConfig:
    """Tests for MoEConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MoEConfig()
        assert config.num_experts == 8
        assert config.num_experts_per_tok == 2
        assert config.hidden_size == 768
        assert config.intermediate_size == 3072
        assert config.load_balancing_loss_coef == 0.01
        assert config.capacity_factor == 1.25
        assert config.drop_tokens is False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MoEConfig(
            num_experts=4,
            num_experts_per_tok=1,
            hidden_size=512,
            intermediate_size=2048
        )
        assert config.num_experts == 4
        assert config.num_experts_per_tok == 1
        assert config.hidden_size == 512


class TestRouter:
    """Tests for Router module."""
    
    @pytest.fixture
    def router(self):
        """Create a test router."""
        return Router(
            hidden_size=64,
            num_experts=4,
            top_k=2,
            router_type=RouterType.TOP_K
        )
    
    def test_router_init(self, router):
        """Test router initialization."""
        assert router.hidden_size == 64
        assert router.num_experts == 4
        assert router.top_k == 2
        assert isinstance(router.gate, nn.Linear)
    
    def test_router_forward_shape(self, router):
        """Test router forward pass output shapes."""
        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, 64)
        
        router_probs, expert_indices, expert_weights = router(hidden_states)
        
        assert router_probs.shape == (batch_size, seq_len, 4)  # num_experts
        assert expert_indices.shape == (batch_size, seq_len, 2)  # top_k
        assert expert_weights.shape == (batch_size, seq_len, 2)  # top_k
    
    def test_router_probs_sum_to_one(self, router):
        """Test that router probabilities sum to 1."""
        hidden_states = torch.randn(2, 8, 64)
        router_probs, _, _ = router(hidden_states)
        
        sums = router_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_expert_weights_sum_to_one(self, router):
        """Test that expert weights sum to 1."""
        hidden_states = torch.randn(2, 8, 64)
        _, _, expert_weights = router(hidden_states)
        
        sums = expert_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_load_balancing_loss(self, router):
        """Test load balancing loss computation."""
        hidden_states = torch.randn(4, 16, 64)
        router_probs, expert_indices, _ = router(hidden_states)
        
        lb_loss = router.compute_load_balancing_loss(router_probs, expert_indices)
        
        assert isinstance(lb_loss, torch.Tensor)
        assert lb_loss.ndim == 0  # Scalar
        assert lb_loss.item() >= 0


class TestExpert:
    """Tests for Expert module."""
    
    @pytest.fixture
    def expert(self):
        """Create a test expert."""
        return Expert(hidden_size=64, intermediate_size=256)
    
    def test_expert_init(self, expert):
        """Test expert initialization."""
        assert isinstance(expert.gate_proj, nn.Linear)
        assert isinstance(expert.up_proj, nn.Linear)
        assert isinstance(expert.down_proj, nn.Linear)
    
    def test_expert_forward_shape(self, expert):
        """Test expert forward pass preserves shape."""
        x = torch.randn(32, 64)  # [tokens, hidden_size]
        output = expert(x)
        assert output.shape == x.shape
    
    def test_expert_forward_batch(self, expert):
        """Test expert with batch dimension."""
        x = torch.randn(2, 8, 64)  # [batch, seq, hidden]
        output = expert(x)
        assert output.shape == x.shape


class TestMoELayer:
    """Tests for MoELayer module."""
    
    @pytest.fixture
    def moe_layer(self):
        """Create a test MoE layer."""
        config = MoEConfig(
            num_experts=4,
            num_experts_per_tok=2,
            hidden_size=64,
            intermediate_size=256
        )
        return MoELayer(config)
    
    def test_moe_layer_init(self, moe_layer):
        """Test MoE layer initialization."""
        assert isinstance(moe_layer.router, Router)
        assert len(moe_layer.experts) == 4
        assert all(isinstance(e, Expert) for e in moe_layer.experts)
    
    def test_moe_layer_forward(self, moe_layer):
        """Test MoE layer forward pass."""
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 64)
        
        output = moe_layer(x)
        
        assert output.shape == x.shape
    
    def test_moe_layer_aux_loss(self, moe_layer):
        """Test auxiliary loss during training."""
        moe_layer.train()
        x = torch.randn(2, 8, 64)
        
        _ = moe_layer(x)
        aux_loss = moe_layer.get_aux_loss()
        
        assert isinstance(aux_loss, float)
        assert aux_loss >= 0
    
    def test_moe_layer_no_aux_loss_eval(self, moe_layer):
        """Test no auxiliary loss accumulation in eval mode."""
        moe_layer.eval()
        x = torch.randn(2, 8, 64)
        
        _ = moe_layer(x)
        aux_loss = moe_layer.get_aux_loss()
        
        assert aux_loss == 0.0


class TestMoETransformerBlock:
    """Tests for MoETransformerBlock."""
    
    @pytest.fixture
    def block(self):
        """Create a test transformer block."""
        config = MoEConfig(
            num_experts=4,
            num_experts_per_tok=2,
            hidden_size=64,
            intermediate_size=256
        )
        return MoETransformerBlock(
            hidden_size=64,
            num_heads=4,
            moe_config=config
        )
    
    def test_block_forward(self, block):
        """Test transformer block forward pass."""
        x = torch.randn(2, 8, 64)
        output = block(x)
        assert output.shape == x.shape
    
    def test_block_with_attention_mask(self, block):
        """Test block with attention mask."""
        x = torch.randn(2, 8, 64)
        # Causal mask
        mask = torch.triu(torch.ones(8, 8) * float('-inf'), diagonal=1)
        
        output = block(x, attention_mask=mask)
        assert output.shape == x.shape


class TestMoETransformer:
    """Tests for MoETransformer model."""
    
    @pytest.fixture
    def model(self):
        """Create a small test model."""
        config = MoEConfig(
            num_experts=4,
            num_experts_per_tok=2,
            hidden_size=64,
            intermediate_size=128
        )
        return MoETransformer(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=128,
            moe_config=config,
            moe_every_n_layers=1
        )
    
    def test_model_init(self, model):
        """Test model initialization."""
        assert model.vocab_size == 1000
        assert model.hidden_size == 64
        assert model.num_layers == 2
    
    def test_model_has_embeddings(self, model):
        """Test model has embedding layers."""
        assert isinstance(model.embed_tokens, nn.Embedding)
        assert isinstance(model.embed_positions, nn.Embedding)


class TestMoEGradients:
    """Tests for gradient flow through MoE."""
    
    def test_gradient_flow(self):
        """Test that gradients flow through MoE layer."""
        config = MoEConfig(
            num_experts=4,
            num_experts_per_tok=2,
            hidden_size=64,
            intermediate_size=128
        )
        moe = MoELayer(config)
        moe.train()
        
        x = torch.randn(2, 8, 64, requires_grad=True)
        output = moe(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Check expert gradients
        for expert in moe.experts:
            assert expert.gate_proj.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
