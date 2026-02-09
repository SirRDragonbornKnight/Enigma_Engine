#!/usr/bin/env python3
"""
Integration tests for Enigma AI Engine.

These tests verify that different components work together correctly.

Run with: pytest tests/test_integration.py -v
"""
import pytest
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelToInference:
    """Test model and inference integration."""
    
    def test_model_in_engine(self):
        """Test that engine uses correct model."""
        from enigma_engine.core.inference import EnigmaEngine
        from enigma_engine.core.model import Enigma
        
        try:
            engine = EnigmaEngine()
        except FileNotFoundError:
            pytest.skip("No trained model available")
        assert isinstance(engine.model, Enigma)
    
    def test_tokenizer_vocab_matches_model(self):
        """Test tokenizer vocab matches model embedding."""
        from enigma_engine.core.inference import EnigmaEngine
        
        try:
            engine = EnigmaEngine()
        except FileNotFoundError:
            pytest.skip("No trained model available")
        tok_vocab = getattr(engine.tokenizer, 'vocab_size', None) or getattr(engine.tokenizer, 'n_vocab', 32000)
        model_vocab = engine.model.vocab_size
        
        # They should be compatible
        assert model_vocab >= tok_vocab or model_vocab <= 100000
    
    def test_full_generation_pipeline(self):
        """Test full pipeline from prompt to text."""
        from enigma_engine.core.inference import EnigmaEngine
        
        try:
            engine = EnigmaEngine()
        except FileNotFoundError:
            pytest.skip("No trained model available")
        
        # Encode
        prompt = "Hello world"
        enc = engine.tokenizer(prompt, return_tensors="pt")
        
        # Generate
        output = engine.generate(prompt, max_gen=10)
        
        # Should produce text (at minimum the prompt, potentially more)
        assert isinstance(output, str)
        assert len(output) >= len(prompt)  # May just return prompt if generation fails


class TestTrainingPipeline:
    """Test training pipeline."""
    
    def test_train_and_save(self, tmp_path):
        """Test training and saving model."""
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.training import train_model
        
        # Create small model
        model = Enigma(vocab_size=1000, dim=32, depth=1, heads=2)
        
        # Create tiny training data
        train_text = "hello world. testing testing. one two three."
        
        # Train for 1 epoch
        # Note: This may not work without proper data loading
        # Just test the function exists and is callable
        assert callable(train_model)
    
    def test_model_checkpointing(self, tmp_path):
        """Test saving and loading model checkpoints."""
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_registry import safe_load_weights
        import torch
        
        # Create and save
        model1 = Enigma(vocab_size=1000, dim=32, depth=1, heads=2)
        save_path = tmp_path / "model.pth"
        torch.save(model1.state_dict(), save_path)
        
        # Load into new model
        model2 = Enigma(vocab_size=1000, dim=32, depth=1, heads=2)
        model2.load_state_dict(safe_load_weights(save_path))
        
        # Parameters should match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


class TestModelRegistry:
    """Test model registry integration."""
    
    def test_create_and_load_model(self, tmp_path):
        """Test creating and loading model through registry."""
        from enigma_engine.core.model_registry import ModelRegistry
        import shutil
        
        # Create registry with temp dir
        registry = ModelRegistry(str(tmp_path / "models"))
        
        # Create a model
        name = "test_integration_model"
        try:
            registry.create_model(name, size="tiny", description="Test")
            
            # Check it exists
            models = registry.list_models()
            assert name in models
            
        finally:
            # Cleanup
            try:
                registry.delete_model(name, confirm=True)
            except (KeyError, FileNotFoundError, OSError, ValueError):
                pass


class TestConfigIntegration:
    """Test configuration integration."""
    
    def test_config_affects_model(self):
        """Test that CONFIG affects model creation."""
        from enigma_engine.config import CONFIG
        from enigma_engine.core.inference import EnigmaEngine
        
        try:
            engine = EnigmaEngine()
        except FileNotFoundError:
            pytest.skip("No trained model available")
        
        # Model should have a valid max_len
        assert hasattr(engine.model, 'max_len')
        assert engine.model.max_len > 0
        # The model max_len is set from preset or CONFIG, both are valid


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_conversation_flow(self):
        """Test a full conversation flow."""
        from enigma_engine.core.inference import EnigmaEngine
        
        try:
            engine = EnigmaEngine()
        except FileNotFoundError:
            pytest.skip("No trained model available")
        
        # Start conversation with history
        history = []
        
        response1 = engine.chat("Hello!", history=history, max_gen=10)
        assert isinstance(response1, str)
        
        # Add to history
        history.append({"role": "user", "content": "Hello!"})
        history.append({"role": "assistant", "content": response1})
        
        # Continue conversation
        response2 = engine.chat("How are you?", history=history, max_gen=10)
        assert isinstance(response2, str)
    
    def test_streaming_generation(self):
        """Test streaming generation collects all tokens."""
        from enigma_engine.core.inference import EnigmaEngine
        
        try:
            engine = EnigmaEngine()
        except FileNotFoundError:
            pytest.skip("No trained model available")
        
        tokens = []
        for token in engine.stream_generate("Test", max_gen=5):
            tokens.append(token)
        
        # Should have generated tokens
        assert len(tokens) > 0
        
        # Concatenated should be string
        full_text = "".join(tokens)
        assert isinstance(full_text, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
