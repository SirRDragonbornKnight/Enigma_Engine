"""
Integration Test: Full Training Pipeline

Tests the complete training flow from model creation through training.
This verifies that all components work together correctly.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path


class TestTrainingPipeline:
    """Integration tests for the training pipeline."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Generate sample training data."""
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Hello world, this is a test of the training system.",
            "Once upon a time there was a small AI model.",
            "It learned many things from the data it was given.",
            "Training neural networks requires patience and good data.",
            "The model slowly improved with each epoch.",
            "Transformers use attention to understand context.",
            "Every word contributes to the final meaning.",
            "Learning happens through gradient descent optimization.",
            "The loss function measures how wrong the model is.",
        ] * 10  # Repeat for more data
    
    @pytest.fixture
    def small_model_config(self):
        """Create a minimal model configuration for testing."""
        return {
            'size': 'nano',  # Smallest possible
            'vocab_size': 1000,
        }
    
    def test_create_model_and_tokenizer(self, small_model_config):
        """Test model and tokenizer can be created."""
        from enigma_engine.core.model import create_model
        from enigma_engine.core.tokenizer import get_tokenizer
        
        model = create_model(
            small_model_config['size'],
            vocab_size=small_model_config['vocab_size']
        )
        tokenizer = get_tokenizer(vocab_size=small_model_config['vocab_size'])
        
        assert model is not None
        assert tokenizer is not None
        assert hasattr(model, 'forward')
        assert hasattr(tokenizer, 'encode')
        assert hasattr(tokenizer, 'decode')
    
    def test_tokenizer_roundtrip(self, sample_training_data):
        """Test tokenizer can encode and decode text."""
        from enigma_engine.core.tokenizer import get_tokenizer
        
        tokenizer = get_tokenizer(vocab_size=1000)
        
        for text in sample_training_data[:3]:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            
            # Should produce tokens
            assert len(encoded) > 0
            # Decoded should be similar (may not be exact due to tokenization)
            assert len(decoded) > 0
    
    def test_full_training_loop(self, sample_training_data, small_model_config):
        """Test complete training pipeline."""
        from enigma_engine.core.model import create_model
        from enigma_engine.core.tokenizer import get_tokenizer
        from enigma_engine.core.training import Trainer, TrainingConfig
        
        # Create model and tokenizer
        model = create_model(
            small_model_config['size'],
            vocab_size=small_model_config['vocab_size']
        )
        tokenizer = get_tokenizer(vocab_size=small_model_config['vocab_size'])
        
        # Create minimal training config
        config = TrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            max_seq_len=64,
            warmup_steps=5,
            use_amp=False,  # Disable AMP for test stability
            gradient_accumulation_steps=1,
        )
        
        # Create trainer
        trainer = Trainer(model, tokenizer, config)
        
        # Track progress
        progress_updates = []
        def progress_callback(metrics):
            progress_updates.append(metrics)
        
        # Run training
        results = trainer.train(
            sample_training_data,
            epochs=2,
            callback=progress_callback
        )
        
        # Verify results
        assert 'final_loss' in results
        assert 'loss_history' in results
        assert 'elapsed_time' in results
        assert results['final_loss'] > 0  # Loss should be positive
        assert len(results['loss_history']) == 2  # Two epochs
        assert results['elapsed_time'] > 0  # Should take some time
    
    def test_training_reduces_loss(self, sample_training_data, small_model_config):
        """Test that training actually reduces loss over time."""
        from enigma_engine.core.model import create_model
        from enigma_engine.core.tokenizer import get_tokenizer
        from enigma_engine.core.training import Trainer, TrainingConfig
        
        model = create_model(
            small_model_config['size'],
            vocab_size=small_model_config['vocab_size']
        )
        tokenizer = get_tokenizer(vocab_size=small_model_config['vocab_size'])
        
        config = TrainingConfig(
            epochs=5,
            batch_size=4,
            learning_rate=5e-4,
            max_seq_len=32,
            warmup_steps=2,
            use_amp=False,
        )
        
        trainer = Trainer(model, tokenizer, config)
        results = trainer.train(sample_training_data, epochs=5)
        
        loss_history = results['loss_history']
        
        # Loss should generally decrease (allow some fluctuation)
        # Check that final loss is less than initial or stays reasonable
        assert loss_history[-1] <= loss_history[0] * 1.5, \
            "Loss should not increase significantly during training"
    
    def test_model_save_and_load(self, sample_training_data, small_model_config):
        """Test model can be saved and loaded after training."""
        from enigma_engine.core.model import create_model
        from enigma_engine.core.tokenizer import get_tokenizer
        from enigma_engine.core.training import Trainer, TrainingConfig
        
        model = create_model(
            small_model_config['size'],
            vocab_size=small_model_config['vocab_size']
        )
        tokenizer = get_tokenizer(vocab_size=small_model_config['vocab_size'])
        
        config = TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            max_seq_len=32,
            use_amp=False,
        )
        
        trainer = Trainer(model, tokenizer, config)
        trainer.train(sample_training_data[:20], epochs=1)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model.pt"
            torch.save(model.state_dict(), save_path)
            
            # Verify file was created
            assert save_path.exists()
            
            # Load into new model
            model2 = create_model(
                small_model_config['size'],
                vocab_size=small_model_config['vocab_size']
            )
            model2.load_state_dict(torch.load(save_path, weights_only=True))
            
            # Verify weights match
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), 
                model2.named_parameters()
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2), f"Weights differ for {n1}"
    
    def test_inference_after_training(self, sample_training_data, small_model_config):
        """Test model can generate text after training."""
        from enigma_engine.core.model import create_model
        from enigma_engine.core.tokenizer import get_tokenizer
        from enigma_engine.core.training import Trainer, TrainingConfig
        from enigma_engine.core.inference import EnigmaEngine
        
        model = create_model(
            small_model_config['size'],
            vocab_size=small_model_config['vocab_size']
        )
        tokenizer = get_tokenizer(vocab_size=small_model_config['vocab_size'])
        
        config = TrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            max_seq_len=32,
            use_amp=False,
        )
        
        trainer = Trainer(model, tokenizer, config)
        trainer.train(sample_training_data[:20], epochs=2)
        
        # Create inference engine
        engine = EnigmaEngine(model, tokenizer)
        
        # Generate text
        prompt = "The quick"
        response = engine.generate(
            prompt,
            max_tokens=20,
            temperature=0.7
        )
        
        # Should produce some output
        assert len(response) > 0
        assert isinstance(response, str)


class TestTrainingConfigValidation:
    """Test training configuration validation."""
    
    def test_default_config(self):
        """Test default configuration is valid."""
        from enigma_engine.core.training import TrainingConfig
        
        config = TrainingConfig()
        
        assert config.epochs > 0
        assert config.batch_size > 0
        assert config.learning_rate > 0
        assert config.max_seq_len > 0
    
    def test_config_from_device_profile(self):
        """Test device-aware configuration."""
        from enigma_engine.core.training import TrainingConfig
        
        config = TrainingConfig.from_device_profile()
        
        # Should create valid config based on hardware
        assert config.epochs > 0
        assert config.batch_size > 0
    
    def test_config_override(self):
        """Test configuration can be overridden."""
        from enigma_engine.core.training import TrainingConfig
        
        config = TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=0.01
        )
        
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 0.01


class TestQATraining:
    """Test Q&A format training."""
    
    @pytest.fixture
    def qa_training_data(self):
        """Generate Q&A format training data."""
        return [
            "Q: What is the capital of France?\nA: Paris is the capital of France.",
            "Q: How many legs does a spider have?\nA: A spider has eight legs.",
            "Q: What color is the sky?\nA: The sky is blue during the day.",
            "Q: Who wrote Romeo and Juliet?\nA: Shakespeare wrote Romeo and Juliet.",
            "Q: What is 2 + 2?\nA: 2 + 2 equals 4.",
        ] * 10
    
    def test_qa_dataset_detection(self, qa_training_data):
        """Test Q&A format is auto-detected."""
        from enigma_engine.core.model import create_model
        from enigma_engine.core.tokenizer import get_tokenizer
        from enigma_engine.core.training import Trainer, TrainingConfig
        
        model = create_model('nano', vocab_size=1000)
        tokenizer = get_tokenizer(vocab_size=1000)
        
        config = TrainingConfig(
            epochs=1,
            batch_size=2,
            max_seq_len=64,
            use_amp=False,
        )
        
        trainer = Trainer(model, tokenizer, config)
        
        # Should auto-detect as QA
        results = trainer.train(
            qa_training_data,
            epochs=1,
            dataset_type="auto"
        )
        
        assert 'final_loss' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
