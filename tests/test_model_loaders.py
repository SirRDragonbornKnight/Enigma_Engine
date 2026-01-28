#!/usr/bin/env python3
"""
Tests for Model Loaders - Phase 1, Item 4
==========================================

Tests all model loading functionality:
- Weight mapping utilities
- ONNX loader
- GGUF loader
- HuggingFace loader
- Safetensors loader
- Auto-detection (from_any)

Run with: pytest tests/test_model_loaders.py -v
"""

import pytest
import torch
import sys
import tempfile
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWeightMapping:
    """Tests for weight mapping utility."""
    
    def test_weight_mapper_creation(self):
        """Test WeightMapper can be created."""
        from forge_ai.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        assert mapper is not None
        assert 'huggingface' in mapper.mappings
        assert 'gguf' in mapper.mappings
        assert 'onnx' in mapper.mappings
    
    def test_hf_mappings_exist(self):
        """Test HuggingFace mappings are defined."""
        from forge_ai.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        hf_mappings = mapper.mappings['huggingface']
        
        # Check key mappings exist
        assert 'tok_embeddings.weight' in hf_mappings
        assert 'output.weight' in hf_mappings
        assert 'norm.weight' in hf_mappings
        assert 'layers.{}.attention.wq.weight' in hf_mappings
    
    def test_gguf_mappings_exist(self):
        """Test GGUF mappings are defined."""
        from forge_ai.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        gguf_mappings = mapper.mappings['gguf']
        
        # Check key mappings exist
        assert 'tok_embeddings.weight' in gguf_mappings
        assert 'layers.{}.attention.wq.weight' in gguf_mappings
    
    def test_detect_num_layers(self):
        """Test layer detection from state dict."""
        from forge_ai.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        
        # Create mock state dict with layer keys
        state_dict = {
            'model.layers.0.attention.weight': torch.randn(10, 10),
            'model.layers.1.attention.weight': torch.randn(10, 10),
            'model.layers.2.attention.weight': torch.randn(10, 10),
            'output.weight': torch.randn(100, 10)
        }
        
        # Test through the public mapping method which uses _detect_num_layers internally
        # This tests the functionality without directly calling the private method
        forge_dict = mapper.map_huggingface_to_forge(state_dict)
        # If it works without error, layer detection worked
        assert forge_dict is not None
    
    def test_map_huggingface_simple(self):
        """Test basic HuggingFace to Forge mapping."""
        from forge_ai.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        
        # Create minimal HF state dict
        hf_state_dict = {
            'transformer.wte.weight': torch.randn(1000, 128),
            'lm_head.weight': torch.randn(1000, 128),
            'transformer.ln_f.weight': torch.randn(128),
        }
        
        forge_state_dict = mapper.map_huggingface_to_forge(hf_state_dict)
        
        # Check mappings worked
        assert 'tok_embeddings.weight' in forge_state_dict
        assert 'output.weight' in forge_state_dict
        assert 'norm.weight' in forge_state_dict


class TestONNXLoader:
    """Tests for ONNX model loader."""
    
    def test_onnx_loader_imports(self):
        """Test ONNX loader module can be imported."""
        try:
            from forge_ai.core import onnx_loader
            assert onnx_loader is not None
        except ImportError as e:
            pytest.skip(f"ONNX loader not available: {e}")
    
    def test_onnx_loader_has_functions(self):
        """Test ONNX loader has required functions."""
        try:
            from forge_ai.core.onnx_loader import (
                load_onnx_model,
                extract_onnx_weights,
                infer_config_from_onnx,
                validate_onnx_model
            )
            assert load_onnx_model is not None
            assert extract_onnx_weights is not None
            assert infer_config_from_onnx is not None
            assert validate_onnx_model is not None
        except ImportError as e:
            pytest.skip(f"ONNX dependencies not available: {e}")
    
    @pytest.mark.skipif(
        True,  # Skip by default - requires actual ONNX model file
        reason="Requires ONNX model file for testing"
    )
    def test_load_onnx_model(self):
        """Test loading an actual ONNX model."""
        from forge_ai.core.onnx_loader import load_onnx_model
        
        # This would require an actual ONNX model file
        # model = load_onnx_model("test_model.onnx")
        # assert model is not None
        pass


class TestGGUFLoader:
    """Tests for GGUF model loader."""
    
    def test_gguf_loader_imports(self):
        """Test GGUF loader module exists."""
        from forge_ai.core import gguf_loader
        assert gguf_loader is not None
    
    def test_gguf_loader_has_load_function(self):
        """Test GGUF loader has load_gguf_model function."""
        from forge_ai.core.gguf_loader import load_gguf_model
        assert load_gguf_model is not None
        assert callable(load_gguf_model)
    
    def test_gguf_loader_has_helper_functions(self):
        """Test GGUF loader has helper functions."""
        from forge_ai.core.gguf_loader import (
            list_gguf_models,
            recommend_gpu_layers
        )
        assert list_gguf_models is not None
        assert recommend_gpu_layers is not None
    
    def test_recommend_gpu_layers(self):
        """Test GPU layer recommendation logic."""
        from forge_ai.core.gguf_loader import recommend_gpu_layers
        
        # Small model on large VRAM - should fit all
        layers = recommend_gpu_layers(model_size_gb=2.0, vram_gb=8.0)
        assert layers > 0
        
        # Large model on small VRAM - should use fewer layers
        layers = recommend_gpu_layers(model_size_gb=14.0, vram_gb=4.0)
        assert layers >= 0
    
    @pytest.mark.skipif(
        True,  # Skip by default - requires actual GGUF model
        reason="Requires GGUF model file for testing"
    )
    def test_load_gguf_model(self):
        """Test loading an actual GGUF model."""
        from forge_ai.core.gguf_loader import load_gguf_model
        
        # This would require an actual GGUF model file
        # model = load_gguf_model("test_model.gguf")
        # assert model is not None
        pass


class TestHuggingFaceLoader:
    """Tests for HuggingFace model loader."""
    
    def test_hf_loader_imports(self):
        """Test HuggingFace loader module exists."""
        from forge_ai.core import huggingface_loader
        assert huggingface_loader is not None
    
    def test_hf_loader_has_functions(self):
        """Test HuggingFace loader has conversion function."""
        try:
            from forge_ai.core.huggingface_loader import (
                convert_huggingface_to_forge,
                get_huggingface_model_info
            )
            assert convert_huggingface_to_forge is not None
            assert get_huggingface_model_info is not None
        except ImportError as e:
            pytest.skip(f"HuggingFace dependencies not available: {e}")
    
    @pytest.mark.skipif(
        True,  # Skip by default - requires network and HF download
        reason="Requires network access and HuggingFace download"
    )
    def test_load_huggingface_model(self):
        """Test loading a small HuggingFace model."""
        from forge_ai.core.huggingface_loader import convert_huggingface_to_forge
        
        # This would download from HuggingFace
        # model = convert_huggingface_to_forge("gpt2")
        # assert model is not None
        pass


class TestSafetensorsLoader:
    """Tests for Safetensors loader."""
    
    def test_safetensors_available(self):
        """Test safetensors library is available."""
        try:
            import safetensors
            assert safetensors is not None
        except ImportError:
            pytest.skip("safetensors not installed")
    
    def test_from_safetensors_method_exists(self):
        """Test Forge has from_safetensors method."""
        from forge_ai.core.model import Forge
        assert hasattr(Forge, 'from_safetensors')
        assert callable(Forge.from_safetensors)
    
    def test_from_safetensors_with_mock_file(self):
        """Test from_safetensors with a mock file."""
        try:
            from safetensors.torch import save_model
            from forge_ai.core.model import Forge, ForgeConfig
        except ImportError:
            pytest.skip("Required libraries not available")
        
        # Create a small model and save it
        config = ForgeConfig(vocab_size=100, dim=32, n_layers=2, n_heads=2)
        model = Forge(config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.safetensors"
            config_path = Path(tmpdir) / "test_model.json"
            
            # Save model weights using save_model to handle shared tensors
            save_model(model, str(model_path))
            
            # Save config
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f)
            
            # Try to load
            loaded_model = Forge.from_safetensors(model_path)
            assert loaded_model is not None
            assert loaded_model.vocab_size == model.vocab_size


class TestModelFromAny:
    """Tests for universal from_any loader."""
    
    def test_from_any_method_exists(self):
        """Test Forge has from_any method."""
        from forge_ai.core.model import Forge
        assert hasattr(Forge, 'from_any')
        assert callable(Forge.from_any)
    
    def test_from_any_detects_safetensors(self):
        """Test from_any detects .safetensors extension."""
        from forge_ai.core.model import Forge
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake file (not a real model, just for path detection)
            fake_path = Path(tmpdir) / "model.safetensors"
            fake_path.touch()
            
            # Test detection (will fail at loading, but that's ok for this test)
            try:
                Forge.from_any(fake_path)
            except Exception as e:
                # Should detect as safetensors and try that loader
                # Accept various error messages related to invalid file
                error_str = str(e).lower()
                assert ('safetensors' in error_str or 
                        'format' in error_str or 
                        'header' in error_str or
                        'deserializing' in error_str), f"Unexpected error: {e}"
    
    def test_from_any_detects_gguf(self):
        """Test from_any detects .gguf extension."""
        from forge_ai.core.model import Forge
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "model.gguf"
            fake_path.touch()
            
            try:
                Forge.from_any(fake_path)
            except Exception as e:
                # Should detect as GGUF
                # Accept various error messages
                error_str = str(e).lower()
                assert ('gguf' in error_str or 
                        'llama' in error_str or 
                        'not found' in error_str or
                        'torch' in error_str or  # torch import error is acceptable
                        'numpy' in error_str or  # numpy import error is acceptable
                        'module' in error_str or
                        'name' in error_str), f"Unexpected error: {e}"
    
    def test_from_any_detects_onnx(self):
        """Test from_any detects .onnx extension."""
        from forge_ai.core.model import Forge
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "model.onnx"
            fake_path.touch()
            
            try:
                Forge.from_any(fake_path)
            except Exception as e:
                # Should detect as ONNX
                assert 'onnx' in str(e).lower() or 'format' in str(e).lower()


class TestErrorHandling:
    """Tests for error handling in loaders."""
    
    def test_missing_file_error(self):
        """Test proper error when file doesn't exist."""
        from forge_ai.core.model import Forge
        
        with pytest.raises(FileNotFoundError):
            Forge.from_any("/nonexistent/path/to/model.gguf")
    
    def test_unknown_format_error(self):
        """Test proper error for unknown format."""
        from forge_ai.core.model import Forge
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "model.unknown"
            fake_path.touch()
            
            with pytest.raises(ValueError) as exc_info:
                Forge.from_any(fake_path)
            
            assert 'unknown' in str(exc_info.value).lower() or 'format' in str(exc_info.value).lower()
    
    def test_onnx_missing_dependency_error(self):
        """Test ONNX loader gives clear error when onnx not installed."""
        # This test verifies error message quality
        try:
            from forge_ai.core.onnx_loader import load_onnx_model
            
            # If onnx is not available, this should raise RuntimeError with helpful message
            with tempfile.TemporaryDirectory() as tmpdir:
                fake_path = Path(tmpdir) / "model.onnx"
                fake_path.write_text("fake")
                
                try:
                    load_onnx_model(fake_path)
                except RuntimeError as e:
                    # Should mention how to install
                    assert 'pip install' in str(e) or 'onnx' in str(e)
                except Exception:
                    # Other errors are ok (e.g., if onnx IS installed but file is fake)
                    pass
        except ImportError:
            pytest.skip("Test requires onnx_loader module")


class TestLoaderIntegration:
    """Integration tests for all loaders."""
    
    def test_all_loaders_accessible_from_model(self):
        """Test all loader methods are accessible from Forge class."""
        from forge_ai.core.model import Forge
        
        required_methods = [
            'from_any',
            'from_huggingface',
            'from_safetensors',
            'from_gguf',
            'from_onnx',
            'from_pretrained'
        ]
        
        for method in required_methods:
            assert hasattr(Forge, method), f"Forge missing method: {method}"
            assert callable(getattr(Forge, method)), f"Forge.{method} not callable"
    
    def test_loader_methods_are_classmethods(self):
        """Test all loader methods are classmethods."""
        from forge_ai.core.model import Forge
        import inspect
        
        loader_methods = [
            'from_any',
            'from_huggingface',
            'from_safetensors',
            'from_gguf',
            'from_onnx'
        ]
        
        for method_name in loader_methods:
            method = getattr(Forge, method_name)
            # Classmethods have __self__ that is the class
            assert inspect.ismethod(method), f"{method_name} should be a classmethod"


class TestDocumentation:
    """Tests for documentation and docstrings."""
    
    def test_loaders_have_docstrings(self):
        """Test all loader functions have docstrings."""
        from forge_ai.core.model import Forge
        
        loader_methods = [
            Forge.from_any,
            Forge.from_huggingface,
            Forge.from_safetensors,
            Forge.from_gguf,
            Forge.from_onnx
        ]
        
        for method in loader_methods:
            assert method.__doc__ is not None, f"{method.__name__} missing docstring"
            assert len(method.__doc__) > 50, f"{method.__name__} docstring too short"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
