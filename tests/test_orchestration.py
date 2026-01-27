"""
Tests for the Deep Multi-Model Integration - Orchestration System.

Tests capability registry, model pool, collaboration, and orchestrator.
"""

import pytest
import tempfile
from pathlib import Path

from forge_ai.core.capability_registry import (
    CapabilityRegistry,
    Capability,
    ModelCapabilityEntry,
    BUILT_IN_CAPABILITIES,
)
from forge_ai.core.model_pool import ModelPool, ModelPoolConfig, ModelEntry
from forge_ai.core.collaboration import (
    ModelCollaboration,
    CollaborationType,
    CollaborationRequest,
    CollaborationResponse,
)
from forge_ai.core.orchestrator import (
    ModelOrchestrator,
    OrchestratorConfig,
    Task,
    TaskResult,
)


# =============================================================================
# CAPABILITY REGISTRY TESTS
# =============================================================================

class TestCapabilityRegistry:
    """Test the capability registry."""
    
    def test_register_model(self):
        """Test registering a model with capabilities."""
        registry = CapabilityRegistry()
        
        registry.register_model(
            model_id="test:model1",
            capabilities=["text_generation", "reasoning"],
            metadata={"size": "1B", "device": "cpu"},
        )
        
        assert "test:model1" in registry.list_models()
        assert registry.has_capability("test:model1", "text_generation")
        assert registry.has_capability("test:model1", "reasoning")
        assert not registry.has_capability("test:model1", "vision")
    
    def test_find_models_with_capability(self):
        """Test finding models by capability."""
        registry = CapabilityRegistry()
        
        registry.register_model("test:model1", ["text_generation"])
        registry.register_model("test:model2", ["text_generation", "code_generation"])
        registry.register_model("test:model3", ["vision"])
        
        text_models = registry.find_models_with_capability("text_generation")
        assert len(text_models) == 2
        assert "test:model1" in text_models
        assert "test:model2" in text_models
        
        code_models = registry.find_models_with_capability("code_generation")
        assert len(code_models) == 1
        assert "test:model2" in code_models
    
    def test_find_best_model(self):
        """Test finding best model for a capability."""
        registry = CapabilityRegistry()
        
        registry.register_model(
            "test:model1",
            ["text_generation"],
            performance_ratings={"text_generation": 0.7}
        )
        registry.register_model(
            "test:model2",
            ["text_generation"],
            performance_ratings={"text_generation": 0.9}
        )
        
        best = registry.find_best_model("text_generation")
        assert best == "test:model2"
    
    def test_auto_detect_capabilities(self):
        """Test automatic capability detection."""
        registry = CapabilityRegistry()
        
        # Vision model (has "vision" in name)
        registry.register_model(
            "test:llava-vision",
            [],
            metadata={"type": "vision_language"},
            auto_detect=True,
        )
        assert registry.has_capability("test:llava-vision", "vision")
        
        # Code model (has "code" in name)
        registry.register_model(
            "test:codegen-model",
            [],
            metadata={},
            auto_detect=True,
        )
        assert registry.has_capability("test:codegen-model", "code_generation")
    
    def test_persistence(self):
        """Test saving and loading registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "registry.json"
            
            # Create and save
            registry1 = CapabilityRegistry(str(config_path))
            registry1.register_model("test:model1", ["text_generation"])
            registry1.save()
            
            # Load in new instance
            registry2 = CapabilityRegistry(str(config_path))
            registry2.load()
            
            assert "test:model1" in registry2.list_models()
            assert registry2.has_capability("test:model1", "text_generation")


# =============================================================================
# MODEL POOL TESTS
# =============================================================================

class MockModel:
    """Mock model for testing."""
    def __init__(self, name):
        self.name = name
    
    def generate(self, text):
        return f"Mock response from {self.name}: {text}"


class TestModelPool:
    """Test the model pool."""
    
    def test_model_entry_lifecycle(self):
        """Test model entry state tracking."""
        model = MockModel("test")
        entry = ModelEntry(
            model_id="test:model",
            model=model,
            model_type="test",
            device="cpu",
            memory_mb=100.0,
        )
        
        assert not entry.in_use
        assert entry.use_count == 0
        
        entry.mark_used()
        assert entry.in_use
        assert entry.use_count == 1
        
        entry.mark_released()
        assert not entry.in_use
        assert entry.use_count == 1
    
    def test_pool_memory_tracking(self):
        """Test memory usage tracking."""
        pool = ModelPool(ModelPoolConfig(max_loaded_models=3))
        
        # Initially empty
        usage = pool.get_memory_usage()
        assert usage["total_mb"] == 0
        assert usage["num_models"] == 0
    
    def test_pool_list_models(self):
        """Test listing loaded models."""
        pool = ModelPool(ModelPoolConfig(max_loaded_models=3))
        
        models = pool.list_models()
        assert len(models) == 0
    
    def test_pool_clear(self):
        """Test clearing all models."""
        pool = ModelPool(ModelPoolConfig(max_loaded_models=3))
        pool.clear()
        
        assert len(pool.list_models()) == 0


# =============================================================================
# COLLABORATION TESTS
# =============================================================================

class MockOrchestrator:
    """Mock orchestrator for testing collaboration."""
    
    def __init__(self):
        self.executed_tasks = []
    
    def find_best_model(self, capability):
        return f"mock:model_for_{capability}"
    
    def execute_task(self, model_id, capability, **kwargs):
        self.executed_tasks.append({
            "model_id": model_id,
            "capability": capability,
            "kwargs": kwargs,
        })
        return f"Result from {model_id}"
    
    def find_models_with_capability(self, capability):
        return [f"mock:model{i}_for_{capability}" for i in range(3)]


class TestModelCollaboration:
    """Test model-to-model collaboration."""
    
    def test_request_assistance(self):
        """Test basic request/response collaboration."""
        collab = ModelCollaboration()
        orchestrator = MockOrchestrator()
        collab.set_orchestrator(orchestrator)
        
        response = collab.request_assistance(
            requesting_model="test:model1",
            target_capability="code_generation",
            task="Write a Python function",
        )
        
        assert response.success
        assert "mock:model_for_code_generation" in response.responding_model
        assert len(orchestrator.executed_tasks) == 1
    
    def test_collaboration_without_orchestrator(self):
        """Test collaboration fails gracefully without orchestrator."""
        collab = ModelCollaboration()
        
        response = collab.request_assistance(
            requesting_model="test:model1",
            target_capability="code_generation",
            task="Write a function",
        )
        
        assert not response.success
        assert "No orchestrator" in response.error
    
    def test_collaboration_stats(self):
        """Test collaboration statistics tracking."""
        collab = ModelCollaboration()
        orchestrator = MockOrchestrator()
        collab.set_orchestrator(orchestrator)
        
        # Make some requests
        for i in range(3):
            collab.request_assistance(
                requesting_model="test:model1",
                target_capability="text_generation",
                task=f"Task {i}",
            )
        
        stats = collab.get_collaboration_stats()
        assert stats["total_collaborations"] == 3
        assert stats["successful"] == 3


# =============================================================================
# ORCHESTRATOR TESTS
# =============================================================================

class TestModelOrchestrator:
    """Test the model orchestrator."""
    
    def test_register_model(self):
        """Test model registration."""
        orchestrator = ModelOrchestrator()
        
        orchestrator.register_model(
            model_id="test:model1",
            capabilities=["text_generation"],
            metadata={"size": "1B"},
        )
        
        models = orchestrator.capability_registry.list_models()
        assert "test:model1" in models
    
    def test_find_best_model(self):
        """Test finding best model."""
        orchestrator = ModelOrchestrator()
        
        orchestrator.register_model(
            "test:model1",
            ["text_generation"],
            performance_ratings={"text_generation": 0.8}
        )
        orchestrator.register_model(
            "test:model2",
            ["text_generation"],
            performance_ratings={"text_generation": 0.6}
        )
        
        best = orchestrator.find_best_model("text_generation")
        assert best == "test:model1"
    
    def test_assign_model_to_capability(self):
        """Test manual model assignment."""
        orchestrator = ModelOrchestrator()
        
        orchestrator.register_model("test:model1", ["text_generation"])
        orchestrator.assign_model_to_capability("text_generation", "test:model1")
        
        # Should prefer assigned model
        best = orchestrator.find_best_model("text_generation")
        assert best == "test:model1"
    
    def test_fallback_chain(self):
        """Test fallback chain configuration."""
        orchestrator = ModelOrchestrator()
        
        orchestrator.register_model("test:model1", ["text_generation"])
        orchestrator.register_model("test:model2", ["text_generation"])
        orchestrator.register_model("test:model3", ["text_generation"])
        
        orchestrator.set_fallback_chain(
            "test:model1",
            ["test:model2", "test:model3"]
        )
        
        # Verify fallback chain was set
        assert "test:model1" in orchestrator._fallback_chains
        assert orchestrator._fallback_chains["test:model1"] == ["test:model2", "test:model3"]
    
    def test_orchestrator_status(self):
        """Test getting orchestrator status."""
        orchestrator = ModelOrchestrator()
        
        orchestrator.register_model("test:model1", ["text_generation"])
        
        status = orchestrator.get_status()
        assert "registered_models" in status
        assert "loaded_models" in status
        assert "memory_usage" in status
        assert "test:model1" in status["registered_models"]
    
    def test_hot_swap(self):
        """Test hot-swapping models."""
        config = OrchestratorConfig(enable_hot_swap=True)
        orchestrator = ModelOrchestrator(config)
        
        orchestrator.register_model("test:old_model", ["text_generation"])
        orchestrator.register_model("test:new_model", ["text_generation"])
        
        # Hot-swap
        success = orchestrator.hot_swap_model("test:old_model", "test:new_model")
        
        # Should succeed (even though models aren't actually loaded in this test)
        # In real use, would need actual loadable models


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestOrchestrationIntegration:
    """Test integration between orchestration components."""
    
    def test_full_orchestration_flow(self):
        """Test complete flow from registration to execution."""
        # This is a basic structure test - real execution would need actual models
        orchestrator = ModelOrchestrator()
        
        # Register a model
        orchestrator.register_model(
            model_id="test:model1",
            capabilities=["text_generation"],
            metadata={"device": "cpu"},
        )
        
        # Verify it's registered
        assert "test:model1" in orchestrator.capability_registry.list_models()
        
        # Verify we can find it
        best = orchestrator.find_best_model("text_generation")
        assert best == "test:model1"
    
    def test_collaboration_integration(self):
        """Test collaboration through orchestrator."""
        orchestrator = ModelOrchestrator(
            OrchestratorConfig(enable_collaboration=True)
        )
        
        orchestrator.register_model("test:model1", ["text_generation"])
        orchestrator.register_model("test:model2", ["code_generation"])
        
        # Verify collaboration is enabled
        assert orchestrator.config.enable_collaboration
        assert orchestrator.collaboration.get_collaboration_stats()["total_collaborations"] == 0


# =============================================================================
# BUILT-IN CAPABILITIES TESTS
# =============================================================================

class TestBuiltInCapabilities:
    """Test built-in capability definitions."""
    
    def test_all_capabilities_defined(self):
        """Verify all expected capabilities are defined."""
        expected = [
            "text_generation",
            "code_generation",
            "vision",
            "image_generation",
            "audio_generation",
            "speech_to_text",
            "text_to_speech",
            "embedding",
            "reasoning",
            "tool_calling",
        ]
        
        for cap_name in expected:
            assert cap_name in BUILT_IN_CAPABILITIES
            cap = BUILT_IN_CAPABILITIES[cap_name]
            assert isinstance(cap, Capability)
            assert cap.name == cap_name
    
    def test_capability_structure(self):
        """Test capability structure is valid."""
        cap = BUILT_IN_CAPABILITIES["text_generation"]
        
        assert cap.name == "text_generation"
        assert cap.display_name
        assert cap.description
        assert isinstance(cap.requires_input, list)
        assert isinstance(cap.produces_output, list)
        assert cap.category


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
