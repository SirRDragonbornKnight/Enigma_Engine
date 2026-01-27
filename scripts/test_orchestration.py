#!/usr/bin/env python3
"""
Orchestration System - Quick Validation Test

Tests the orchestration system without requiring PyTorch or other heavy dependencies.
This directly imports the orchestration modules to validate they work.
"""

import sys
from pathlib import Path

print("="*70)
print("ORCHESTRATION SYSTEM - VALIDATION TEST")
print("="*70)

print("\n1. Testing direct module imports...")

# Test importing orchestration modules directly
try:
    # These imports should work without PyTorch
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Import just the orchestration components
    from forge_ai.core.capability_registry import (
        CapabilityRegistry, Capability, BUILT_IN_CAPABILITIES
    )
    from forge_ai.core.collaboration import ModelCollaboration
    from forge_ai.core.standalone_tools import list_available_tools
    
    print("   ✓ Direct imports successful!")
    
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

print("\n2. Testing Capability Registry...")
try:
    registry = CapabilityRegistry()
    
    # Register test models
    registry.register_model(
        "test:model1",
        ["text_generation", "reasoning"],
        metadata={"size": "27M"}
    )
    
    registry.register_model(
        "test:model2",
        ["code_generation"],
        metadata={"size": "85M"},
        performance_ratings={"code_generation": 0.9}
    )
    
    # Test queries
    models = registry.list_models()
    assert len(models) == 2, f"Expected 2 models, got {len(models)}"
    
    assert registry.has_capability("test:model1", "text_generation")
    assert registry.has_capability("test:model2", "code_generation")
    
    text_models = registry.find_models_with_capability("text_generation")
    assert "test:model1" in text_models
    
    best_code = registry.find_best_model("code_generation")
    assert best_code == "test:model2"
    
    print(f"   ✓ Registry working! {len(models)} models registered")
    print(f"   ✓ Capabilities: {len(BUILT_IN_CAPABILITIES)} built-in")
    
except Exception as e:
    print(f"   ✗ Registry test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing Collaboration...")
try:
    collaboration = ModelCollaboration()
    
    # Test statistics
    stats = collaboration.get_collaboration_stats()
    assert stats['total_collaborations'] == 0
    
    print("   ✓ Collaboration working!")
    print(f"   ✓ Stats: {stats}")
    
except Exception as e:
    print(f"   ✗ Collaboration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Testing Standalone Tools...")
try:
    tools = list_available_tools()
    assert len(tools) > 0, "No tools found"
    
    from forge_ai.core.standalone_tools import get_tool_info
    
    # Test a few tool infos
    for tool_name in ['image', 'vision', 'code']:
        info = get_tool_info(tool_name)
        assert 'description' in info, f"Tool {tool_name} missing description"
        assert 'parameters' in info, f"Tool {tool_name} missing parameters"
    
    print(f"   ✓ Standalone tools working! {len(tools)} tools available")
    print(f"   ✓ Tools: {', '.join(tools[:5])}...")
    
except Exception as e:
    print(f"   ✗ Standalone tools test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. Testing Capability Auto-Detection...")
try:
    registry2 = CapabilityRegistry()
    
    # Test auto-detection with model names
    registry2.register_model(
        "test:llava-vision-model",
        [],  # Empty - should auto-detect
        auto_detect=True
    )
    
    assert registry2.has_capability("test:llava-vision-model", "vision")
    print("   ✓ Auto-detected 'vision' from model name")
    
    registry2.register_model(
        "test:codegen-python",
        [],
        auto_detect=True
    )
    
    assert registry2.has_capability("test:codegen-python", "code_generation")
    print("   ✓ Auto-detected 'code_generation' from model name")
    
except Exception as e:
    print(f"   ✗ Auto-detection test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n6. Testing Performance Ratings...")
try:
    registry3 = CapabilityRegistry()
    
    # Register multiple models with different ratings
    registry3.register_model(
        "model:weak",
        ["text_generation"],
        performance_ratings={"text_generation": 0.5}
    )
    
    registry3.register_model(
        "model:strong",
        ["text_generation"],
        performance_ratings={"text_generation": 0.9}
    )
    
    best = registry3.find_best_model("text_generation")
    assert best == "model:strong", f"Expected model:strong, got {best}"
    
    print("   ✓ Performance ratings working correctly")
    
except Exception as e:
    print(f"   ✗ Performance ratings test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL ORCHESTRATION TESTS PASSED!")
print("="*70)

print("\n📦 Validated Components:")
print("   ✓ forge_ai/core/capability_registry.py")
print("   ✓ forge_ai/core/collaboration.py")
print("   ✓ forge_ai/core/standalone_tools.py")

print("\n🎯 Key Features Working:")
print("   ✓ Model registration and capability tracking")
print("   ✓ Capability queries and best model selection")
print("   ✓ Auto-detection of capabilities from model names")
print("   ✓ Performance ratings and rankings")
print("   ✓ Collaboration infrastructure")
print("   ✓ Standalone tool interface")

print("\n📚 Next Steps:")
print("   - See docs/ORCHESTRATION_GUIDE.md for full documentation")
print("   - Run examples/orchestration_demo.py (requires torch)")
print("   - Integrate with your ForgeAI models")

print()
