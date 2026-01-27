#!/usr/bin/env python3
"""
Example: Deep Multi-Model Integration - Orchestration System

This example demonstrates the unified orchestration system that enables
multiple AI models to work together seamlessly.

Run this with:
    python examples/orchestration_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_capability_registry():
    """Demonstrate the capability registry."""
    print("\n" + "="*70)
    print("1. CAPABILITY REGISTRY - Track What Each Model Can Do")
    print("="*70)
    
    from forge_ai.core import get_capability_registry
    
    registry = get_capability_registry()
    
    # Register some models
    print("\n📝 Registering models...")
    
    registry.register_model(
        model_id="forge:small",
        capabilities=["text_generation", "reasoning"],
        metadata={"size": "27M", "device": "cpu"},
        performance_ratings={"text_generation": 0.7, "reasoning": 0.6}
    )
    print("   ✓ Registered forge:small (text, reasoning)")
    
    registry.register_model(
        model_id="forge:code",
        capabilities=["code_generation", "text_generation"],
        metadata={"size": "85M", "device": "cpu"},
        performance_ratings={"code_generation": 0.9}
    )
    print("   ✓ Registered forge:code (code generation)")
    
    registry.register_model(
        model_id="huggingface:llava",
        capabilities=["vision", "text_generation"],
        metadata={"size": "7B", "device": "cuda"},
        performance_ratings={"vision": 0.85}
    )
    print("   ✓ Registered llava (vision)")
    
    # Query capabilities
    print("\n🔍 Querying capabilities...")
    
    models = registry.list_models()
    print(f"   Total models: {len(models)}")
    
    text_models = registry.find_models_with_capability("text_generation")
    print(f"   Text generation models: {text_models}")
    
    best_code = registry.find_best_model("code_generation")
    print(f"   Best for code: {best_code}")
    
    best_vision = registry.find_best_model("vision")
    print(f"   Best for vision: {best_vision}")


def demo_model_pool():
    """Demonstrate the model pool."""
    print("\n" + "="*70)
    print("2. MODEL POOL - Efficient Model Lifecycle Management")
    print("="*70)
    
    from forge_ai.core import get_model_pool, ModelPoolConfig
    
    # Configure the pool
    config = ModelPoolConfig(
        max_loaded_models=3,
        gpu_memory_limit_mb=8000,
        enable_auto_eviction=True,
        fallback_to_cpu=True,
    )
    
    pool = get_model_pool(config)
    
    print("\n📊 Pool configuration:")
    print(f"   Max models: {config.max_loaded_models}")
    print(f"   GPU limit: {config.gpu_memory_limit_mb}MB")
    print(f"   Auto-eviction: {config.enable_auto_eviction}")
    
    print("\n💾 Memory usage:")
    usage = pool.get_memory_usage()
    print(f"   Total: {usage['total_mb']:.1f}MB")
    print(f"   GPU: {usage['gpu_mb']:.1f}MB")
    print(f"   CPU: {usage['cpu_mb']:.1f}MB")
    print(f"   Models loaded: {usage['num_models']}")
    
    print("\n🖥️  System memory:")
    sys_info = pool.get_system_memory_info()
    print(f"   System RAM: {sys_info['system_total_mb']:.0f}MB")
    print(f"   Available: {sys_info['system_available_mb']:.0f}MB")
    print(f"   Used: {sys_info['system_percent']:.1f}%")


def demo_collaboration():
    """Demonstrate model collaboration."""
    print("\n" + "="*70)
    print("3. MODEL COLLABORATION - Models Working Together")
    print("="*70)
    
    from forge_ai.core import get_collaboration
    
    collaboration = get_collaboration()
    
    print("\n🤝 Collaboration features:")
    print("   • Request/Response - Simple ask-and-answer")
    print("   • Confidence Handoff - Hand off if confidence too low")
    print("   • Pipeline - Multi-stage processing")
    print("   • Consensus - Multiple models vote")
    
    print("\n📊 Collaboration statistics:")
    stats = collaboration.get_collaboration_stats()
    print(f"   Total collaborations: {stats['total_collaborations']}")
    if stats['total_collaborations'] > 0:
        print(f"   Success rate: {stats['success_rate']*100:.1f}%")


def demo_orchestrator():
    """Demonstrate the orchestrator."""
    print("\n" + "="*70)
    print("4. ORCHESTRATOR - Central Intelligence Coordinator")
    print("="*70)
    
    from forge_ai.core import get_orchestrator, OrchestratorConfig
    
    # Configure orchestrator
    config = OrchestratorConfig(
        default_chat_model="auto",
        max_loaded_models=3,
        enable_collaboration=True,
        enable_auto_fallback=True,
    )
    
    orchestrator = get_orchestrator(config)
    
    print("\n⚙️  Orchestrator configuration:")
    print(f"   Max models: {config.max_loaded_models}")
    print(f"   Collaboration: {config.enable_collaboration}")
    print(f"   Auto-fallback: {config.enable_auto_fallback}")
    
    # Register models
    print("\n📝 Registering models with orchestrator...")
    
    orchestrator.register_model(
        model_id="demo:chat",
        capabilities=["text_generation", "reasoning"],
        metadata={"size": "small"},
    )
    print("   ✓ Registered demo:chat")
    
    orchestrator.register_model(
        model_id="demo:code",
        capabilities=["code_generation"],
        metadata={"size": "medium"},
    )
    print("   ✓ Registered demo:code")
    
    orchestrator.register_model(
        model_id="demo:vision",
        capabilities=["vision"],
        metadata={"size": "large"},
    )
    print("   ✓ Registered demo:vision")
    
    # Assign models to capabilities
    print("\n🎯 Assigning models to capabilities...")
    orchestrator.assign_model_to_capability("text_generation", "demo:chat")
    orchestrator.assign_model_to_capability("code_generation", "demo:code")
    orchestrator.assign_model_to_capability("vision", "demo:vision")
    print("   ✓ Assignments complete")
    
    # Set fallback chain
    print("\n🔄 Setting fallback chains...")
    orchestrator.set_fallback_chain(
        "demo:chat",
        ["demo:code"]  # If chat fails, try code model
    )
    print("   ✓ Fallback chain: demo:chat → demo:code")
    
    # Get status
    print("\n📊 Orchestrator status:")
    status = orchestrator.get_status()
    print(f"   Registered models: {len(status['registered_models'])}")
    print(f"   Loaded models: {len(status['loaded_models'])}")
    print(f"   Collaboration enabled: {status['collaboration_enabled']}")


def demo_standalone_tools():
    """Demonstrate standalone tools."""
    print("\n" + "="*70)
    print("5. STANDALONE TOOLS - Use AI Without LLM")
    print("="*70)
    
    from forge_ai.core.standalone_tools import list_available_tools, get_tool_info
    
    print("\n🛠️  Available standalone tools:")
    tools = list_available_tools()
    for tool in tools:
        info = get_tool_info(tool)
        desc = info.get('description', 'N/A')
        print(f"   • {tool:15s} - {desc}")
    
    print("\n💡 Example usage:")
    print("   from forge_ai import use_tool")
    print()
    print("   # Generate image without chat")
    print('   use_tool("image", prompt="sunset", width=512, height=512)')
    print()
    print("   # Analyze image without chat")
    print('   use_tool("vision", image_path="photo.jpg")')
    print()
    print("   # Generate code without chat")
    print('   use_tool("code", prompt="sort function")')


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("🚀 FORGE AI - UNIFIED ORCHESTRATION SYSTEM DEMO")
    print("="*70)
    print("\nThis demo shows how multiple AI models can work together seamlessly.")
    print("Features: Capability tracking, efficient loading, collaboration, and more!")
    
    try:
        demo_capability_registry()
        demo_model_pool()
        demo_collaboration()
        demo_orchestrator()
        demo_standalone_tools()
        
        print("\n" + "="*70)
        print("✅ DEMO COMPLETE!")
        print("="*70)
        print("\n📚 For more information, see:")
        print("   docs/ORCHESTRATION_GUIDE.md")
        print("   forge_ai/core/orchestrator.py")
        print("   forge_ai/core/standalone_tools.py")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
