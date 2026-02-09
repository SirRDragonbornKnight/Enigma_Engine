#!/usr/bin/env python3
"""
Example: Using Specialized Model Routing in Enigma AI Engine

This example demonstrates:
1. Training specialized models
2. Using routing in inference
3. Accessing specialized model methods directly
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def example_1_basic_routing():
    """Example 1: Basic routing with EnigmaEngine"""
    print("=" * 70)
    print("Example 1: Basic Routing with EnigmaEngine")
    print("=" * 70)
    print()
    
    try:
        from enigma_engine.core.inference import EnigmaEngine
        
        # Create engine with routing enabled
        print("Creating EnigmaEngine with routing enabled...")
        engine = EnigmaEngine(use_routing=True)
        
        # The router will automatically classify intent and route to appropriate model
        print("\n1. Asking for code generation:")
        print('   Input: "write a Python function to sort a list"')
        print('   Expected: Router detects "code" intent')
        
        print("\n2. Asking for image generation:")
        print('   Input: "draw me a sunset over mountains"')
        print('   Expected: Router detects "image" intent')
        
        print("\n3. Regular chat:")
        print('   Input: "how are you today?"')
        print('   Expected: Router detects "chat" intent')
        
        print("\n✓ EnigmaEngine routing setup successful!")
        print("  Note: Specialized models need to be trained first")
        print("  Run: python scripts/train_specialized_model.py --type router --data data/specialized/router_training.txt --model-size nano")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Make sure PyTorch and dependencies are installed")


def example_2_direct_router_access():
    """Example 2: Direct router access"""
    print("\n")
    print("=" * 70)
    print("Example 2: Direct Router Access")
    print("=" * 70)
    print()
    
    try:
        from enigma_engine.core.tool_router import get_router
        
        # Get router with specialized models disabled (works without training)
        print("Getting router with keyword-based detection...")
        router = get_router(use_specialized=False)
        
        # Classify intent using keyword matching
        test_inputs = [
            "draw me a picture of a cat",
            "write a Python function",
            "what do you see in this image",
            "how does gravity work",
        ]
        
        print("\nClassifying intents (keyword-based):")
        for text in test_inputs:
            intent = router.detect_tool(text)
            print(f"  '{text}' -> {intent}")
        
        print("\n✓ Router keyword detection working!")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_3_training_workflow():
    """Example 3: Training workflow"""
    print("\n")
    print("=" * 70)
    print("Example 3: Training Specialized Models Workflow")
    print("=" * 70)
    print()
    
    print("Step 1: Prepare training data")
    print("  - Create or edit data/specialized/router_training.txt")
    print("  - Format: Q: <question>\\nA: [E:tool]<intent>")
    print("  - Example:")
    print("    Q: draw me a picture")
    print("    A: [E:tool]image")
    print()
    
    print("Step 2: Train the router model (nano size, fast)")
    print("  Command:")
    print("    python scripts/train_specialized_model.py \\")
    print("      --type router \\")
    print("      --data data/specialized/router_training.txt \\")
    print("      --model-size nano \\")
    print("      --epochs 50")
    print()
    
    print("Step 3: Train vision model (optional, for image captioning)")
    print("  Command:")
    print("    python scripts/train_specialized_model.py \\")
    print("      --type vision \\")
    print("      --data data/specialized/vision_training.txt \\")
    print("      --model-size tiny \\")
    print("      --epochs 40")
    print()
    
    print("Step 4: Train code model (optional, for code generation)")
    print("  Command:")
    print("    python scripts/train_specialized_model.py \\")
    print("      --type code \\")
    print("      --data data/specialized/code_training.txt \\")
    print("      --model-size small \\")
    print("      --epochs 40")
    print()
    
    print("Step 5: Use in your application")
    print("  Python code:")
    print("    from enigma_engine.core.inference import EnigmaEngine")
    print("    engine = EnigmaEngine(use_routing=True)")
    print("    response = engine.generate('write a sort function')")
    print()


def example_4_module_system():
    """Example 4: Using with module system"""
    print("\n")
    print("=" * 70)
    print("Example 4: Module System Integration")
    print("=" * 70)
    print()
    
    try:
        from enigma_engine.modules.manager import ModuleManager
        
        print("Using ModuleManager to load tool router...")
        manager = ModuleManager()
        
        # Check if tool_router module is available
        available = manager.list_available()
        tool_router_available = any(m.id == "tool_router" for m in available)
        
        if tool_router_available:
            print("✓ ToolRouterModule is available in registry")
            print("\nTo load it:")
            print("  manager.load('tool_router')")
            print("\nTo enable via GUI:")
            print("  1. Launch: python run.py --gui")
            print("  2. Go to Module Manager tab")
            print("  3. Find 'Tool Router' and click Load")
        else:
            print("✗ ToolRouterModule not found in registry")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "SPECIALIZED MODEL ROUTING EXAMPLES" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    example_1_basic_routing()
    example_2_direct_router_access()
    example_3_training_workflow()
    example_4_module_system()
    
    print("\n")
    print("=" * 70)
    print("For more information, see: docs/SPECIALIZED_MODELS.md")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
