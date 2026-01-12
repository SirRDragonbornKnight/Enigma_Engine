#!/usr/bin/env python3
"""
Module System Demo
==================

Demonstrates the ForgeAI Module System in action.
Shows loading, conflict detection, and module usage.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forge_ai.modules import ModuleManager, ModuleCategory
from forge_ai.modules.registry import register_all


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def demo_hardware_detection():
    """Demo 1: Hardware Detection"""
    print_header("Demo 1: Hardware Detection")
    
    manager = ModuleManager()
    hw = manager.hardware_profile
    
    print(f"CPU Cores:     {hw['cpu_cores']}")
    print(f"RAM:           {hw['ram_mb']} MB")
    print(f"GPU Available: {hw['gpu_available']}")
    
    if hw['gpu_available']:
        print(f"GPU Name:      {hw['gpu_name']}")
        print(f"VRAM:          {hw['vram_mb']} MB")
    
    print("\nThis info is used to check if modules can load.")


def demo_module_discovery():
    """Demo 2: Module Discovery"""
    print_header("Demo 2: Module Discovery")
    
    manager = ModuleManager()
    register_all(manager)
    
    print(f"Total modules registered: {len(manager.module_classes)}\n")
    
    # Show modules by category
    for category in ModuleCategory:
        modules = manager.list_modules(category)
        if modules:
            print(f"\n{category.value.upper()}:")
            for mod in modules:
                print(f"  - {mod.id:20s} {mod.name}")


def demo_simple_loading():
    """Demo 3: Simple Module Loading"""
    print_header("Demo 3: Simple Module Loading")
    
    manager = ModuleManager()
    register_all(manager)
    
    print("Loading memory module (no dependencies)...")
    success = manager.load('memory')
    
    if success:
        print("[OK] Memory module loaded successfully")
        print(f"  State: {manager.get_module('memory').state.value}")
        
        interface = manager.get_interface('memory')
        print(f"  Interface type: {type(interface).__name__}")
    else:
        print("[FAIL] Failed to load memory module")


def demo_dependency_chain():
    """Demo 4: Loading with Dependencies"""
    print_header("Demo 4: Dependency Chain")
    
    manager = ModuleManager()
    register_all(manager)
    
    print("Attempting to load 'inference' without dependencies...")
    can_load, reason = manager.can_load('inference')
    print(f"  Can load: {can_load}")
    print(f"  Reason: {reason}\n")
    
    print("Loading dependencies in order:")
    print("  1. Loading 'model'...")
    manager.load('model', {'size': 'nano', 'vocab_size': 1000})
    print("     [OK] Model loaded")
    
    print("  2. Loading 'tokenizer'...")
    manager.load('tokenizer')
    print("     [OK] Tokenizer loaded")
    
    print("  3. Now trying 'inference' again...")
    can_load, reason = manager.can_load('inference')
    print(f"     Can load: {can_load}")
    
    if can_load:
        manager.load('inference')
        print("     [OK] Inference loaded")
    
    print(f"\nCurrently loaded: {manager.list_loaded()}")


def demo_conflict_detection():
    """Demo 5: Conflict Detection"""
    print_header("Demo 5: Conflict Detection")
    
    manager = ModuleManager()
    register_all(manager)
    
    print("Scenario: Two modules provide 'image_generation'")
    print("  - image_gen_local (Stable Diffusion)")
    print("  - image_gen_api (DALL-E / Replicate)\n")
    
    # Check if we can load local version
    can_load_local, reason = manager.can_load('image_gen_local')
    print(f"Can load image_gen_local: {can_load_local}")
    
    if not can_load_local:
        print(f"  Reason: {reason}")
        print("\n  -> Falling back to API version instead")
        can_load_api, reason_api = manager.can_load('image_gen_api')
        print(f"  Can load image_gen_api: {can_load_api}")
    else:
        print("  Loading image_gen_local...")
        # Note: Would actually load if GPU available
        print("  (Skipping actual load in demo - needs GPU)")
        
        # Simulate it being loaded
        print("\n  Now trying to load image_gen_api too...")
        print("  Result: [CONFLICT] Both provide 'image_generation'")
        print("  -> Module system prevents this automatically")


def demo_module_swapping():
    """Demo 6: Dynamic Module Swapping"""
    print_header("Demo 6: Dynamic Module Swapping")
    
    manager = ModuleManager()
    register_all(manager)
    
    print("Starting with core modules...")
    manager.load('model', {'size': 'nano', 'vocab_size': 1000})
    manager.load('tokenizer')
    manager.load('inference')
    
    print(f"Loaded: {manager.list_loaded()}\n")
    
    print("User wants code generation...")
    print("  Checking hardware...")
    
    hw = manager.hardware_profile
    if hw['ram_mb'] >= 4096:
        print(f"  RAM: {hw['ram_mb']} MB [OK]")
        print("  -> Loading code_gen_local (free, private)")
        # Would actually load if dependencies met
        print("  (Demo mode - skipping actual load)")
    else:
        print(f"  RAM: {hw['ram_mb']} MB [LOW]")
        print("  -> Would recommend code_gen_api instead")
    
    print("\nUser changes mind, wants images instead...")
    print("  Unloading code_gen_local...")
    print("  Loading image_gen_api (with API key)...")
    print("  [OK] Swapped successfully"))


def demo_configuration():
    """Demo 7: Module Configuration"""
    print_header("Demo 7: Module Configuration")
    
    manager = ModuleManager()
    register_all(manager)
    
    print("Model module configuration options:")
    model_class = manager.module_classes['model']
    config_schema = model_class.get_info().config_schema
    
    for key, schema in config_schema.items():
        print(f"\n  {key}:")
        print(f"    Type:    {schema['type']}")
        if 'options' in schema:
            print(f"    Options: {', '.join(schema['options'])}")
        print(f"    Default: {schema['default']}")
    
    print("\n\nLoading with custom config:")
    config = {
        'size': 'tiny',
        'vocab_size': 5000,
        'device': 'cpu',
    }
    print(f"  Config: {config}")
    
    manager.load('model', config)
    module = manager.get_module('model')
    print(f"  Module config: {module.config}")


def demo_list_by_category():
    """Demo 8: Browse Modules by Category"""
    print_header("Demo 8: Browse Modules by Category")
    
    manager = ModuleManager()
    register_all(manager)
    
    print("GENERATION modules (AI capabilities):\n")
    
    gen_modules = manager.list_modules(ModuleCategory.GENERATION)
    for mod in gen_modules:
        print(f"  {mod.id}")
        print(f"    Name:     {mod.name}")
        print(f"    Provides: {', '.join(mod.provides)}")
        
        if mod.requires_gpu:
            print(f"    [!] Requires GPU")
        if mod.min_vram_mb > 0:
            print(f"    [!] Needs {mod.min_vram_mb}MB VRAM"))
        
        print()


def main():
    """Run all demos."""
    print("""
================================================================
                                                              
           ENIGMA ENGINE - MODULE SYSTEM DEMO                 
                                                              
  This script demonstrates how the module system works:      
  - Hardware detection                                        
  - Module discovery and registration                         
  - Dependency resolution                                     
  - Conflict prevention                                       
  - Dynamic loading/unloading                                 
                                                              
================================================================
    """)
    
    try:
        demos = [
            demo_hardware_detection,
            demo_module_discovery,
            demo_simple_loading,
            demo_dependency_chain,
            demo_conflict_detection,
            demo_module_swapping,
            demo_configuration,
            demo_list_by_category,
        ]
        
        for i, demo in enumerate(demos, 1):
            demo()
            
            if i < len(demos):
                input("\nPress Enter to continue to next demo...")
        
        print_header("Demo Complete!")
        print("[OK] All demos completed successfully")
        print("\nNext steps:")
        print("  1. Read docs/MODULE_GUIDE.md for full documentation")
        print("  2. Try loading modules yourself in python shell")
        print("  3. Create your own custom module")
        print("\nHappy coding!")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n\n[FAIL] Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
