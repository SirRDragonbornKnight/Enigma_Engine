#!/usr/bin/env python3
"""
Example: Using Enigma Engine's Local vs Cloud Module System

This example demonstrates how to:
1. Check which modules are local vs cloud
2. Use local-only mode for privacy
3. Enable cloud modules when needed
"""

import sys
from pathlib import Path

# Add parent to path for running from examples dir
sys.path.insert(0, str(Path(__file__).parent.parent))

from enigma.modules import registry
from enigma.modules.manager import ModuleManager

print("="*70)
print("EXAMPLE: Local vs Cloud Module Usage")
print("="*70)
print()

# ============================================================================
# Example 1: List available modules
# ============================================================================
print("Example 1: Discovering what's available")
print("-" * 70)

# Get all local modules (no internet required)
local_modules = registry.list_local_modules()
print(f"Found {len(local_modules)} modules that work 100% offline:")
for module in local_modules[:5]:
    print(f"  - {module.id} - {module.name}")
print()

# Get all cloud modules (require API keys)
cloud_modules = registry.list_cloud_modules()
print(f"Found {len(cloud_modules)} cloud API modules:")
for module in cloud_modules:
    print(f"  - {module.id} - {module.name}")
print()

# ============================================================================
# Example 2: Privacy-first setup (default)
# ============================================================================
print("Example 2: Privacy-first setup (100% local)")
print("-" * 70)
print("""
For maximum privacy, use local-only mode (this is the DEFAULT):

    manager = ModuleManager()  # local_only=True by default
    
This ensures:
  - No API keys needed
  - No data leaves your machine
  - Works completely offline
  - Cloud modules cannot be loaded
  
Perfect for: Privacy-conscious users, offline environments, learning AI
""")

# ============================================================================
# Example 3: Hybrid setup (local + selective cloud)
# ============================================================================
print("Example 3: Hybrid setup (mostly local + selective cloud)")
print("-" * 70)
print("""
When you need specific cloud services:

    manager = ModuleManager(local_only=False)
    
    # Load local modules for core functionality
    manager.load('model')
    manager.load('tokenizer')
    manager.load('inference')
    manager.load('image_gen_local')  # Local Stable Diffusion
    
    # Use cloud API only for specific high-quality tasks
    manager.load('code_gen_api', config={
        'api_key': 'sk-...',
        'model': 'gpt-4'
    })
    
    # WARNING: Module 'code_gen_api' connects to external cloud services
    
Best for: Production apps needing both privacy and cloud capabilities
""")

# ============================================================================
# Example 4: Checking module privacy before loading
# ============================================================================
print("Example 4: Checking if a module is local or cloud")
print("-" * 70)
print("""
Before loading a module, check if it's cloud-based:

    module_class = registry.get_module('image_gen_api')
    info = module_class.get_info()
    
    if info.is_cloud_service:
        print("[!] This module requires internet and API keys")
        print(f"   Description: {info.description}")
        # Ask user for confirmation before loading
    else:
        print("[OK] This module runs 100% locally")
        # Safe to load without concerns
""")

module_class = registry.get_module('image_gen_local')
info = module_class.get_info()
print(f"\nExample: {info.id}")
print(f"  Is cloud service: {info.is_cloud_service}")
print(f"  Description: {info.description}")

module_class = registry.get_module('image_gen_api')
info = module_class.get_info()
print(f"\nExample: {info.id}")
print(f"  Is cloud service: {info.is_cloud_service}")
print(f"  Description: {info.description}")

print()

# ============================================================================
# Example 5: Recommended configurations
# ============================================================================
print("Example 5: Recommended configurations")
print("-" * 70)
print("""
Configuration 1: Raspberry Pi / Low-end PC (100% Local)
-------------------------------------------------------
manager = ModuleManager(local_only=True)  # Default!
manager.load('model')
manager.load('tokenizer')
manager.load('inference')
manager.load('memory')
manager.load('audio_gen_local')  # pyttsx3 TTS

Configuration 2: Desktop with GPU (Local + Generation)
-----------------------------------------------------
manager = ModuleManager(local_only=True)  # Still local!
manager.load('model')
manager.load('tokenizer')
manager.load('inference')
manager.load('image_gen_local')  # Stable Diffusion
manager.load('audio_gen_local')
manager.load('embedding_local')  # sentence-transformers

Configuration 3: Production (Hybrid)
-----------------------------------
manager = ModuleManager(local_only=False)
manager.load('model')
manager.load('inference')
manager.load('image_gen_local')  # Primary: local SD
# Only for complex code generation:
manager.load('code_gen_api', config={'api_key': '...'})
""")

print("="*70)
print("For complete documentation, see: docs/LOCAL_VS_CLOUD.md")
print("="*70)
print()
