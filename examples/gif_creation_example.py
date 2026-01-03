#!/usr/bin/env python3
"""
GIF Creation Example
====================

Demonstrates how to create animated GIFs using the generate_gif tool.

This example shows:
1. Basic GIF generation from text prompts
2. Customizing FPS and loop settings
3. Creating GIFs with different frame counts
4. Error handling for GIF generation

Requirements:
- Image generation module loaded (image_gen_local or image_gen_api)
- Pillow installed (pip install Pillow)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from enigma.modules import ModuleManager
from enigma.tools.tool_executor import ToolExecutor


def example_basic_gif():
    """Create a basic 3-frame GIF."""
    print("\n=== Example 1: Basic GIF Generation ===\n")
    
    # Initialize module manager
    manager = ModuleManager()
    
    # Note: You need to have image_gen_local loaded for this to work
    # This is just a demonstration of the tool interface
    print("To use this tool, first load an image generation module:")
    print("  manager.load('image_gen_local')  # or 'image_gen_api'")
    print()
    
    # Create tool executor
    executor = ToolExecutor(module_manager=manager)
    
    # Define frame prompts
    frames = [
        "A sunrise over mountains with orange sky",
        "A bright sunny day over mountains",
        "A sunset over mountains with purple sky"
    ]
    
    # Generate GIF
    print(f"Generating GIF with {len(frames)} frames...")
    result = executor.execute_tool(
        "generate_gif",
        {
            "frames": frames,
            "fps": 2,  # 2 frames per second
            "loop": 0,  # Loop forever
            "width": 512,
            "height": 512
        }
    )
    
    if result["success"]:
        print(f"[OK] Success! GIF saved to: {result['output_path']}")
        print(f"  Frames: {result['frames']}")
        print(f"  FPS: {result['fps']}")
    else:
        print(f"[FAIL] Error: {result['error']}")


def example_fast_gif():
    """Create a fast-paced GIF with more frames."""
    print("\n=== Example 2: Fast Animation GIF ===\n")
    
    manager = ModuleManager()
    executor = ToolExecutor(module_manager=manager)
    
    # Define more frames for smoother animation
    frames = [
        "A bouncing ball at the top",
        "A bouncing ball falling down",
        "A bouncing ball at the bottom",
        "A bouncing ball bouncing up",
        "A bouncing ball in mid-air"
    ]
    
    print(f"Generating fast-paced GIF with {len(frames)} frames at 10 FPS...")
    result = executor.execute_tool(
        "generate_gif",
        {
            "frames": frames,
            "fps": 10,  # Faster animation
            "loop": 0,
            "width": 256,  # Smaller for faster generation
            "height": 256
        }
    )
    
    if result["success"]:
        print(f"[OK] Success! GIF saved to: {result['output_path']}")
    else:
        print(f"[FAIL] Error: {result['error']}")


def example_loop_limited_gif():
    """Create a GIF that loops a specific number of times."""
    print("\n=== Example 3: Limited Loop GIF ===\n")
    
    manager = ModuleManager()
    executor = ToolExecutor(module_manager=manager)
    
    frames = [
        "A door opening",
        "A door fully open",
        "A door closing"
    ]
    
    print(f"Generating GIF that loops 3 times...")
    result = executor.execute_tool(
        "generate_gif",
        {
            "frames": frames,
            "fps": 3,
            "loop": 3,  # Loop 3 times then stop
            "width": 512,
            "height": 512
        }
    )
    
    if result["success"]:
        print(f"[OK] Success! GIF saved to: {result['output_path']}")
        print(f"  This GIF will play 3 times then stop")
    else:
        print(f"[FAIL] Error: {result['error']}")


def example_validation():
    """Demonstrate parameter validation."""
    print("\n=== Example 4: Parameter Validation ===\n")
    
    executor = ToolExecutor()
    
    # Test with missing required parameter
    print("Testing with missing 'frames' parameter...")
    result = executor.execute_tool("generate_gif", {"fps": 5})
    
    if not result["success"]:
        print(f"[OK] Correctly caught error: {result['error']}")
    
    # Test with empty frames list
    print("\nTesting with empty frames list...")
    result = executor.execute_tool("generate_gif", {"frames": []})
    
    if not result["success"]:
        print(f"[OK] Correctly caught error: {result['error']}")
    
    # Test with valid parameters
    print("\nTesting parameter validation with valid params...")
    is_valid, error, validated = executor.validate_params(
        "generate_gif",
        {"frames": ["test1", "test2"], "fps": 5, "loop": 0}
    )
    
    if is_valid:
        print(f"[OK] Parameters validated successfully!")
        print(f"  Validated params: {validated}")
    else:
        print(f"[FAIL] Validation failed: {error}")


def main():
    """Run all examples."""
    print("="*70)
    print("GIF Creation Examples")
    print("="*70)
    print()
    print("NOTE: These examples require an image generation module to be loaded.")
    print("Load image_gen_local or image_gen_api before running the actual generation.")
    print()
    
    # Run examples
    example_basic_gif()
    example_fast_gif()
    example_loop_limited_gif()
    example_validation()
    
    print("\n" + "="*70)
    print("Examples Complete!")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Load an image generation module (image_gen_local or image_gen_api)")
    print("2. Run these examples with actual image generation")
    print("3. Check the 'outputs/' directory for generated GIFs")
    print()


if __name__ == "__main__":
    main()
