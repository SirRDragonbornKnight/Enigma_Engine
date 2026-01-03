#!/usr/bin/env python3
"""
Media Editing Example
=====================

Demonstrates how to edit images, GIFs, and videos using the editing tools.

This example shows:
1. Image editing (resize, rotate, filters, etc.)
2. GIF editing (speed, reverse, extract frames, etc.)
3. Video editing (trim, convert to GIF, etc.)
4. Error handling and validation

Requirements:
- Pillow installed (pip install Pillow)
- MoviePy installed for video editing (pip install moviepy) - optional
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from enigma.tools.tool_executor import ToolExecutor


def create_test_image():
    """Create a test image for editing examples."""
    from PIL import Image, ImageDraw
    import os
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Create a simple test image
    img = Image.new('RGB', (400, 300), color='skyblue')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.rectangle([50, 50, 350, 250], fill='green', outline='darkgreen', width=3)
    draw.ellipse([150, 100, 250, 200], fill='yellow', outline='orange', width=3)
    
    # Save test image
    test_path = "outputs/test_image.png"
    img.save(test_path)
    print(f"Created test image: {test_path}")
    return test_path


def create_test_gif():
    """Create a test GIF for editing examples."""
    from PIL import Image
    import os
    
    os.makedirs("outputs", exist_ok=True)
    
    # Create frames with different colors
    frames = []
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    
    for color in colors:
        img = Image.new('RGB', (200, 200), color=color)
        frames.append(img)
    
    # Save as GIF
    test_path = "outputs/test_animation.gif"
    frames[0].save(
        test_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )
    print(f"Created test GIF: {test_path}")
    return test_path


def example_image_resize():
    """Example: Resize an image."""
    print("\n=== Example 1: Resize Image ===\n")
    
    executor = ToolExecutor()
    test_image = create_test_image()
    
    print("Resizing image to 800x600...")
    result = executor.execute_tool(
        "edit_image",
        {
            "image_path": test_image,
            "edit_type": "resize",
            "width": 800,
            "height": 600
        }
    )
    
    if result["success"]:
        print(f"[OK] Success! Edited image saved to: {result['output_path']}")
    else:
        print(f"[FAIL] Error: {result['error']}")


def example_image_rotate():
    """Example: Rotate an image."""
    print("\n=== Example 2: Rotate Image ===\n")
    
    executor = ToolExecutor()
    test_image = "outputs/test_image.png"
    
    print("Rotating image 45 degrees...")
    result = executor.execute_tool(
        "edit_image",
        {
            "image_path": test_image,
            "edit_type": "rotate",
            "angle": 45
        }
    )
    
    if result["success"]:
        print(f"[OK] Success! Rotated image saved to: {result['output_path']}")
    else:
        print(f"[FAIL] Error: {result['error']}")


def example_image_brightness():
    """Example: Adjust image brightness."""
    print("\n=== Example 3: Adjust Brightness ===\n")
    
    executor = ToolExecutor()
    test_image = "outputs/test_image.png"
    
    print("Increasing brightness by 1.5x...")
    result = executor.execute_tool(
        "edit_image",
        {
            "image_path": test_image,
            "edit_type": "brightness",
            "factor": 1.5
        }
    )
    
    if result["success"]:
        print(f"[OK] Success! Brightened image saved to: {result['output_path']}")
    else:
        print(f"[FAIL] Error: {result['error']}")


def example_image_filters():
    """Example: Apply image filters."""
    print("\n=== Example 4: Apply Filters ===\n")
    
    executor = ToolExecutor()
    test_image = "outputs/test_image.png"
    
    filters = ["blur", "sharpen", "grayscale"]
    
    for filter_type in filters:
        print(f"Applying {filter_type} filter...")
        result = executor.execute_tool(
            "edit_image",
            {
                "image_path": test_image,
                "edit_type": filter_type
            }
        )
        
        if result["success"]:
            print(f"  [OK] {filter_type.capitalize()} applied: {result['output_path']}")
        else:
            print(f"  [FAIL] Error: {result['error']}")


def example_gif_reverse():
    """Example: Reverse a GIF animation."""
    print("\n=== Example 5: Reverse GIF ===\n")
    
    executor = ToolExecutor()
    test_gif = create_test_gif()
    
    print("Reversing GIF animation...")
    result = executor.execute_tool(
        "edit_gif",
        {
            "gif_path": test_gif,
            "edit_type": "reverse"
        }
    )
    
    if result["success"]:
        print(f"[OK] Success! Reversed GIF saved to: {result['output_path']}")
        print(f"  Frame count: {result['frame_count']}")
    else:
        print(f"[FAIL] Error: {result['error']}")


def example_gif_speed():
    """Example: Change GIF speed."""
    print("\n=== Example 6: Speed Up GIF ===\n")
    
    executor = ToolExecutor()
    test_gif = "outputs/test_animation.gif"
    
    print("Making GIF 2x faster...")
    result = executor.execute_tool(
        "edit_gif",
        {
            "gif_path": test_gif,
            "edit_type": "speed",
            "speed_factor": 2.0
        }
    )
    
    if result["success"]:
        print(f"[OK] Success! Faster GIF saved to: {result['output_path']}")
    else:
        print(f"[FAIL] Error: {result['error']}")


def example_gif_extract_frames():
    """Example: Extract frames from GIF."""
    print("\n=== Example 7: Extract GIF Frames ===\n")
    
    executor = ToolExecutor()
    test_gif = "outputs/test_animation.gif"
    
    print("Extracting individual frames from GIF...")
    result = executor.execute_tool(
        "edit_gif",
        {
            "gif_path": test_gif,
            "edit_type": "extract_frames"
        }
    )
    
    if result["success"]:
        print(f"[OK] Success! Extracted {result['frame_count']} frames")
        print(f"  First frame: {result['frame_paths'][0]}")
    else:
        print(f"[FAIL] Error: {result['error']}")


def example_video_editing():
    """Example: Video editing (requires moviepy)."""
    print("\n=== Example 8: Video Editing ===\n")
    
    executor = ToolExecutor()
    
    # Note: This requires a video file and moviepy installed
    print("Video editing examples require:")
    print("  1. A video file (e.g., test_video.mp4)")
    print("  2. MoviePy installed: pip install moviepy")
    print()
    print("Example operations:")
    print("  - Trim: Cut video from start to end time")
    print("  - Speed: Change playback speed")
    print("  - Resize: Change video dimensions")
    print("  - To GIF: Convert video to animated GIF")
    print("  - Extract frames: Save individual video frames")
    print()
    print("Sample usage:")
    print("""
    result = executor.execute_tool(
        "edit_video",
        {
            "video_path": "outputs/my_video.mp4",
            "edit_type": "to_gif",
            "fps": 10
        }
    )
    """)


def example_error_handling():
    """Example: Error handling for media editing."""
    print("\n=== Example 9: Error Handling ===\n")
    
    executor = ToolExecutor()
    
    # Test with non-existent file
    print("Testing with non-existent file...")
    result = executor.execute_tool(
        "edit_image",
        {
            "image_path": "nonexistent.png",
            "edit_type": "resize",
            "width": 100,
            "height": 100
        }
    )
    
    if not result["success"]:
        print(f"[OK] Correctly caught error: {result['error']}")
    
    # Test with invalid edit type
    print("\nTesting with invalid edit type...")
    result = executor.execute_tool(
        "edit_image",
        {
            "image_path": "outputs/test_image.png",
            "edit_type": "invalid_operation"
        }
    )
    
    if not result["success"]:
        print(f"[OK] Correctly caught error: {result['error']}")


def main():
    """Run all examples."""
    print("="*70)
    print("Media Editing Examples")
    print("="*70)
    print()
    print("This script demonstrates various image and GIF editing operations.")
    print("Test files will be created in the 'outputs/' directory.")
    print()
    
    try:
        # Image editing examples
        example_image_resize()
        example_image_rotate()
        example_image_brightness()
        example_image_filters()
        
        # GIF editing examples
        example_gif_reverse()
        example_gif_speed()
        example_gif_extract_frames()
        
        # Video editing info
        example_video_editing()
        
        # Error handling
        example_error_handling()
        
        print("\n" + "="*70)
        print("Examples Complete!")
        print("="*70)
        print()
        print("Check the 'outputs/' directory for edited images and GIFs.")
        print()
        
    except Exception as e:
        print(f"\n[FAIL] Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
