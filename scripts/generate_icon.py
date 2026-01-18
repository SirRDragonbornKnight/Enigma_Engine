#!/usr/bin/env python3
"""
Generate ForgeAI Icon

Creates a custom icon for ForgeAI with:
- Modern gradient design
- "F" or forge/anvil symbol
- Multiple sizes for different uses

Requirements:
    pip install pillow

Usage:
    python scripts/generate_icon.py                    # Generate default icon
    python scripts/generate_icon.py --style anvil      # Anvil/forge style
    python scripts/generate_icon.py --style letter     # Letter "F" style
    python scripts/generate_icon.py --style neural     # Neural network style
    python scripts/generate_icon.py --color "#3498db"  # Custom primary color
"""

import argparse
import math
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    print("Installing Pillow...")
    import os
    os.system("pip install pillow")
    from PIL import Image, ImageDraw, ImageFont, ImageFilter


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_gradient(size, color1, color2, direction='diagonal'):
    """Create a gradient background."""
    img = Image.new('RGBA', (size, size))
    draw = ImageDraw.Draw(img)
    
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    
    for y in range(size):
        for x in range(size):
            if direction == 'diagonal':
                ratio = (x + y) / (2 * size)
            elif direction == 'vertical':
                ratio = y / size
            else:
                ratio = x / size
            
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            
            img.putpixel((x, y), (r, g, b, 255))
    
    return img


def create_letter_icon(size, primary_color, secondary_color):
    """Create icon with stylized 'F' letter."""
    # Create gradient background
    img = create_gradient(size, primary_color, secondary_color, 'diagonal')
    draw = ImageDraw.Draw(img)
    
    # Add rounded corners
    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    radius = size // 6
    mask_draw.rounded_rectangle([0, 0, size-1, size-1], radius=radius, fill=255)
    
    # Apply mask for rounded corners
    background = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    img = Image.composite(img, background, mask)
    draw = ImageDraw.Draw(img)
    
    # Draw "F" letter
    margin = size // 6
    stroke_width = size // 10
    
    # Vertical bar of F
    draw.rectangle([
        margin, margin,
        margin + stroke_width, size - margin
    ], fill=(255, 255, 255, 230))
    
    # Top horizontal bar
    draw.rectangle([
        margin, margin,
        size - margin, margin + stroke_width
    ], fill=(255, 255, 255, 230))
    
    # Middle horizontal bar
    mid_y = size // 2 - stroke_width // 2
    draw.rectangle([
        margin, mid_y,
        size - margin - margin // 2, mid_y + stroke_width
    ], fill=(255, 255, 255, 230))
    
    return img


def create_anvil_icon(size, primary_color, secondary_color):
    """Create icon with anvil/forge symbol."""
    img = create_gradient(size, primary_color, secondary_color, 'diagonal')
    draw = ImageDraw.Draw(img)
    
    # Rounded corners
    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    radius = size // 6
    mask_draw.rounded_rectangle([0, 0, size-1, size-1], radius=radius, fill=255)
    background = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    img = Image.composite(img, background, mask)
    draw = ImageDraw.Draw(img)
    
    # Draw simplified anvil shape
    margin = size // 5
    white = (255, 255, 255, 230)
    
    # Anvil top (wider rectangle)
    top_y = margin + size // 10
    draw.rounded_rectangle([
        margin, top_y,
        size - margin, top_y + size // 5
    ], radius=size//20, fill=white)
    
    # Anvil body (narrower)
    body_y = top_y + size // 5
    body_margin = margin + size // 8
    draw.rectangle([
        body_margin, body_y,
        size - body_margin, body_y + size // 4
    ], fill=white)
    
    # Anvil base (wider)
    base_y = body_y + size // 4
    draw.rounded_rectangle([
        margin + size // 16, base_y,
        size - margin - size // 16, size - margin
    ], radius=size//20, fill=white)
    
    # Hammer spark (small circle)
    spark_x = size - margin - size // 8
    spark_y = margin
    spark_r = size // 12
    draw.ellipse([
        spark_x - spark_r, spark_y,
        spark_x + spark_r, spark_y + spark_r * 2
    ], fill=(255, 200, 100, 200))
    
    return img


def create_neural_icon(size, primary_color, secondary_color):
    """Create icon with neural network nodes."""
    img = create_gradient(size, primary_color, secondary_color, 'diagonal')
    draw = ImageDraw.Draw(img)
    
    # Rounded corners
    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    radius = size // 6
    mask_draw.rounded_rectangle([0, 0, size-1, size-1], radius=radius, fill=255)
    background = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    img = Image.composite(img, background, mask)
    draw = ImageDraw.Draw(img)
    
    white = (255, 255, 255, 230)
    line_color = (255, 255, 255, 150)
    
    # Node positions (3 layers)
    margin = size // 5
    layer1_x = margin
    layer2_x = size // 2
    layer3_x = size - margin
    
    node_radius = size // 14
    
    # Layer 1 nodes (2 nodes)
    l1_nodes = [(layer1_x, size // 3), (layer1_x, 2 * size // 3)]
    
    # Layer 2 nodes (3 nodes)
    l2_nodes = [(layer2_x, size // 4), (layer2_x, size // 2), (layer2_x, 3 * size // 4)]
    
    # Layer 3 nodes (2 nodes)
    l3_nodes = [(layer3_x, size // 3), (layer3_x, 2 * size // 3)]
    
    # Draw connections
    line_width = max(1, size // 50)
    for n1 in l1_nodes:
        for n2 in l2_nodes:
            draw.line([n1, n2], fill=line_color, width=line_width)
    
    for n2 in l2_nodes:
        for n3 in l3_nodes:
            draw.line([n2, n3], fill=line_color, width=line_width)
    
    # Draw nodes
    all_nodes = l1_nodes + l2_nodes + l3_nodes
    for (x, y) in all_nodes:
        draw.ellipse([
            x - node_radius, y - node_radius,
            x + node_radius, y + node_radius
        ], fill=white)
    
    return img


def save_icon_sizes(base_img, output_dir, name="forge"):
    """Save icon in multiple sizes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sizes = [16, 32, 48, 64, 128, 256]
    
    # Save PNG versions
    for size in sizes:
        resized = base_img.resize((size, size), Image.Resampling.LANCZOS)
        resized.save(output_dir / f"{name}_{size}.png", "PNG")
        print(f"  Saved {name}_{size}.png")
    
    # Save ICO (Windows icon with multiple sizes)
    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    ico_images = [base_img.resize(s, Image.Resampling.LANCZOS) for s in ico_sizes]
    
    ico_path = output_dir / f"{name}.ico"
    ico_images[0].save(
        ico_path,
        format='ICO',
        sizes=ico_sizes,
        append_images=ico_images[1:]
    )
    print(f"  Saved {name}.ico")
    
    return ico_path


def main():
    parser = argparse.ArgumentParser(description="Generate ForgeAI icon")
    parser.add_argument("--style", choices=["letter", "anvil", "neural"], 
                        default="neural", help="Icon style")
    parser.add_argument("--color", default="#3498db", help="Primary color (hex)")
    parser.add_argument("--color2", default="#9b59b6", help="Secondary color (hex)")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "data" / "icons"
    
    primary = hex_to_rgb(args.color)
    secondary = hex_to_rgb(args.color2)
    
    print(f"Generating {args.style} icon...")
    print(f"  Primary: {args.color}")
    print(f"  Secondary: {args.color2}")
    
    # Generate at high resolution
    size = 512
    
    if args.style == "letter":
        icon = create_letter_icon(size, primary, secondary)
    elif args.style == "anvil":
        icon = create_anvil_icon(size, primary, secondary)
    else:
        icon = create_neural_icon(size, primary, secondary)
    
    # Save all sizes
    print(f"\nSaving to {output_dir}...")
    ico_path = save_icon_sizes(icon, output_dir)
    
    # Also save to gui/icons for PyQt
    gui_icons = project_root / "forge_ai" / "gui" / "icons"
    if gui_icons.exists():
        save_icon_sizes(icon, gui_icons)
    
    print(f"\nDone! Icon saved to {ico_path}")


if __name__ == "__main__":
    main()
