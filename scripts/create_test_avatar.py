#!/usr/bin/env python3
"""
Create Test Avatar Images

Generates avatar images from the built-in SVG sprites for testing.
Run this script to create test avatar files in data/avatar/
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forge_ai.avatar.renderers.default_sprites import SPRITE_TEMPLATES, generate_sprite
from forge_ai.config import CONFIG
import json


def create_test_avatars():
    """Create test avatar images and config."""
    
    avatar_dir = Path(CONFIG["data_dir"]) / "avatar"
    avatar_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating test avatars...")
    print(f"Output directory: {avatar_dir}")
    
    # Color schemes to generate
    color_schemes = {
        "default": {
            "primary_color": "#6366f1",
            "secondary_color": "#8b5cf6", 
            "accent_color": "#10b981"
        },
        "warm": {
            "primary_color": "#f59e0b",
            "secondary_color": "#ef4444",
            "accent_color": "#fbbf24"
        },
        "cool": {
            "primary_color": "#3b82f6",
            "secondary_color": "#06b6d4",
            "accent_color": "#8b5cf6"
        },
        "nature": {
            "primary_color": "#10b981",
            "secondary_color": "#22c55e",
            "accent_color": "#84cc16"
        },
        "sunset": {
            "primary_color": "#f59e0b",
            "secondary_color": "#ec4899",
            "accent_color": "#8b5cf6"
        },
        "fire": {
            "primary_color": "#ef4444",
            "secondary_color": "#f59e0b",
            "accent_color": "#fbbf24"
        }
    }
    
    # Available expressions
    expressions = ["idle", "happy", "thinking", "sad", "surprised", "confused", "excited"]
    
    created_files = []
    
    for scheme_name, colors in color_schemes.items():
        print(f"\n  Creating {scheme_name} avatar set...")
        
        # Create folder for this scheme
        scheme_dir = avatar_dir / scheme_name
        scheme_dir.mkdir(exist_ok=True)
        
        # Generate each expression as SVG
        expression_files = {}
        for expr in expressions:
            if expr in SPRITE_TEMPLATES:
                svg_content = generate_sprite(expr, **colors)
                svg_path = scheme_dir / f"{expr}.svg"
                svg_path.write_text(svg_content)
                expression_files[expr] = f"{expr}.svg"
                print(f"    ✓ {expr}.svg")
        
        # Create config file for this avatar
        config = {
            "name": f"{scheme_name.title()} Avatar",
            "style": scheme_name,
            "image": "idle.svg",
            "colors": colors,
            "expressions": expression_files,
            "description": f"Test avatar with {scheme_name} color scheme"
        }
        
        config_path = scheme_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        created_files.append(str(config_path))
        print(f"    ✓ config.json")
    
    # Create a simple default avatar config in the root avatar folder
    default_config = {
        "name": "Default Test Avatar",
        "image": "default/idle.svg",
        "expressions": {
            "neutral": "default/idle.svg",
            "happy": "default/happy.svg",
            "thinking": "default/thinking.svg",
            "sad": "default/sad.svg",
            "surprised": "default/surprised.svg",
            "confused": "default/confused.svg",
            "excited": "default/excited.svg"
        }
    }
    
    default_config_path = avatar_dir / "test_avatar.json"
    with open(default_config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    created_files.append(str(default_config_path))
    
    print(f"\n✅ Created {len(created_files)} avatar configs!")
    print(f"\nTo use in GUI:")
    print(f"  1. Go to Avatar tab")
    print(f"  2. Click 'Refresh' to see new configs")
    print(f"  3. Select a config from the dropdown")
    print(f"\nAvatar directory: {avatar_dir}")
    
    return created_files


if __name__ == "__main__":
    create_test_avatars()
