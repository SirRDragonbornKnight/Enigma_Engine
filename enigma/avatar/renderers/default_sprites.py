"""
Default Avatar Sprites

Built-in SVG sprite templates that don't require external assets.
"""

from typing import Dict
import base64
from io import BytesIO


# SVG sprite templates
SPRITE_TEMPLATES = {
    "idle": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes -->
    <ellipse cx="80" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    <ellipse cx="120" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    
    <!-- Pupils -->
    <circle cx="80" cy="92" r="4" fill="#1e1e2e"/>
    <circle cx="120" cy="92" r="4" fill="#1e1e2e"/>
    
    <!-- Mouth - neutral -->
    <path d="M 80 115 Q 100 120 120 115" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
</svg>""",
    
    "happy": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - happy closed -->
    <path d="M 70 90 Q 80 85 90 90" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
    <path d="M 110 90 Q 120 85 130 90" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
    
    <!-- Mouth - big smile -->
    <path d="M 70 110 Q 100 130 130 110" stroke="{accent_color}" stroke-width="4" fill="none" stroke-linecap="round"/>
    
    <!-- Blush marks -->
    <circle cx="60" cy="105" r="8" fill="{secondary_color}" opacity="0.5"/>
    <circle cx="140" cy="105" r="8" fill="{secondary_color}" opacity="0.5"/>
</svg>""",
    
    "thinking": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - looking up -->
    <ellipse cx="75" cy="85" rx="8" ry="12" fill="{accent_color}"/>
    <ellipse cx="115" cy="85" rx="8" ry="12" fill="{accent_color}"/>
    
    <!-- Pupils - looking up -->
    <circle cx="75" cy="82" r="4" fill="#1e1e2e"/>
    <circle cx="115" cy="82" r="4" fill="#1e1e2e"/>
    
    <!-- Mouth - thoughtful -->
    <line x1="85" y1="115" x2="115" y2="115" stroke="{accent_color}" stroke-width="2" stroke-linecap="round"/>
    
    <!-- Thought bubble -->
    <circle cx="140" cy="60" r="4" fill="{secondary_color}" opacity="0.6"/>
    <circle cx="150" cy="50" r="6" fill="{secondary_color}" opacity="0.6"/>
    <circle cx="165" cy="40" r="10" fill="{secondary_color}" opacity="0.6"/>
</svg>""",
    
    "sad": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - sad -->
    <ellipse cx="80" cy="90" rx="8" ry="14" fill="{accent_color}"/>
    <ellipse cx="120" cy="90" rx="8" ry="14" fill="{accent_color}"/>
    
    <!-- Pupils -->
    <circle cx="80" cy="92" r="4" fill="#1e1e2e"/>
    <circle cx="120" cy="92" r="4" fill="#1e1e2e"/>
    
    <!-- Eyebrows - sad -->
    <path d="M 70 75 Q 80 72 90 75" stroke="{accent_color}" stroke-width="2" fill="none"/>
    <path d="M 110 75 Q 120 72 130 75" stroke="{accent_color}" stroke-width="2" fill="none"/>
    
    <!-- Mouth - frown -->
    <path d="M 80 125 Q 100 115 120 125" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
</svg>""",
    
    "surprised": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - wide open -->
    <circle cx="80" cy="90" r="12" fill="{accent_color}"/>
    <circle cx="120" cy="90" r="12" fill="{accent_color}"/>
    
    <!-- Pupils - large -->
    <circle cx="80" cy="90" r="6" fill="#1e1e2e"/>
    <circle cx="120" cy="90" r="6" fill="#1e1e2e"/>
    
    <!-- Mouth - open O -->
    <circle cx="100" cy="120" r="10" fill="{accent_color}"/>
    <circle cx="100" cy="120" r="7" fill="{primary_color}"/>
</svg>""",
    
    "confused": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - asymmetric -->
    <ellipse cx="80" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    <circle cx="120" cy="88" r="6" fill="{accent_color}"/>
    
    <!-- Pupils -->
    <circle cx="80" cy="92" r="4" fill="#1e1e2e"/>
    <circle cx="120" cy="88" r="3" fill="#1e1e2e"/>
    
    <!-- Eyebrows - confused -->
    <path d="M 70 75 L 90 78" stroke="{accent_color}" stroke-width="2" stroke-linecap="round"/>
    <path d="M 110 78 L 130 75" stroke="{accent_color}" stroke-width="2" stroke-linecap="round"/>
    
    <!-- Mouth - squiggle -->
    <path d="M 75 115 Q 85 110 95 115 Q 105 120 115 115" stroke="{accent_color}" stroke-width="2" fill="none" stroke-linecap="round"/>
    
    <!-- Question mark -->
    <text x="145" y="65" font-size="20" fill="{secondary_color}">?</text>
</svg>""",
    
    "excited": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle with glow -->
    <circle cx="100" cy="100" r="85" fill="{secondary_color}" opacity="0.3"/>
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - sparkly -->
    <circle cx="80" cy="90" r="10" fill="{accent_color}"/>
    <circle cx="120" cy="90" r="10" fill="{accent_color}"/>
    
    <!-- Pupils with highlights -->
    <circle cx="80" cy="90" r="5" fill="#1e1e2e"/>
    <circle cx="120" cy="90" r="5" fill="#1e1e2e"/>
    <circle cx="77" cy="87" r="2" fill="white"/>
    <circle cx="117" cy="87" r="2" fill="white"/>
    
    <!-- Mouth - big excited smile -->
    <path d="M 65 108 Q 100 140 135 108" stroke="{accent_color}" stroke-width="4" fill="none" stroke-linecap="round"/>
    
    <!-- Sparkles -->
    <path d="M 150 70 L 152 75 L 157 73 L 153 78 L 158 80 L 152 82 L 153 87 L 150 82 L 145 84 L 148 79 L 143 77 L 148 75 Z" fill="{secondary_color}"/>
    <path d="M 40 65 L 41 68 L 44 67 L 42 70 L 45 71 L 41 72 L 42 75 L 40 72 L 37 73 L 39 70 L 36 69 L 39 68 Z" fill="{secondary_color}"/>
</svg>""",
    
    "speaking_1": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes -->
    <ellipse cx="80" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    <ellipse cx="120" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    
    <!-- Pupils -->
    <circle cx="80" cy="92" r="4" fill="#1e1e2e"/>
    <circle cx="120" cy="92" r="4" fill="#1e1e2e"/>
    
    <!-- Mouth - speaking position 1 (semi-open) -->
    <ellipse cx="100" cy="118" rx="15" ry="8" fill="{accent_color}"/>
    <ellipse cx="100" cy="116" rx="12" ry="6" fill="{primary_color}"/>
</svg>""",
    
    "speaking_2": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes -->
    <ellipse cx="80" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    <ellipse cx="120" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    
    <!-- Pupils -->
    <circle cx="80" cy="92" r="4" fill="#1e1e2e"/>
    <circle cx="120" cy="92" r="4" fill="#1e1e2e"/>
    
    <!-- Mouth - speaking position 2 (more closed) -->
    <ellipse cx="100" cy="117" rx="12" ry="6" fill="{accent_color}"/>
</svg>""",
}


def generate_sprite(
    sprite_name: str,
    primary_color: str = "#6366f1",
    secondary_color: str = "#8b5cf6",
    accent_color: str = "#10b981"
) -> str:
    """
    Generate an SVG sprite with custom colors.
    
    Args:
        sprite_name: Name of sprite template to use
        primary_color: Primary color (hex)
        secondary_color: Secondary color (hex)
        accent_color: Accent color (hex)
        
    Returns:
        SVG string with colors applied
    """
    if sprite_name not in SPRITE_TEMPLATES:
        sprite_name = "idle"
    
    template = SPRITE_TEMPLATES[sprite_name]
    
    # Replace color placeholders
    svg = template.format(
        primary_color=primary_color,
        secondary_color=secondary_color,
        accent_color=accent_color
    )
    
    return svg


def generate_sprite_png(
    sprite_name: str,
    primary_color: str = "#6366f1",
    secondary_color: str = "#8b5cf6",
    accent_color: str = "#10b981",
    size: int = 200
) -> bytes:
    """
    Generate a PNG sprite from SVG template.
    
    Requires cairosvg or PIL. Falls back to SVG if not available.
    
    Args:
        sprite_name: Name of sprite template
        primary_color: Primary color (hex)
        secondary_color: Secondary color (hex)
        accent_color: Accent color (hex)
        size: Output size in pixels
        
    Returns:
        PNG image data as bytes
    """
    svg = generate_sprite(sprite_name, primary_color, secondary_color, accent_color)
    
    try:
        # Try cairosvg first (best quality)
        import cairosvg
        png_data = cairosvg.svg2png(
            bytestring=svg.encode('utf-8'),
            output_width=size,
            output_height=size
        )
        return png_data
    except ImportError:
        pass
    
    try:
        # Try PIL with svg support
        from PIL import Image
        import io
        
        # For now, return SVG data - PIL doesn't handle SVG well without extra deps
        return svg.encode('utf-8')
    except ImportError:
        pass
    
    # Fallback: return SVG as bytes
    return svg.encode('utf-8')


def get_sprite_data_url(
    sprite_name: str,
    primary_color: str = "#6366f1",
    secondary_color: str = "#8b5cf6",
    accent_color: str = "#10b981"
) -> str:
    """
    Get sprite as a data URL for use in HTML/CSS.
    
    Args:
        sprite_name: Name of sprite template
        primary_color: Primary color (hex)
        secondary_color: Secondary color (hex)
        accent_color: Accent color (hex)
        
    Returns:
        Data URL string
    """
    svg = generate_sprite(sprite_name, primary_color, secondary_color, accent_color)
    
    # Encode as base64
    svg_bytes = svg.encode('utf-8')
    b64 = base64.b64encode(svg_bytes).decode('utf-8')
    
    return f"data:image/svg+xml;base64,{b64}"


def save_sprite(
    sprite_name: str,
    filepath: str,
    primary_color: str = "#6366f1",
    secondary_color: str = "#8b5cf6",
    accent_color: str = "#10b981"
):
    """
    Save sprite to file.
    
    Args:
        sprite_name: Name of sprite template
        filepath: Path to save to (should end in .svg or .png)
        primary_color: Primary color (hex)
        secondary_color: Secondary color (hex)
        accent_color: Accent color (hex)
    """
    from pathlib import Path
    
    path = Path(filepath)
    
    if path.suffix.lower() == '.svg':
        # Save as SVG
        svg = generate_sprite(sprite_name, primary_color, secondary_color, accent_color)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(svg)
    else:
        # Save as PNG
        png_data = generate_sprite_png(sprite_name, primary_color, secondary_color, accent_color)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(png_data)


def generate_all_sprites(
    output_dir: str,
    primary_color: str = "#6366f1",
    secondary_color: str = "#8b5cf6",
    accent_color: str = "#10b981"
):
    """
    Generate all sprite templates and save to directory.
    
    Args:
        output_dir: Directory to save sprites to
        primary_color: Primary color (hex)
        secondary_color: Secondary color (hex)
        accent_color: Accent color (hex)
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for sprite_name in SPRITE_TEMPLATES.keys():
        save_sprite(
            sprite_name,
            str(output_path / f"{sprite_name}.svg"),
            primary_color,
            secondary_color,
            accent_color
        )
