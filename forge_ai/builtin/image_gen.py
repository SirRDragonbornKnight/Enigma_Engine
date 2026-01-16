"""
Built-in Image Generator

Zero-dependency image generation using basic drawing primitives.
Creates simple procedural images without needing Stable Diffusion or any ML models.
"""

import math
import struct
import zlib
from typing import List, Tuple, Optional, Dict, Any


class BuiltinImageGen:
    """
    Built-in image generator using pure Python.
    Creates simple procedural art without any external dependencies.
    
    Supported styles:
    - gradient: Color gradients
    - pattern: Geometric patterns
    - noise: Procedural noise
    - shapes: Random shapes
    - text: Text on background
    """
    
    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self.is_loaded = False
        
    def load(self) -> bool:
        """Load the generator."""
        self.is_loaded = True
        return True
        
    def unload(self):
        """Unload."""
        self.is_loaded = False
    
    def _hash_string(self, s: str, seed: int = 0) -> int:
        """Simple hash function for generating deterministic randomness."""
        h = seed
        for c in s:
            h = ((h * 31) + ord(c)) & 0xFFFFFFFF
        return h
    
    def _pseudo_random(self, seed: int, index: int) -> float:
        """Generate pseudo-random number from seed and index."""
        x = (seed * 1103515245 + index * 12345 + 1) & 0xFFFFFFFF
        return (x & 0x7FFFFFFF) / 0x7FFFFFFF
    
    def _lerp(self, a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + (b - a) * t
    
    def _smooth_step(self, t: float) -> float:
        """Smooth step function."""
        return t * t * (3 - 2 * t)
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV to RGB."""
        h = h % 1.0
        if s == 0:
            r = g = b = int(v * 255)
            return (r, g, b)
        
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if i % 6 == 0:
            r, g, b = v, t, p
        elif i % 6 == 1:
            r, g, b = q, v, p
        elif i % 6 == 2:
            r, g, b = p, v, t
        elif i % 6 == 3:
            r, g, b = p, q, v
        elif i % 6 == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def _color_from_prompt(self, prompt: str) -> Tuple[float, float, float]:
        """Extract dominant hue from prompt words."""
        prompt_lower = prompt.lower()
        
        color_map = {
            "red": 0.0, "fire": 0.02, "flame": 0.03, "hot": 0.04,
            "orange": 0.08, "sunset": 0.07, "autumn": 0.06,
            "yellow": 0.15, "sun": 0.14, "gold": 0.12, "bright": 0.13,
            "green": 0.33, "forest": 0.35, "nature": 0.32, "grass": 0.30, "plant": 0.34,
            "cyan": 0.5, "teal": 0.48,
            "blue": 0.6, "sky": 0.58, "ocean": 0.62, "water": 0.55, "sea": 0.63,
            "purple": 0.75, "violet": 0.78, "lavender": 0.76, "magic": 0.74,
            "pink": 0.9, "rose": 0.92,
            "black": -1, "dark": -1, "night": -1,
            "white": -2, "light": -2, "bright": -2,
            "gray": -3, "grey": -3, "neutral": -3,
        }
        
        hue = 0.6  # Default blue
        saturation = 0.7
        value = 0.8
        
        for word, h in color_map.items():
            if word in prompt_lower:
                if h == -1:  # Dark
                    saturation = 0.3
                    value = 0.2
                elif h == -2:  # Light
                    saturation = 0.2
                    value = 0.95
                elif h == -3:  # Gray
                    saturation = 0.1
                    value = 0.5
                else:
                    hue = h
                break
        
        return (hue, saturation, value)
    
    def _detect_style(self, prompt: str) -> str:
        """Detect what style of image to generate."""
        prompt_lower = prompt.lower()
        
        if any(w in prompt_lower for w in ["abstract", "art", "painting", "artistic"]):
            return "abstract"
        elif any(w in prompt_lower for w in ["pattern", "geometric", "tile", "repeat"]):
            return "pattern"
        elif any(w in prompt_lower for w in ["gradient", "fade", "smooth", "blend"]):
            return "gradient"
        elif any(w in prompt_lower for w in ["noise", "texture", "rough", "grain"]):
            return "noise"
        elif any(w in prompt_lower for w in ["circle", "sphere", "ball", "dot"]):
            return "circles"
        elif any(w in prompt_lower for w in ["star", "space", "galaxy", "cosmic"]):
            return "stars"
        elif any(w in prompt_lower for w in ["wave", "water", "ocean", "flow"]):
            return "waves"
        elif any(w in prompt_lower for w in ["mountain", "landscape", "terrain"]):
            return "terrain"
        else:
            return "abstract"
    
    def _generate_gradient(self, seed: int, hue: float, sat: float, val: float) -> List[List[Tuple[int, int, int]]]:
        """Generate a gradient image."""
        pixels = []
        hue2 = (hue + self._pseudo_random(seed, 0) * 0.3) % 1.0
        
        for y in range(self.height):
            row = []
            for x in range(self.width):
                t = x / self.width
                ty = y / self.height
                
                # Mix horizontal and vertical gradients
                h = self._lerp(hue, hue2, t * 0.5 + ty * 0.5)
                s = sat * (0.8 + 0.2 * t)
                v = val * (0.9 + 0.1 * self._smooth_step(ty))
                
                row.append(self._hsv_to_rgb(h, s, v))
            pixels.append(row)
        return pixels
    
    def _generate_pattern(self, seed: int, hue: float, sat: float, val: float) -> List[List[Tuple[int, int, int]]]:
        """Generate a geometric pattern."""
        pixels = []
        scale = 8 + int(self._pseudo_random(seed, 0) * 24)
        
        for y in range(self.height):
            row = []
            for x in range(self.width):
                # Create checkerboard-like pattern with variations
                px = x // scale
                py = y // scale
                
                # Various patterns based on position
                pattern_val = (px + py) % 2
                if self._pseudo_random(seed, 1) > 0.5:
                    pattern_val = (px * py) % 3
                
                h = hue + pattern_val * 0.1
                v = val * (0.6 + 0.4 * pattern_val / 2)
                
                row.append(self._hsv_to_rgb(h, sat, v))
            pixels.append(row)
        return pixels
    
    def _generate_noise(self, seed: int, hue: float, sat: float, val: float) -> List[List[Tuple[int, int, int]]]:
        """Generate noise texture."""
        pixels = []
        
        for y in range(self.height):
            row = []
            for x in range(self.width):
                # Simple value noise
                noise_val = self._pseudo_random(seed, x * self.height + y)
                
                # Add some larger scale variation
                large_scale = self._pseudo_random(seed, (x // 32) * 1000 + (y // 32))
                noise_val = noise_val * 0.5 + large_scale * 0.5
                
                h = hue + (noise_val - 0.5) * 0.1
                v = val * (0.3 + 0.7 * noise_val)
                
                row.append(self._hsv_to_rgb(h, sat, v))
            pixels.append(row)
        return pixels
    
    def _generate_circles(self, seed: int, hue: float, sat: float, val: float) -> List[List[Tuple[int, int, int]]]:
        """Generate circles/bubbles."""
        # Create background
        pixels = [[self._hsv_to_rgb(hue, sat * 0.3, val * 0.2) for _ in range(self.width)] for _ in range(self.height)]
        
        # Add circles
        num_circles = 10 + int(self._pseudo_random(seed, 0) * 20)
        for i in range(num_circles):
            cx = int(self._pseudo_random(seed, i * 3) * self.width)
            cy = int(self._pseudo_random(seed, i * 3 + 1) * self.height)
            radius = 20 + int(self._pseudo_random(seed, i * 3 + 2) * 80)
            circle_hue = (hue + self._pseudo_random(seed, i) * 0.3) % 1.0
            
            for y in range(max(0, cy - radius), min(self.height, cy + radius)):
                for x in range(max(0, cx - radius), min(self.width, cx + radius)):
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if dist < radius:
                        # Soft edge
                        alpha = 1.0 - (dist / radius)
                        alpha = alpha ** 0.5  # Softer falloff
                        
                        old_color = pixels[y][x]
                        new_color = self._hsv_to_rgb(circle_hue, sat, val * (0.5 + 0.5 * alpha))
                        
                        r = int(old_color[0] * (1 - alpha) + new_color[0] * alpha)
                        g = int(old_color[1] * (1 - alpha) + new_color[1] * alpha)
                        b = int(old_color[2] * (1 - alpha) + new_color[2] * alpha)
                        pixels[y][x] = (r, g, b)
        
        return pixels
    
    def _generate_stars(self, seed: int, hue: float, sat: float, val: float) -> List[List[Tuple[int, int, int]]]:
        """Generate starfield."""
        # Dark background
        pixels = [[self._hsv_to_rgb(hue, sat * 0.5, 0.05) for _ in range(self.width)] for _ in range(self.height)]
        
        # Add stars
        num_stars = 50 + int(self._pseudo_random(seed, 0) * 150)
        for i in range(num_stars):
            x = int(self._pseudo_random(seed, i * 2) * self.width)
            y = int(self._pseudo_random(seed, i * 2 + 1) * self.height)
            brightness = self._pseudo_random(seed, i * 2 + 2)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                star_hue = self._pseudo_random(seed, i * 3) * 0.2  # White to yellow-ish
                pixels[y][x] = self._hsv_to_rgb(star_hue, 0.2, 0.5 + brightness * 0.5)
                
                # Add glow for bright stars
                if brightness > 0.7:
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                glow = brightness * 0.3
                                old = pixels[ny][nx]
                                pixels[ny][nx] = (
                                    min(255, int(old[0] + 255 * glow)),
                                    min(255, int(old[1] + 255 * glow)),
                                    min(255, int(old[2] + 255 * glow))
                                )
        
        return pixels
    
    def _generate_waves(self, seed: int, hue: float, sat: float, val: float) -> List[List[Tuple[int, int, int]]]:
        """Generate wave pattern."""
        pixels = []
        freq = 0.02 + self._pseudo_random(seed, 0) * 0.03
        
        for y in range(self.height):
            row = []
            for x in range(self.width):
                wave1 = math.sin(x * freq + y * freq * 0.5) * 0.5 + 0.5
                wave2 = math.sin(x * freq * 1.5 - y * freq) * 0.5 + 0.5
                wave = wave1 * 0.6 + wave2 * 0.4
                
                h = hue + wave * 0.1
                v = val * (0.4 + 0.6 * wave)
                
                row.append(self._hsv_to_rgb(h, sat, v))
            pixels.append(row)
        return pixels
    
    def _generate_terrain(self, seed: int, hue: float, sat: float, val: float) -> List[List[Tuple[int, int, int]]]:
        """Generate simple terrain/landscape."""
        pixels = []
        
        # Generate horizon line
        horizon = int(self.height * (0.4 + self._pseudo_random(seed, 0) * 0.2))
        
        # Mountain peaks
        peaks = []
        for i in range(5):
            x = int(self._pseudo_random(seed, i * 2 + 10) * self.width)
            h = int(self._pseudo_random(seed, i * 2 + 11) * horizon * 0.8)
            peaks.append((x, horizon - h))
        peaks.sort()
        
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if y < horizon:
                    # Sky with gradient
                    sky_t = y / horizon
                    h = 0.58  # Blue
                    s = 0.5 + 0.3 * (1 - sky_t)
                    v = 0.7 + 0.25 * (1 - sky_t)
                    
                    # Check if behind mountain
                    is_mountain = False
                    for i, (px, py) in enumerate(peaks):
                        if i < len(peaks) - 1:
                            next_px, next_py = peaks[i + 1]
                            if px <= x <= next_px:
                                t = (x - px) / (next_px - px) if next_px != px else 0
                                mountain_height = py * (1 - t) + next_py * t
                                if y > mountain_height:
                                    is_mountain = True
                                    h = hue
                                    s = sat * 0.4
                                    v = val * (0.2 + 0.1 * (y - mountain_height) / (horizon - mountain_height))
                                break
                    
                    row.append(self._hsv_to_rgb(h, s, v))
                else:
                    # Ground
                    ground_t = (y - horizon) / (self.height - horizon)
                    h = hue
                    s = sat * 0.6
                    v = val * (0.4 - 0.2 * ground_t)
                    row.append(self._hsv_to_rgb(h, s, v))
            pixels.append(row)
        return pixels
    
    def _generate_abstract(self, seed: int, hue: float, sat: float, val: float) -> List[List[Tuple[int, int, int]]]:
        """Generate abstract art combining multiple techniques."""
        # Start with gradient
        pixels = self._generate_gradient(seed, hue, sat, val)
        
        # Add some circles
        num_shapes = 5 + int(self._pseudo_random(seed, 100) * 10)
        for i in range(num_shapes):
            cx = int(self._pseudo_random(seed, i * 5 + 200) * self.width)
            cy = int(self._pseudo_random(seed, i * 5 + 201) * self.height)
            radius = 30 + int(self._pseudo_random(seed, i * 5 + 202) * 100)
            shape_hue = (hue + self._pseudo_random(seed, i * 5 + 203) * 0.5) % 1.0
            alpha = 0.3 + self._pseudo_random(seed, i * 5 + 204) * 0.4
            
            for y in range(max(0, cy - radius), min(self.height, cy + radius)):
                for x in range(max(0, cx - radius), min(self.width, cx + radius)):
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if dist < radius:
                        edge_alpha = alpha * (1.0 - dist / radius) ** 2
                        
                        old_color = pixels[y][x]
                        new_color = self._hsv_to_rgb(shape_hue, sat * 0.8, val * 0.9)
                        
                        r = int(old_color[0] * (1 - edge_alpha) + new_color[0] * edge_alpha)
                        g = int(old_color[1] * (1 - edge_alpha) + new_color[1] * edge_alpha)
                        b = int(old_color[2] * (1 - edge_alpha) + new_color[2] * edge_alpha)
                        pixels[y][x] = (min(255, r), min(255, g), min(255, b))
        
        return pixels
    
    def _pixels_to_png(self, pixels: List[List[Tuple[int, int, int]]]) -> bytes:
        """Convert pixel data to PNG format (pure Python implementation)."""
        height = len(pixels)
        width = len(pixels[0]) if pixels else 0
        
        def crc32(data: bytes) -> int:
            return zlib.crc32(data) & 0xFFFFFFFF
        
        def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
            length = len(data)
            chunk = chunk_type + data
            return struct.pack(">I", length) + chunk + struct.pack(">I", crc32(chunk))
        
        # PNG signature
        png_data = b'\x89PNG\r\n\x1a\n'
        
        # IHDR chunk
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        png_data += png_chunk(b'IHDR', ihdr_data)
        
        # IDAT chunk (image data) - Optimized with bytearray
        raw_data = bytearray()
        for row in pixels:
            raw_data.append(0)  # Filter type: None
            for r, g, b in row:
                raw_data.extend((r, g, b))
        
        compressed = zlib.compress(bytes(raw_data), 6)  # Level 6 is faster than 9
        png_data += png_chunk(b'IDAT', compressed)
        
        # IEND chunk
        png_data += png_chunk(b'IEND', b'')
        
        return png_data
    
    def generate(self, prompt: str, width: Optional[int] = None, height: Optional[int] = None,
                 seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Generate an image from a text prompt."""
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        if not prompt.strip():
            return {"success": False, "error": "Empty prompt"}
        
        # Update dimensions if provided
        if width:
            self.width = width
        if height:
            self.height = height
        
        # Get seed from prompt if not provided
        if seed is None:
            seed = self._hash_string(prompt)
        
        try:
            import time
            start = time.time()
            
            # Detect style and colors from prompt
            style = self._detect_style(prompt)
            hue, sat, val = self._color_from_prompt(prompt)
            
            # Generate based on style
            generators = {
                "gradient": self._generate_gradient,
                "pattern": self._generate_pattern,
                "noise": self._generate_noise,
                "circles": self._generate_circles,
                "stars": self._generate_stars,
                "waves": self._generate_waves,
                "terrain": self._generate_terrain,
                "abstract": self._generate_abstract,
            }
            
            generator = generators.get(style, self._generate_abstract)
            pixels = generator(seed, hue, sat, val)
            
            # Convert to PNG
            png_data = self._pixels_to_png(pixels)
            
            return {
                "success": True,
                "image_data": png_data,
                "width": self.width,
                "height": self.height,
                "style": style,
                "duration": time.time() - start,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_to_file(self, prompt: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Generate image and save to file."""
        result = self.generate(prompt, **kwargs)
        
        if result["success"]:
            try:
                with open(output_path, 'wb') as f:
                    f.write(result["image_data"])
                result["output_path"] = output_path
            except Exception as e:
                result["success"] = False
                result["error"] = f"Failed to save: {e}"
        
        return result
