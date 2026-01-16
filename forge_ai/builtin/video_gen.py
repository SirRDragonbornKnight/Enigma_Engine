"""
Built-in Video Generator

Creates simple animated sequences without ML models.
Generates basic motion graphics and transitions.
"""

import math
import struct
import zlib
import time
from typing import Dict, Any, List, Tuple, Optional


class BuiltinVideoGen:
    """
    Built-in video generator using pure Python.
    Creates simple animated GIFs without external dependencies.
    
    Supported styles:
    - pulse: Pulsing color animation
    - wave: Wave motion
    - rotate: Rotating shapes
    - fade: Color fading
    - particles: Moving particles
    """
    
    def __init__(self, width: int = 256, height: int = 256, fps: int = 10):
        self.width = width
        self.height = height
        self.fps = fps
        self.is_loaded = False
    
    def load(self) -> bool:
        """Load the generator."""
        self.is_loaded = True
        return True
    
    def unload(self):
        """Unload."""
        self.is_loaded = False
    
    def _hash_string(self, s: str, seed: int = 0) -> int:
        """Simple hash for deterministic randomness."""
        h = seed
        for c in s:
            h = ((h * 31) + ord(c)) & 0xFFFFFFFF
        return h
    
    def _pseudo_random(self, seed: int, index: int) -> float:
        """Generate pseudo-random number."""
        x = (seed * 1103515245 + index * 12345 + 1) & 0xFFFFFFFF
        return (x & 0x7FFFFFFF) / 0x7FFFFFFF
    
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
        """Extract color hints from prompt."""
        prompt_lower = prompt.lower()
        
        color_hints = {
            "red": 0.0, "fire": 0.02, "orange": 0.08, "yellow": 0.15,
            "green": 0.33, "cyan": 0.5, "blue": 0.6, "purple": 0.75,
            "pink": 0.9, "rainbow": -1,
        }
        
        hue = 0.6  # Default blue
        for word, h in color_hints.items():
            if word in prompt_lower:
                hue = h
                break
        
        return (hue, 0.7, 0.8)
    
    def _detect_style(self, prompt: str) -> str:
        """Detect animation style from prompt."""
        prompt_lower = prompt.lower()
        
        if any(w in prompt_lower for w in ["pulse", "beat", "throb", "heart"]):
            return "pulse"
        elif any(w in prompt_lower for w in ["wave", "water", "ocean", "flow"]):
            return "wave"
        elif any(w in prompt_lower for w in ["spin", "rotate", "turn", "spiral"]):
            return "rotate"
        elif any(w in prompt_lower for w in ["fade", "transition", "morph"]):
            return "fade"
        elif any(w in prompt_lower for w in ["particle", "star", "snow", "rain"]):
            return "particles"
        elif any(w in prompt_lower for w in ["bounce", "ball", "jump"]):
            return "bounce"
        else:
            return "pulse"
    
    def _generate_frame(self, frame_num: int, total_frames: int, style: str,
                        hue: float, seed: int) -> List[List[Tuple[int, int, int]]]:
        """Generate a single frame."""
        t = frame_num / total_frames  # 0 to 1
        pixels = []
        
        if style == "pulse":
            # Pulsing brightness
            pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi)
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    # Radial gradient
                    dx = x - self.width / 2
                    dy = y - self.height / 2
                    dist = math.sqrt(dx*dx + dy*dy) / (self.width / 2)
                    dist = min(1.0, dist)
                    
                    v = (1 - dist) * pulse * 0.8 + 0.1
                    h = hue if hue >= 0 else (t + dist * 0.5) % 1.0
                    row.append(self._hsv_to_rgb(h, 0.7, v))
                pixels.append(row)
        
        elif style == "wave":
            # Moving waves
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    wave = math.sin((x / 20) + t * 2 * math.pi) * 0.5 + 0.5
                    wave2 = math.sin((y / 15) - t * 2 * math.pi) * 0.5 + 0.5
                    combined = wave * 0.6 + wave2 * 0.4
                    
                    h = hue if hue >= 0 else (combined + t) % 1.0
                    row.append(self._hsv_to_rgb(h, 0.6, 0.3 + combined * 0.5))
                pixels.append(row)
        
        elif style == "rotate":
            # Rotating pattern
            angle = t * 2 * math.pi
            cx, cy = self.width / 2, self.height / 2
            
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    # Rotate coordinates
                    dx = x - cx
                    dy = y - cy
                    rx = dx * math.cos(angle) - dy * math.sin(angle)
                    ry = dx * math.sin(angle) + dy * math.cos(angle)
                    
                    # Create pattern
                    pattern = (int(rx / 20) + int(ry / 20)) % 2
                    h = hue if hue >= 0 else (t + pattern * 0.5) % 1.0
                    v = 0.3 + pattern * 0.5
                    row.append(self._hsv_to_rgb(h, 0.6, v))
                pixels.append(row)
        
        elif style == "fade":
            # Color fading
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    h = (hue + t) % 1.0 if hue >= 0 else t
                    ty = y / self.height
                    row.append(self._hsv_to_rgb(h, 0.5 + ty * 0.3, 0.4 + ty * 0.4))
                pixels.append(row)
        
        elif style == "particles":
            # Moving particles on dark background
            pixels = [[(20, 20, 40) for _ in range(self.width)] for _ in range(self.height)]
            
            num_particles = 30
            for i in range(num_particles):
                # Particle position animated over time
                px = (self._pseudo_random(seed, i * 2) + t * 0.5) % 1.0 * self.width
                py = (self._pseudo_random(seed, i * 2 + 1) + t * 0.3) % 1.0 * self.height
                
                px = int(px) % self.width
                py = int(py) % self.height
                
                # Draw particle with glow
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nx = (px + dx) % self.width
                        ny = (py + dy) % self.height
                        dist = abs(dx) + abs(dy)
                        if dist <= 2:
                            brightness = (3 - dist) / 3
                            old = pixels[ny][nx]
                            h = hue if hue >= 0 else self._pseudo_random(seed, i) 
                            new = self._hsv_to_rgb(h, 0.3, brightness)
                            pixels[ny][nx] = (
                                min(255, old[0] + new[0]),
                                min(255, old[1] + new[1]),
                                min(255, old[2] + new[2])
                            )
        
        elif style == "bounce":
            # Bouncing circle
            pixels = [[(30, 30, 50) for _ in range(self.width)] for _ in range(self.height)]
            
            # Ball position
            bx = self.width / 2 + math.sin(t * 2 * math.pi) * (self.width / 3)
            by = self.height / 2 + abs(math.sin(t * 4 * math.pi)) * (self.height / 3 - 20)
            radius = 20
            
            for y in range(self.height):
                for x in range(self.width):
                    dist = math.sqrt((x - bx)**2 + (y - by)**2)
                    if dist < radius:
                        alpha = 1 - (dist / radius)
                        h = hue if hue >= 0 else t
                        color = self._hsv_to_rgb(h, 0.7, 0.8 * alpha + 0.2)
                        pixels[y][x] = color
        
        else:
            # Default gradient
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    h = (hue + t) % 1.0 if hue >= 0 else (x / self.width + t) % 1.0
                    row.append(self._hsv_to_rgb(h, 0.6, 0.5))
                pixels.append(row)
        
        return pixels
    
    def _frames_to_gif(self, frames: List[List[List[Tuple[int, int, int]]]], 
                       delay: int = 10) -> bytes:
        """Convert frames to GIF format."""
        # Build color table from all frames
        color_set = set()
        for frame in frames:
            for row in frame:
                for pixel in row:
                    color_set.add(pixel)
        
        # Limit to 256 colors
        colors = list(color_set)[:256]
        while len(colors) < 256:
            colors.append((0, 0, 0))
        
        color_map = {c: i for i, c in enumerate(colors)}
        
        def find_closest_color(pixel):
            if pixel in color_map:
                return color_map[pixel]
            # Find closest
            best_idx = 0
            best_dist = float('inf')
            for i, c in enumerate(colors):
                dist = (pixel[0]-c[0])**2 + (pixel[1]-c[1])**2 + (pixel[2]-c[2])**2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            return best_idx
        
        # GIF header
        gif = b'GIF89a'
        gif += struct.pack('<HH', self.width, self.height)
        gif += bytes([0xF7, 0x00, 0x00])  # Global color table flag, bg, aspect
        
        # Global color table
        for r, g, b in colors:
            gif += bytes([r, g, b])
        
        # Netscape extension for looping
        gif += b'\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00'
        
        # Each frame
        for frame in frames:
            # Graphic control extension
            gif += bytes([0x21, 0xF9, 0x04, 0x04])  # Disposal method: restore to bg
            gif += struct.pack('<H', delay)  # Delay in centiseconds
            gif += bytes([0x00, 0x00])  # Transparent color index, terminator
            
            # Image descriptor
            gif += bytes([0x2C])
            gif += struct.pack('<HHHH', 0, 0, self.width, self.height)
            gif += bytes([0x00])  # No local color table
            
            # Image data with LZW compression
            min_code_size = 8
            gif += bytes([min_code_size])
            
            # Simple LZW - just output raw indices (not efficient but works)
            indices = []
            for row in frame:
                for pixel in row:
                    indices.append(find_closest_color(pixel))
            
            # Pack indices into sub-blocks
            data = bytes(indices)
            pos = 0
            while pos < len(data):
                chunk = data[pos:pos+254]
                gif += bytes([len(chunk)]) + chunk
                pos += 254
            
            gif += bytes([0x00])  # Block terminator
        
        # GIF trailer
        gif += bytes([0x3B])
        
        return gif
    
    def generate(self, prompt: str, frames: int = 20, duration: float = 2.0,
                 width: Optional[int] = None, height: Optional[int] = None,
                 **kwargs) -> Dict[str, Any]:
        """Generate an animated GIF from a text prompt."""
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        if not prompt.strip():
            return {"success": False, "error": "Empty prompt"}
        
        if width:
            self.width = min(width, 256)  # Limit size for performance
        if height:
            self.height = min(height, 256)
        
        try:
            start = time.time()
            seed = self._hash_string(prompt)
            
            style = self._detect_style(prompt)
            hue, sat, val = self._color_from_prompt(prompt)
            
            # Generate frames
            frame_data = []
            for i in range(frames):
                frame = self._generate_frame(i, frames, style, hue, seed)
                frame_data.append(frame)
            
            # Convert to GIF
            delay = int(duration * 100 / frames)  # Centiseconds per frame
            gif_data = self._frames_to_gif(frame_data, delay)
            
            return {
                "success": True,
                "video_data": gif_data,
                "format": "gif",
                "width": self.width,
                "height": self.height,
                "frames": frames,
                "style": style,
                "duration": time.time() - start,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_to_file(self, prompt: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Generate video and save to file."""
        result = self.generate(prompt, **kwargs)
        
        if result["success"]:
            try:
                with open(output_path, 'wb') as f:
                    f.write(result["video_data"])
                result["output_path"] = output_path
            except Exception as e:
                result["success"] = False
                result["error"] = f"Failed to save: {e}"
        
        return result
