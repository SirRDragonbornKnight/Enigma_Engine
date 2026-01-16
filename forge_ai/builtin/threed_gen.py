"""
Built-in 3D Generator

Creates simple 3D models in OBJ format without ML models.
Generates basic geometric shapes and primitives.
"""

import math
import time
from typing import Dict, Any, List, Tuple, Optional


class Builtin3DGen:
    """
    Built-in 3D model generator using pure Python.
    Creates simple OBJ files without external dependencies.
    
    Supported shapes:
    - cube: Basic cube/box
    - sphere: UV sphere
    - cylinder: Cylinder shape
    - cone: Cone shape  
    - torus: Donut shape
    - pyramid: Pyramid
    - plane: Flat plane
    """
    
    def __init__(self):
        self.is_loaded = False
    
    def load(self) -> bool:
        """Load the generator."""
        self.is_loaded = True
        return True
    
    def unload(self):
        """Unload."""
        self.is_loaded = False
    
    def _detect_shape(self, prompt: str) -> str:
        """Detect what shape to generate from prompt."""
        prompt_lower = prompt.lower()
        
        if any(w in prompt_lower for w in ["cube", "box", "square", "block"]):
            return "cube"
        elif any(w in prompt_lower for w in ["sphere", "ball", "orb", "globe"]):
            return "sphere"
        elif any(w in prompt_lower for w in ["cylinder", "tube", "pipe", "column"]):
            return "cylinder"
        elif any(w in prompt_lower for w in ["cone", "spike", "point"]):
            return "cone"
        elif any(w in prompt_lower for w in ["torus", "donut", "ring", "hoop"]):
            return "torus"
        elif any(w in prompt_lower for w in ["pyramid", "triangle"]):
            return "pyramid"
        elif any(w in prompt_lower for w in ["plane", "flat", "floor", "ground"]):
            return "plane"
        else:
            return "cube"  # Default
    
    def _generate_cube(self, size: float = 1.0) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, ...]]]:
        """Generate cube vertices and faces."""
        s = size / 2
        vertices = [
            (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
            (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s),
        ]
        faces = [
            (1, 2, 3, 4), (5, 8, 7, 6), (1, 5, 6, 2),
            (2, 6, 7, 3), (3, 7, 8, 4), (5, 1, 4, 8),
        ]
        return vertices, faces
    
    def _generate_sphere(self, radius: float = 0.5, segments: int = 16, rings: int = 12) -> Tuple[List, List]:
        """Generate UV sphere."""
        vertices = []
        faces = []
        
        # Generate vertices
        for i in range(rings + 1):
            phi = math.pi * i / rings
            for j in range(segments):
                theta = 2 * math.pi * j / segments
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                vertices.append((x, y, z))
        
        # Generate faces
        for i in range(rings):
            for j in range(segments):
                p1 = i * segments + j + 1
                p2 = i * segments + (j + 1) % segments + 1
                p3 = (i + 1) * segments + (j + 1) % segments + 1
                p4 = (i + 1) * segments + j + 1
                
                if i == 0:
                    faces.append((p1, p4, p3))
                elif i == rings - 1:
                    faces.append((p1, p2, p4))
                else:
                    faces.append((p1, p2, p3, p4))
        
        return vertices, faces
    
    def _generate_cylinder(self, radius: float = 0.5, height: float = 1.0, segments: int = 16) -> Tuple[List, List]:
        """Generate cylinder."""
        vertices = []
        faces = []
        h = height / 2
        
        # Top center, bottom center
        vertices.append((0, h, 0))   # 1
        vertices.append((0, -h, 0))  # 2
        
        # Top ring
        for i in range(segments):
            theta = 2 * math.pi * i / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            vertices.append((x, h, z))
        
        # Bottom ring
        for i in range(segments):
            theta = 2 * math.pi * i / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            vertices.append((x, -h, z))
        
        # Top cap
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append((1, 3 + i, 3 + next_i))
        
        # Bottom cap
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append((2, 3 + segments + next_i, 3 + segments + i))
        
        # Side faces
        for i in range(segments):
            next_i = (i + 1) % segments
            top1 = 3 + i
            top2 = 3 + next_i
            bot1 = 3 + segments + i
            bot2 = 3 + segments + next_i
            faces.append((top1, top2, bot2, bot1))
        
        return vertices, faces
    
    def _generate_cone(self, radius: float = 0.5, height: float = 1.0, segments: int = 16) -> Tuple[List, List]:
        """Generate cone."""
        vertices = []
        faces = []
        
        # Apex and base center
        vertices.append((0, height, 0))  # 1 - apex
        vertices.append((0, 0, 0))       # 2 - base center
        
        # Base ring
        for i in range(segments):
            theta = 2 * math.pi * i / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            vertices.append((x, 0, z))
        
        # Side faces (triangles to apex)
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append((1, 3 + next_i, 3 + i))
        
        # Base cap
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append((2, 3 + i, 3 + next_i))
        
        return vertices, faces
    
    def _generate_torus(self, major_radius: float = 0.5, minor_radius: float = 0.2,
                        major_segments: int = 16, minor_segments: int = 12) -> Tuple[List, List]:
        """Generate torus (donut)."""
        vertices = []
        faces = []
        
        for i in range(major_segments):
            theta = 2 * math.pi * i / major_segments
            for j in range(minor_segments):
                phi = 2 * math.pi * j / minor_segments
                
                x = (major_radius + minor_radius * math.cos(phi)) * math.cos(theta)
                y = minor_radius * math.sin(phi)
                z = (major_radius + minor_radius * math.cos(phi)) * math.sin(theta)
                
                vertices.append((x, y, z))
        
        for i in range(major_segments):
            next_i = (i + 1) % major_segments
            for j in range(minor_segments):
                next_j = (j + 1) % minor_segments
                
                p1 = i * minor_segments + j + 1
                p2 = i * minor_segments + next_j + 1
                p3 = next_i * minor_segments + next_j + 1
                p4 = next_i * minor_segments + j + 1
                
                faces.append((p1, p2, p3, p4))
        
        return vertices, faces
    
    def _generate_pyramid(self, base_size: float = 1.0, height: float = 1.0) -> Tuple[List, List]:
        """Generate pyramid."""
        s = base_size / 2
        vertices = [
            (0, height, 0),    # 1 - apex
            (-s, 0, -s),       # 2
            (s, 0, -s),        # 3
            (s, 0, s),         # 4
            (-s, 0, s),        # 5
        ]
        faces = [
            (1, 3, 2),  # Front
            (1, 4, 3),  # Right
            (1, 5, 4),  # Back
            (1, 2, 5),  # Left
            (2, 3, 4, 5),  # Base
        ]
        return vertices, faces
    
    def _generate_plane(self, size: float = 2.0, subdivisions: int = 1) -> Tuple[List, List]:
        """Generate flat plane."""
        vertices = []
        faces = []
        s = size / 2
        step = size / (subdivisions + 1)
        
        # Generate grid vertices
        for j in range(subdivisions + 2):
            for i in range(subdivisions + 2):
                x = -s + i * step
                z = -s + j * step
                vertices.append((x, 0, z))
        
        # Generate faces
        cols = subdivisions + 2
        for j in range(subdivisions + 1):
            for i in range(subdivisions + 1):
                p1 = j * cols + i + 1
                p2 = j * cols + i + 2
                p3 = (j + 1) * cols + i + 2
                p4 = (j + 1) * cols + i + 1
                faces.append((p1, p2, p3, p4))
        
        return vertices, faces
    
    def _to_obj(self, vertices: List[Tuple[float, float, float]], 
                faces: List[Tuple[int, ...]], name: str = "model") -> str:
        """Convert to OBJ format string."""
        lines = [
            f"# Generated by ForgeAI Built-in 3D Generator",
            f"# Shape: {name}",
            f"o {name}",
            "",
        ]
        
        # Vertices
        for v in vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        
        lines.append("")
        
        # Faces
        for f in faces:
            face_str = "f " + " ".join(str(i) for i in f)
            lines.append(face_str)
        
        return "\n".join(lines)
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a 3D model from a text prompt."""
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        if not prompt.strip():
            return {"success": False, "error": "Empty prompt"}
        
        try:
            start = time.time()
            
            shape = self._detect_shape(prompt)
            
            # Generate the shape
            generators = {
                "cube": self._generate_cube,
                "sphere": self._generate_sphere,
                "cylinder": self._generate_cylinder,
                "cone": self._generate_cone,
                "torus": self._generate_torus,
                "pyramid": self._generate_pyramid,
                "plane": self._generate_plane,
            }
            
            gen_func = generators.get(shape, self._generate_cube)
            vertices, faces = gen_func()
            
            # Convert to OBJ
            obj_data = self._to_obj(vertices, faces, shape)
            
            return {
                "success": True,
                "obj_data": obj_data,
                "format": "obj",
                "shape": shape,
                "vertices": len(vertices),
                "faces": len(faces),
                "duration": time.time() - start,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_to_file(self, prompt: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Generate 3D model and save to file."""
        result = self.generate(prompt, **kwargs)
        
        if result["success"]:
            try:
                with open(output_path, 'w') as f:
                    f.write(result["obj_data"])
                result["output_path"] = output_path
            except Exception as e:
                result["success"] = False
                result["error"] = f"Failed to save: {e}"
        
        return result
