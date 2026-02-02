#!/usr/bin/env python3
"""
ForgeAI 3D Generation Example
==============================

Complete example showing how to generate 3D models including:
- Text-to-3D generation (Shap-E, Point-E)
- Image-to-3D conversion
- 3D model export (OBJ, GLB, STL)
- 3D model viewing and manipulation

3D generation allows creating 3D models from text descriptions or
images, useful for games, VR, 3D printing, and visualization.

Dependencies:
    pip install torch  # PyTorch
    pip install trimesh  # 3D mesh operations
    pip install shap-e  # OpenAI's Shap-E (optional)

Run: python examples/threed_example.py
"""

import time
import math
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass


# =============================================================================
# 3D Generation Configuration
# =============================================================================

@dataclass
class ThreeDConfig:
    """3D generation configuration."""
    output_dir: str = "outputs/3d"
    resolution: int = 64  # Voxel/point cloud resolution
    num_inference_steps: int = 64
    guidance_scale: float = 15.0
    export_formats: List[str] = None
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["obj", "glb"]


# =============================================================================
# 3D Model Representation
# =============================================================================

@dataclass
class Point3D:
    """A point in 3D space."""
    x: float
    y: float
    z: float


@dataclass
class Face:
    """A face (triangle) of a 3D mesh."""
    v1: int  # Vertex index
    v2: int
    v3: int


class Mesh3D:
    """
    Simple 3D mesh representation.
    
    A mesh consists of vertices (3D points) and faces (triangles
    connecting vertices).
    """
    
    def __init__(self):
        self.vertices: List[Point3D] = []
        self.faces: List[Face] = []
        self.name: str = "mesh"
    
    def add_vertex(self, x: float, y: float, z: float) -> int:
        """Add a vertex and return its index."""
        self.vertices.append(Point3D(x, y, z))
        return len(self.vertices) - 1
    
    def add_face(self, v1: int, v2: int, v3: int):
        """Add a triangular face."""
        self.faces.append(Face(v1, v2, v3))
    
    def get_bounds(self) -> Tuple[Point3D, Point3D]:
        """Get bounding box (min, max points)."""
        if not self.vertices:
            return Point3D(0, 0, 0), Point3D(0, 0, 0)
        
        min_p = Point3D(
            min(v.x for v in self.vertices),
            min(v.y for v in self.vertices),
            min(v.z for v in self.vertices)
        )
        max_p = Point3D(
            max(v.x for v in self.vertices),
            max(v.y for v in self.vertices),
            max(v.z for v in self.vertices)
        )
        return min_p, max_p
    
    def export_obj(self, path: str) -> bool:
        """Export mesh to OBJ format."""
        try:
            with open(path, 'w') as f:
                f.write(f"# {self.name}\n")
                f.write(f"# Vertices: {len(self.vertices)}\n")
                f.write(f"# Faces: {len(self.faces)}\n\n")
                
                # Write vertices
                for v in self.vertices:
                    f.write(f"v {v.x:.6f} {v.y:.6f} {v.z:.6f}\n")
                
                f.write("\n")
                
                # Write faces (1-indexed in OBJ format)
                for face in self.faces:
                    f.write(f"f {face.v1+1} {face.v2+1} {face.v3+1}\n")
            
            print(f"Exported to: {path}")
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    @staticmethod
    def create_cube(size: float = 1.0) -> 'Mesh3D':
        """Create a simple cube mesh."""
        mesh = Mesh3D()
        mesh.name = "cube"
        
        s = size / 2
        
        # 8 vertices
        mesh.add_vertex(-s, -s, -s)  # 0
        mesh.add_vertex( s, -s, -s)  # 1
        mesh.add_vertex( s,  s, -s)  # 2
        mesh.add_vertex(-s,  s, -s)  # 3
        mesh.add_vertex(-s, -s,  s)  # 4
        mesh.add_vertex( s, -s,  s)  # 5
        mesh.add_vertex( s,  s,  s)  # 6
        mesh.add_vertex(-s,  s,  s)  # 7
        
        # 12 faces (2 per side)
        # Front
        mesh.add_face(0, 1, 2)
        mesh.add_face(0, 2, 3)
        # Back
        mesh.add_face(4, 6, 5)
        mesh.add_face(4, 7, 6)
        # Top
        mesh.add_face(3, 2, 6)
        mesh.add_face(3, 6, 7)
        # Bottom
        mesh.add_face(0, 5, 1)
        mesh.add_face(0, 4, 5)
        # Right
        mesh.add_face(1, 5, 6)
        mesh.add_face(1, 6, 2)
        # Left
        mesh.add_face(0, 3, 7)
        mesh.add_face(0, 7, 4)
        
        return mesh


# =============================================================================
# 3D Generators
# =============================================================================

class LocalThreeDGenerator:
    """
    Local 3D generation using Shap-E or similar.
    
    Shap-E is OpenAI's text-to-3D model that generates
    3D meshes from text descriptions.
    """
    
    def __init__(self, model: str = "openai/shap-e"):
        self.model_name = model
        self.model = None
        self.is_loaded = False
        self._use_fallback = True
    
    def _log(self, message: str):
        print(f"[3D Generator] {message}")
    
    def load(self) -> bool:
        """Load 3D generation model."""
        self._log(f"Loading model: {self.model_name}")
        
        try:
            # Try to load Shap-E
            import torch
            from shap_e.diffusion.sample import sample_latents
            from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
            from shap_e.models.download import load_model, load_config
            
            self._log("Loading Shap-E...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.xm = load_model('transmitter', device=device)
            self.model = load_model('text300M', device=device)
            self.diffusion = diffusion_from_config(load_config('diffusion'))
            
            self.is_loaded = True
            self._use_fallback = False
            self._log("Shap-E loaded successfully")
            return True
            
        except ImportError:
            self._log("Shap-E not installed, using fallback generator")
            self._use_fallback = True
            self.is_loaded = True
            return True
        except Exception as e:
            self._log(f"Error loading model: {e}")
            self._use_fallback = True
            self.is_loaded = True
            return True
    
    def generate(self, prompt: str, config: ThreeDConfig) -> Optional[str]:
        """
        Generate 3D model from text prompt.
        
        Args:
            prompt: Text description of 3D object
            config: Generation configuration
            
        Returns:
            Path to generated 3D model file
        """
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        
        self._log(f"Generating 3D model: '{prompt}'")
        
        if self._use_fallback:
            return self._generate_fallback(prompt, config, timestamp)
        else:
            return self._generate_shape(prompt, config, timestamp)
    
    def _generate_shape(self, prompt: str, config: ThreeDConfig, 
                        timestamp: int) -> Optional[str]:
        """Generate using Shap-E."""
        import torch
        from shap_e.diffusion.sample import sample_latents
        from shap_e.util.notebooks import decode_latent_mesh
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 1
        
        self._log(f"Sampling latents ({config.num_inference_steps} steps)...")
        
        latents = sample_latents(
            batch_size=batch_size,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=config.guidance_scale,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            device=device,
        )
        
        self._log("Decoding mesh...")
        
        for i, latent in enumerate(latents):
            mesh = decode_latent_mesh(self.xm, latent).tri_mesh()
            
            # Export to OBJ
            obj_path = Path(config.output_dir) / f"model_{timestamp}.obj"
            with open(obj_path, 'w') as f:
                mesh.write_obj(f)
            
            self._log(f"Saved to: {obj_path}")
            return str(obj_path)
        
        return None
    
    def _generate_fallback(self, prompt: str, config: ThreeDConfig,
                           timestamp: int) -> Optional[str]:
        """Generate simple fallback mesh."""
        self._log("Using fallback generator (simple shapes)")
        
        # Create simple mesh based on keywords in prompt
        mesh = Mesh3D()
        mesh.name = f"generated_{prompt[:20].replace(' ', '_')}"
        
        prompt_lower = prompt.lower()
        
        if 'cube' in prompt_lower or 'box' in prompt_lower:
            mesh = Mesh3D.create_cube(1.0)
        elif 'sphere' in prompt_lower or 'ball' in prompt_lower:
            mesh = self._create_sphere(0.5, 16)
        elif 'pyramid' in prompt_lower:
            mesh = self._create_pyramid(1.0, 1.0)
        else:
            # Default to cube
            mesh = Mesh3D.create_cube(1.0)
        
        # Export
        obj_path = Path(config.output_dir) / f"model_{timestamp}.obj"
        mesh.export_obj(str(obj_path))
        
        return str(obj_path)
    
    def _create_sphere(self, radius: float, segments: int) -> Mesh3D:
        """Create a UV sphere."""
        mesh = Mesh3D()
        mesh.name = "sphere"
        
        # Create vertices
        for i in range(segments + 1):
            lat = math.pi * i / segments - math.pi / 2
            for j in range(segments):
                lon = 2 * math.pi * j / segments
                x = radius * math.cos(lat) * math.cos(lon)
                y = radius * math.cos(lat) * math.sin(lon)
                z = radius * math.sin(lat)
                mesh.add_vertex(x, y, z)
        
        # Create faces
        for i in range(segments):
            for j in range(segments):
                v1 = i * segments + j
                v2 = i * segments + (j + 1) % segments
                v3 = (i + 1) * segments + (j + 1) % segments
                v4 = (i + 1) * segments + j
                
                mesh.add_face(v1, v2, v3)
                mesh.add_face(v1, v3, v4)
        
        return mesh
    
    def _create_pyramid(self, base: float, height: float) -> Mesh3D:
        """Create a pyramid."""
        mesh = Mesh3D()
        mesh.name = "pyramid"
        
        b = base / 2
        
        # Base vertices
        mesh.add_vertex(-b, -b, 0)  # 0
        mesh.add_vertex( b, -b, 0)  # 1
        mesh.add_vertex( b,  b, 0)  # 2
        mesh.add_vertex(-b,  b, 0)  # 3
        # Apex
        mesh.add_vertex(0, 0, height)  # 4
        
        # Base
        mesh.add_face(0, 2, 1)
        mesh.add_face(0, 3, 2)
        # Sides
        mesh.add_face(0, 1, 4)
        mesh.add_face(1, 2, 4)
        mesh.add_face(2, 3, 4)
        mesh.add_face(3, 0, 4)
        
        return mesh


class CloudThreeDGenerator:
    """
    Cloud-based 3D generation (Replicate, etc.).
    
    Uses cloud APIs for 3D generation without local GPU.
    """
    
    def __init__(self, model: str = "cjwbw/shap-e"):
        self.model = model
        self.api_key = os.environ.get("REPLICATE_API_TOKEN")
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, config: ThreeDConfig) -> Optional[str]:
        """Generate 3D model via cloud API."""
        if not self.api_key:
            print("REPLICATE_API_TOKEN not set")
            return None
        
        try:
            import replicate
            
            print(f"[Cloud 3D] Generating: '{prompt}'")
            
            output = replicate.run(
                self.model,
                input={"prompt": prompt}
            )
            
            if output:
                import urllib.request
                timestamp = int(time.time())
                output_path = Path(config.output_dir) / f"model_{timestamp}.glb"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                url = output[0] if isinstance(output, list) else output
                urllib.request.urlretrieve(url, str(output_path))
                
                print(f"[Cloud 3D] Saved to: {output_path}")
                return str(output_path)
            
            return None
            
        except ImportError:
            print("replicate not installed")
            return None


# =============================================================================
# Image to 3D Converter
# =============================================================================

class ImageTo3DConverter:
    """
    Convert 2D images to 3D models.
    
    Uses depth estimation and mesh generation to create
    3D models from single images.
    """
    
    def __init__(self):
        self.depth_model = None
        self.is_loaded = False
    
    def _log(self, message: str):
        print(f"[Image-to-3D] {message}")
    
    def load(self) -> bool:
        """Load depth estimation model."""
        self._log("Loading depth estimation model...")
        
        try:
            from transformers import pipeline
            
            self.depth_model = pipeline(
                "depth-estimation", 
                model="Intel/dpt-hybrid-midas"
            )
            self.is_loaded = True
            return True
            
        except ImportError:
            self._log("transformers not installed, using fallback")
            self.is_loaded = True
            return True
    
    def convert(self, image_path: str, output_path: str) -> Optional[str]:
        """
        Convert image to 3D model.
        
        Args:
            image_path: Path to input image
            output_path: Path for output 3D model
            
        Returns:
            Path to generated 3D model
        """
        self._log(f"Converting image: {image_path}")
        
        # In real implementation:
        # 1. Load image
        # 2. Estimate depth map
        # 3. Create point cloud from depth
        # 4. Generate mesh from points
        # 5. Export mesh
        
        # Simulated - create simple mesh
        mesh = Mesh3D.create_cube(1.0)
        mesh.name = f"from_image_{Path(image_path).stem}"
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        mesh.export_obj(str(output))
        
        return str(output)


# =============================================================================
# 3D Model Utilities
# =============================================================================

class ThreeDUtils:
    """Utilities for working with 3D models."""
    
    @staticmethod
    def convert_format(input_path: str, output_format: str) -> Optional[str]:
        """
        Convert 3D model to different format.
        
        Args:
            input_path: Path to input model
            output_format: Target format (obj, glb, stl, ply)
            
        Returns:
            Path to converted model
        """
        try:
            import trimesh
            
            mesh = trimesh.load(input_path)
            
            output_path = Path(input_path).with_suffix(f".{output_format}")
            mesh.export(str(output_path))
            
            print(f"Converted to: {output_path}")
            return str(output_path)
            
        except ImportError:
            print("trimesh not installed. Install with: pip install trimesh")
            return None
    
    @staticmethod
    def scale_model(input_path: str, scale: float) -> Optional[str]:
        """Scale a 3D model."""
        try:
            import trimesh
            
            mesh = trimesh.load(input_path)
            mesh.apply_scale(scale)
            
            output_path = Path(input_path).with_stem(
                Path(input_path).stem + f"_scaled_{scale}"
            )
            mesh.export(str(output_path))
            
            return str(output_path)
            
        except ImportError:
            print("trimesh not installed")
            return None
    
    @staticmethod
    def get_model_info(path: str) -> Dict:
        """Get information about a 3D model."""
        try:
            import trimesh
            
            mesh = trimesh.load(path)
            
            return {
                "path": path,
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "bounds": mesh.bounds.tolist(),
                "volume": mesh.volume if mesh.is_watertight else "N/A",
                "is_watertight": mesh.is_watertight
            }
            
        except ImportError:
            # Fallback for OBJ files
            vertices = 0
            faces = 0
            
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        vertices += 1
                    elif line.startswith('f '):
                        faces += 1
            
            return {
                "path": path,
                "vertices": vertices,
                "faces": faces
            }


# =============================================================================
# Example Usage
# =============================================================================

# Import os at module level for CloudThreeDGenerator
import os


def example_basic_generation():
    """Basic 3D model generation."""
    print("\n" + "="*60)
    print("Example 1: Basic 3D Generation")
    print("="*60)
    
    config = ThreeDConfig(
        output_dir="outputs/3d",
        resolution=64
    )
    
    generator = LocalThreeDGenerator()
    generator.load()
    
    # Generate a cube
    result = generator.generate("a simple cube", config)
    if result:
        print(f"Generated: {result}")
        
        # Get model info
        info = ThreeDUtils.get_model_info(result)
        print(f"Model info: {info}")


def example_various_shapes():
    """Generate various shapes."""
    print("\n" + "="*60)
    print("Example 2: Various Shapes")
    print("="*60)
    
    config = ThreeDConfig(output_dir="outputs/3d")
    generator = LocalThreeDGenerator()
    generator.load()
    
    prompts = [
        "a red cube",
        "a blue sphere ball",
        "a stone pyramid",
        "a wooden box"
    ]
    
    for prompt in prompts:
        result = generator.generate(prompt, config)
        print(f"  '{prompt}' -> {result}")


def example_mesh_operations():
    """Working with meshes directly."""
    print("\n" + "="*60)
    print("Example 3: Mesh Operations")
    print("="*60)
    
    # Create custom mesh
    mesh = Mesh3D()
    mesh.name = "custom_shape"
    
    # Add vertices for a simple house shape
    # Base
    mesh.add_vertex(-1, -1, 0)  # 0
    mesh.add_vertex(1, -1, 0)   # 1
    mesh.add_vertex(1, 1, 0)    # 2
    mesh.add_vertex(-1, 1, 0)   # 3
    # Roof
    mesh.add_vertex(-1, -1, 1)  # 4
    mesh.add_vertex(1, -1, 1)   # 5
    mesh.add_vertex(1, 1, 1)    # 6
    mesh.add_vertex(-1, 1, 1)   # 7
    mesh.add_vertex(0, 0, 1.5)  # 8 (roof peak)
    
    # Add faces (walls)
    mesh.add_face(0, 1, 5)
    mesh.add_face(0, 5, 4)
    # ... (more faces would be added for complete house)
    
    # Export
    Path("outputs/3d").mkdir(parents=True, exist_ok=True)
    mesh.export_obj("outputs/3d/custom_house.obj")
    
    print(f"Created custom mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")


def example_format_conversion():
    """Convert between 3D formats."""
    print("\n" + "="*60)
    print("Example 4: Format Conversion")
    print("="*60)
    
    print("Supported formats:")
    print("  - OBJ: Universal, text-based")
    print("  - GLB: Binary glTF, good for web")
    print("  - STL: 3D printing")
    print("  - PLY: Point clouds")
    
    print("\nConversion example:")
    print("  ThreeDUtils.convert_format('model.obj', 'glb')")
    print("  ThreeDUtils.convert_format('model.obj', 'stl')")


def example_forge_integration():
    """ForgeAI integration."""
    print("\n" + "="*60)
    print("Example 5: ForgeAI Integration")
    print("="*60)
    
    print("For actual ForgeAI 3D generation:")
    print("""
    from forge_ai.gui.tabs.threed_tab import Local3DGen, Cloud3DGen
    
    # Local generation (needs GPU)
    local = Local3DGen()
    if local.load():
        result = local.generate(
            prompt="a medieval castle",
            resolution=64
        )
    
    # Cloud generation (needs API key)
    cloud = Cloud3DGen()
    result = cloud.generate(
        prompt="a spaceship",
        format="glb"
    )
    
    # GUI usage
    python run.py --gui
    # Go to 3D tab to:
    # - Generate from text prompts
    # - Convert images to 3D
    # - Preview 3D models
    # - Export to various formats
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("ForgeAI 3D Generation Examples")
    print("="*60)
    
    example_basic_generation()
    example_various_shapes()
    example_mesh_operations()
    example_format_conversion()
    example_forge_integration()
    
    print("\n" + "="*60)
    print("3D Generation Summary:")
    print("="*60)
    print("""
3D Generation Methods:

1. Text-to-3D (Shap-E):
   - Generate from text descriptions
   - "a wooden chair" -> 3D mesh
   - Needs GPU for local generation

2. Image-to-3D:
   - Convert 2D images to 3D
   - Uses depth estimation
   - Good for product visualization

3. Cloud Generation:
   - Uses Replicate or similar APIs
   - No local GPU needed
   - Pay per generation

Output Formats:
   - OBJ: Universal, good compatibility
   - GLB: Binary, good for web/games
   - STL: 3D printing
   - PLY: Point clouds

Typical Uses:
   - Game asset creation
   - VR/AR content
   - 3D printing models
   - Product visualization
   - Avatar/character generation

Requirements:
   pip install torch trimesh
   pip install shap-e  # Optional for Shap-E
   
GUI:
   python run.py --gui
   # Use 3D tab for visual interface
""")
