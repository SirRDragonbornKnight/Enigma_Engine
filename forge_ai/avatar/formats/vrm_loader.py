"""
VRM Model Loader

VRM (Virtual Reality Model) is a 3D humanoid avatar format
commonly used for VTubers. Based on glTF 2.0.

VRM models include:
- Humanoid bone structure
- Blend shapes for expressions
- Material/texture information
- First person settings
- Spring bones for hair/cloth physics

Requirements:
- pygltflib: Load glTF/VRM base format
- trimesh (optional): Mesh processing
- numpy: Math operations

Install: pip install pygltflib trimesh numpy
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Check for required libraries
try:
    import pygltflib  # type: ignore[import-not-found]
    PYGLTFLIB_AVAILABLE = True
except ImportError:
    PYGLTFLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

VRM_AVAILABLE = PYGLTFLIB_AVAILABLE and NUMPY_AVAILABLE


@dataclass
class VRMBlendShape:
    """A VRM blend shape (expression)."""
    name: str
    preset: str = ""  # VRM preset name (joy, angry, sorrow, fun, etc.)
    binds: List[Dict] = field(default_factory=list)  # Mesh + index + weight
    material_values: List[Dict] = field(default_factory=list)
    is_binary: bool = False


@dataclass  
class VRMHumanBone:
    """A VRM humanoid bone."""
    bone_name: str  # VRM bone name (hips, spine, head, etc.)
    node_index: int  # glTF node index
    use_default_values: bool = True


@dataclass
class VRMModel:
    """
    A loaded VRM model with all its data.
    
    Contains:
    - Mesh data (vertices, faces, UVs)
    - Skeleton/bone data
    - Blend shapes for expressions
    - Material/texture references
    - VRM metadata (title, author, etc.)
    """
    
    # Metadata
    title: str = ""
    version: str = ""
    author: str = ""
    contact_info: str = ""
    reference: str = ""
    allowed_user: str = "OnlyAuthor"  # OnlyAuthor, ExplicitlyLicensedPerson, Everyone
    
    # Mesh data
    mesh_count: int = 0
    vertex_count: int = 0
    
    # Skeleton
    humanoid_bones: Dict[str, VRMHumanBone] = field(default_factory=dict)
    
    # Expressions
    blend_shapes: Dict[str, VRMBlendShape] = field(default_factory=dict)
    
    # Internal data
    gltf_data: Any = None
    file_path: str = ""
    
    @property
    def has_expressions(self) -> bool:
        """Check if model has blend shapes for expressions."""
        return len(self.blend_shapes) > 0
    
    def get_expression_names(self) -> List[str]:
        """Get list of available expression names."""
        return list(self.blend_shapes.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "version": self.version,
            "author": self.author,
            "mesh_count": self.mesh_count,
            "vertex_count": self.vertex_count,
            "humanoid_bones": list(self.humanoid_bones.keys()),
            "blend_shapes": list(self.blend_shapes.keys()),
            "file_path": self.file_path,
        }


class VRMLoader:
    """
    Loader for VRM format avatar models.
    
    Usage:
        loader = VRMLoader()
        if loader.is_available():
            model = loader.load("avatar.vrm")
            if model:
                print(f"Loaded: {model.title}")
                print(f"Expressions: {model.get_expression_names()}")
    """
    
    # VRM preset expression names
    VRM_PRESETS = [
        "neutral", "joy", "angry", "sorrow", "fun",
        "a", "i", "u", "e", "o",  # Vowel shapes for lip sync
        "blink", "blink_l", "blink_r",
        "lookup", "lookdown", "lookleft", "lookright",
    ]
    
    def __init__(self):
        """Initialize VRM loader."""
        self._cache: Dict[str, VRMModel] = {}
    
    @staticmethod
    def is_available() -> bool:
        """Check if required libraries are installed."""
        return VRM_AVAILABLE
    
    @staticmethod
    def get_requirements() -> List[str]:
        """Get list of required packages."""
        return ["pygltflib", "numpy", "trimesh (optional)"]
    
    def load(self, filepath: str) -> Optional[VRMModel]:
        """
        Load a VRM model file.
        
        Args:
            filepath: Path to .vrm file
            
        Returns:
            VRMModel or None if loading failed
        """
        if not VRM_AVAILABLE:
            print("[VRMLoader] Required libraries not installed.")
            print(f"  Install with: pip install {' '.join(self.get_requirements())}")
            return None
        
        path = Path(filepath)
        if not path.exists():
            print(f"[VRMLoader] File not found: {filepath}")
            return None
        
        if path.suffix.lower() not in ['.vrm', '.glb', '.gltf']:
            print(f"[VRMLoader] Unsupported format: {path.suffix}")
            return None
        
        # Check cache
        cache_key = str(path.absolute())
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            return self._load_vrm(path)
        except Exception as e:
            print(f"[VRMLoader] Error loading {filepath}: {e}")
            return None
    
    def _load_vrm(self, path: Path) -> Optional[VRMModel]:
        """Internal VRM loading."""
        import pygltflib  # type: ignore[import-not-found]
        
        # Load glTF base
        gltf = pygltflib.GLTF2().load(str(path))
        
        model = VRMModel()
        model.file_path = str(path)
        model.gltf_data = gltf
        
        # Count meshes and vertices
        model.mesh_count = len(gltf.meshes) if gltf.meshes else 0
        
        # Parse VRM extension data
        if gltf.extensions and 'VRM' in gltf.extensions:
            vrm_ext = gltf.extensions['VRM']
            self._parse_vrm_extension(model, vrm_ext)
        elif gltf.extensions and 'VRMC_vrm' in gltf.extensions:
            # VRM 1.0 format
            vrm_ext = gltf.extensions['VRMC_vrm']
            self._parse_vrm1_extension(model, vrm_ext)
        else:
            # No VRM extension, just basic glTF
            print(f"[VRMLoader] No VRM extension found in {path.name}")
        
        # Cache and return
        self._cache[str(path.absolute())] = model
        return model
    
    def _parse_vrm_extension(self, model: VRMModel, vrm_ext: Dict):
        """Parse VRM 0.x extension data."""
        # Metadata
        if 'meta' in vrm_ext:
            meta = vrm_ext['meta']
            model.title = meta.get('title', '')
            model.version = meta.get('version', '')
            model.author = meta.get('author', '')
            model.contact_info = meta.get('contactInformation', '')
            model.reference = meta.get('reference', '')
            model.allowed_user = meta.get('allowedUserName', 'OnlyAuthor')
        
        # Humanoid bones
        if 'humanoid' in vrm_ext:
            humanoid = vrm_ext['humanoid']
            for bone in humanoid.get('humanBones', []):
                bone_name = bone.get('bone', '')
                node_index = bone.get('node', -1)
                if bone_name and node_index >= 0:
                    model.humanoid_bones[bone_name] = VRMHumanBone(
                        bone_name=bone_name,
                        node_index=node_index
                    )
        
        # Blend shapes
        if 'blendShapeMaster' in vrm_ext:
            bsm = vrm_ext['blendShapeMaster']
            for group in bsm.get('blendShapeGroups', []):
                name = group.get('name', '')
                preset = group.get('presetName', '')
                
                shape = VRMBlendShape(
                    name=name,
                    preset=preset,
                    binds=group.get('binds', []),
                    material_values=group.get('materialValues', []),
                    is_binary=group.get('isBinary', False)
                )
                
                # Use preset name if available, otherwise use name
                key = preset if preset else name
                model.blend_shapes[key.lower()] = shape
    
    def _parse_vrm1_extension(self, model: VRMModel, vrm_ext: Dict):
        """Parse VRM 1.0 extension data."""
        # VRM 1.0 has different structure
        if 'meta' in vrm_ext:
            meta = vrm_ext['meta']
            model.title = meta.get('name', '')
            model.version = meta.get('version', '')
            model.author = ', '.join(meta.get('authors', []))
            model.contact_info = meta.get('contactInformation', '')
            model.reference = ', '.join(meta.get('references', []))
            model.allowed_user = meta.get('allowedUserName', 'OnlyAuthor')
        
        # Expressions in VRM 1.0
        if 'expressions' in vrm_ext:
            for preset_name, expr_data in vrm_ext['expressions'].get('preset', {}).items():
                shape = VRMBlendShape(
                    name=preset_name,
                    preset=preset_name,
                    is_binary=expr_data.get('isBinary', False)
                )
                model.blend_shapes[preset_name.lower()] = shape
    
    def clear_cache(self):
        """Clear the model cache."""
        self._cache.clear()
    
    def get_cached_models(self) -> List[str]:
        """Get list of cached model paths."""
        return list(self._cache.keys())


def get_vrm_loader() -> VRMLoader:
    """Get or create VRM loader instance."""
    return VRMLoader()
