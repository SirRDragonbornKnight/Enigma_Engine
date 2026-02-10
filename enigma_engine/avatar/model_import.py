"""
3D Model Import

Supports GLB, GLTF, and FBX model import with auto-rigging.
Handles various 3D formats for avatar use.

FILE: enigma_engine/avatar/model_import.py
TYPE: Avatar System
MAIN CLASSES: ModelImporter, GLBLoader, FBXLoader, AutoRigger
"""

import json
import logging
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported 3D model formats."""
    GLB = "glb"
    GLTF = "gltf"
    FBX = "fbx"
    OBJ = "obj"
    VRM = "vrm"


@dataclass
class Vector3:
    """3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    """Quaternion for rotation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Transform:
    """3D transformation."""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))


@dataclass
class Bone:
    """A bone in the skeleton."""
    name: str
    index: int
    parent_index: int = -1
    transform: Transform = field(default_factory=Transform)
    children: list[int] = field(default_factory=list)


@dataclass
class Skeleton:
    """Model skeleton/armature."""
    bones: list[Bone] = field(default_factory=list)
    root_bone_index: int = 0
    
    def get_bone(self, name: str) -> Optional[Bone]:
        """Get bone by name."""
        for bone in self.bones:
            if bone.name == name:
                return bone
        return None
    
    def get_bone_hierarchy(self) -> dict[str, list[str]]:
        """Get bone parent-child relationships."""
        hierarchy = {}
        for bone in self.bones:
            parent_name = ""
            if bone.parent_index >= 0:
                parent_name = self.bones[bone.parent_index].name
            if parent_name not in hierarchy:
                hierarchy[parent_name] = []
            hierarchy[parent_name].append(bone.name)
        return hierarchy


@dataclass
class MeshData:
    """Mesh geometry data."""
    name: str = ""
    vertex_count: int = 0
    index_count: int = 0
    positions: list[float] = field(default_factory=list)
    normals: list[float] = field(default_factory=list)
    uvs: list[float] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    bone_weights: list[float] = field(default_factory=list)
    bone_indices: list[int] = field(default_factory=list)


@dataclass
class Material:
    """Model material."""
    name: str = ""
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 1.0
    base_color_texture: Optional[str] = None
    normal_texture: Optional[str] = None


@dataclass
class Model3D:
    """Complete 3D model data."""
    name: str = ""
    format: ModelFormat = ModelFormat.GLB
    meshes: list[MeshData] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    skeleton: Optional[Skeleton] = None
    animations: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_skeleton(self) -> bool:
        return self.skeleton is not None and len(self.skeleton.bones) > 0
    
    @property
    def total_vertices(self) -> int:
        return sum(m.vertex_count for m in self.meshes)
    
    @property
    def total_triangles(self) -> int:
        return sum(m.index_count // 3 for m in self.meshes)


class GLBLoader:
    """Loads GLB/glTF files."""
    
    def __init__(self):
        """Initialize GLB loader."""
    
    def load(self, path: Path) -> Model3D:
        """
        Load a GLB or glTF file.
        
        Args:
            path: Path to model file
            
        Returns:
            Loaded model
        """
        path = Path(path)
        
        if path.suffix.lower() == '.glb':
            return self._load_glb(path)
        elif path.suffix.lower() == '.gltf':
            return self._load_gltf(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    
    def _load_glb(self, path: Path) -> Model3D:
        """Load binary GLB file."""
        with open(path, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != b'glTF':
                raise ValueError("Invalid GLB magic")
            
            version = struct.unpack('<I', f.read(4))[0]
            length = struct.unpack('<I', f.read(4))[0]
            
            # Read JSON chunk
            chunk_length = struct.unpack('<I', f.read(4))[0]
            chunk_type = f.read(4)
            
            if chunk_type != b'JSON':
                raise ValueError("Expected JSON chunk")
            
            json_data = json.loads(f.read(chunk_length))
            
            # Read binary chunk if present
            binary_data = b''
            if f.tell() < length:
                bin_chunk_length = struct.unpack('<I', f.read(4))[0]
                bin_chunk_type = f.read(4)
                if bin_chunk_type == b'BIN\x00':
                    binary_data = f.read(bin_chunk_length)
        
        return self._parse_gltf(json_data, binary_data, path)
    
    def _load_gltf(self, path: Path) -> Model3D:
        """Load JSON glTF file."""
        with open(path) as f:
            json_data = json.load(f)
        
        # Load external binary buffer if referenced
        binary_data = b''
        buffers = json_data.get('buffers', [])
        if buffers and 'uri' in buffers[0]:
            uri = buffers[0]['uri']
            if not uri.startswith('data:'):
                bin_path = path.parent / uri
                if bin_path.exists():
                    binary_data = bin_path.read_bytes()
        
        return self._parse_gltf(json_data, binary_data, path)
    
    def _parse_gltf(self, json_data: dict, binary_data: bytes, path: Path) -> Model3D:
        """Parse glTF JSON data."""
        model = Model3D(
            name=path.stem,
            format=ModelFormat.GLB if path.suffix.lower() == '.glb' else ModelFormat.GLTF,
            metadata={"gltf": json_data}
        )
        
        # Parse meshes
        for mesh_data in json_data.get('meshes', []):
            mesh = MeshData(name=mesh_data.get('name', ''))
            
            for primitive in mesh_data.get('primitives', []):
                # We would parse accessors here to get actual vertex data
                # This is a simplified version
                indices_idx = primitive.get('indices')
                if indices_idx is not None:
                    mesh.index_count += self._get_accessor_count(json_data, indices_idx)
                
                attributes = primitive.get('attributes', {})
                pos_idx = attributes.get('POSITION')
                if pos_idx is not None:
                    mesh.vertex_count = self._get_accessor_count(json_data, pos_idx)
            
            model.meshes.append(mesh)
        
        # Parse materials
        for mat_data in json_data.get('materials', []):
            material = Material(name=mat_data.get('name', ''))
            
            pbr = mat_data.get('pbrMetallicRoughness', {})
            base_color = pbr.get('baseColorFactor', [1, 1, 1, 1])
            material.base_color = tuple(base_color)
            material.metallic = pbr.get('metallicFactor', 0.0)
            material.roughness = pbr.get('roughnessFactor', 1.0)
            
            model.materials.append(material)
        
        # Parse skeleton
        skins = json_data.get('skins', [])
        if skins:
            model.skeleton = self._parse_skeleton(json_data, skins[0])
        
        return model
    
    def _get_accessor_count(self, json_data: dict, accessor_index: int) -> int:
        """Get element count from accessor."""
        accessors = json_data.get('accessors', [])
        if 0 <= accessor_index < len(accessors):
            return accessors[accessor_index].get('count', 0)
        return 0
    
    def _parse_skeleton(self, json_data: dict, skin_data: dict) -> Skeleton:
        """Parse skeleton from glTF skin."""
        skeleton = Skeleton()
        
        joints = skin_data.get('joints', [])
        nodes = json_data.get('nodes', [])
        
        for i, joint_idx in enumerate(joints):
            if joint_idx < len(nodes):
                node = nodes[joint_idx]
                
                bone = Bone(
                    name=node.get('name', f'bone_{i}'),
                    index=i
                )
                
                # Parse transform
                if 'translation' in node:
                    t = node['translation']
                    bone.transform.position = Vector3(t[0], t[1], t[2])
                
                if 'rotation' in node:
                    r = node['rotation']
                    bone.transform.rotation = Quaternion(r[0], r[1], r[2], r[3])
                
                if 'scale' in node:
                    s = node['scale']
                    bone.transform.scale = Vector3(s[0], s[1], s[2])
                
                skeleton.bones.append(bone)
        
        # Establish parent relationships
        for i, joint_idx in enumerate(joints):
            node = nodes[joint_idx]
            children = node.get('children', [])
            for child_idx in children:
                if child_idx in joints:
                    child_bone_idx = joints.index(child_idx)
                    skeleton.bones[child_bone_idx].parent_index = i
                    skeleton.bones[i].children.append(child_bone_idx)
        
        return skeleton


class FBXLoader:
    """Loads FBX files (simplified - full support requires FBX SDK)."""
    
    def __init__(self):
        """Initialize FBX loader."""
    
    def load(self, path: Path) -> Model3D:
        """
        Load an FBX file.
        
        Note: Full FBX support requires the Autodesk FBX SDK.
        This provides basic structure detection.
        """
        path = Path(path)
        
        model = Model3D(
            name=path.stem,
            format=ModelFormat.FBX
        )
        
        # Check if it's binary or ASCII FBX
        with open(path, 'rb') as f:
            header = f.read(20)
        
        if header.startswith(b'Kaydara FBX Binary'):
            logger.info("Binary FBX detected")
            model.metadata['fbx_type'] = 'binary'
        else:
            logger.info("ASCII FBX detected")
            model.metadata['fbx_type'] = 'ascii'
        
        # Full FBX parsing requires FBX SDK or libraries like pyfbx
        logger.warning("Full FBX parsing requires additional libraries")
        
        return model


class AutoRigger:
    """Auto-rigging for unrigged models."""
    
    # Standard humanoid bone names to search for
    BONE_PATTERNS = {
        'hips': ['hips', 'pelvis', 'root', 'hip'],
        'spine': ['spine', 'spine1', 'back'],
        'chest': ['chest', 'spine2', 'spine3', 'upper_back'],
        'neck': ['neck'],
        'head': ['head'],
        'left_shoulder': ['l_shoulder', 'left_shoulder', 'shoulder.l', 'lshoulder'],
        'left_upper_arm': ['l_upperarm', 'left_upperarm', 'upperarm.l', 'lupperarm', 'l_arm'],
        'left_lower_arm': ['l_lowerarm', 'left_lowerarm', 'lowerarm.l', 'llowerarm', 'l_forearm'],
        'left_hand': ['l_hand', 'left_hand', 'hand.l', 'lhand'],
        'right_shoulder': ['r_shoulder', 'right_shoulder', 'shoulder.r', 'rshoulder'],
        'right_upper_arm': ['r_upperarm', 'right_upperarm', 'upperarm.r', 'rupperarm', 'r_arm'],
        'right_lower_arm': ['r_lowerarm', 'right_lowerarm', 'lowerarm.r', 'rlowerarm', 'r_forearm'],
        'right_hand': ['r_hand', 'right_hand', 'hand.r', 'rhand'],
        'left_upper_leg': ['l_upperleg', 'left_upperleg', 'upperleg.l', 'lupperleg', 'l_thigh'],
        'left_lower_leg': ['l_lowerleg', 'left_lowerleg', 'lowerleg.l', 'llowerleg', 'l_shin'],
        'left_foot': ['l_foot', 'left_foot', 'foot.l', 'lfoot'],
        'right_upper_leg': ['r_upperleg', 'right_upperleg', 'upperleg.r', 'rupperleg', 'r_thigh'],
        'right_lower_leg': ['r_lowerleg', 'right_lowerleg', 'lowerleg.r', 'rlowerleg', 'r_shin'],
        'right_foot': ['r_foot', 'right_foot', 'foot.r', 'rfoot'],
    }
    
    def __init__(self):
        """Initialize auto-rigger."""
    
    def detect_bones(self, skeleton: Skeleton) -> dict[str, str]:
        """
        Detect standard humanoid bones from skeleton.
        
        Args:
            skeleton: Model skeleton
            
        Returns:
            Dict mapping standard name to actual bone name
        """
        mapping = {}
        
        for standard_name, patterns in self.BONE_PATTERNS.items():
            for bone in skeleton.bones:
                bone_lower = bone.name.lower()
                for pattern in patterns:
                    if pattern in bone_lower:
                        mapping[standard_name] = bone.name
                        break
                if standard_name in mapping:
                    break
        
        return mapping
    
    def validate_humanoid(self, mapping: dict[str, str]) -> tuple[bool, list[str]]:
        """
        Check if mapping has required humanoid bones.
        
        Returns:
            Tuple of (is_valid, missing_bones)
        """
        required = ['hips', 'spine', 'head', 
                   'left_upper_arm', 'left_lower_arm', 'left_hand',
                   'right_upper_arm', 'right_lower_arm', 'right_hand',
                   'left_upper_leg', 'left_lower_leg', 'left_foot',
                   'right_upper_leg', 'right_lower_leg', 'right_foot']
        
        missing = [b for b in required if b not in mapping]
        return len(missing) == 0, missing


class ModelImporter:
    """Main model importer coordinating format-specific loaders."""
    
    def __init__(self):
        """Initialize model importer."""
        self._glb_loader = GLBLoader()
        self._fbx_loader = FBXLoader()
        self._auto_rigger = AutoRigger()
    
    def load(self, path: Path, auto_rig: bool = True) -> Model3D:
        """
        Load a 3D model.
        
        Args:
            path: Path to model file
            auto_rig: Attempt to detect humanoid bones
            
        Returns:
            Loaded and optionally rigged model
        """
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix in ('.glb', '.gltf'):
            model = self._glb_loader.load(path)
        elif suffix == '.fbx':
            model = self._fbx_loader.load(path)
        elif suffix == '.vrm':
            model = self._glb_loader.load(path)
            model.format = ModelFormat.VRM
        else:
            raise ValueError(f"Unsupported format: {suffix}")
        
        # Attempt auto-rigging
        if auto_rig and model.skeleton:
            mapping = self._auto_rigger.detect_bones(model.skeleton)
            is_valid, missing = self._auto_rigger.validate_humanoid(mapping)
            
            model.metadata['bone_mapping'] = mapping
            model.metadata['is_humanoid'] = is_valid
            model.metadata['missing_bones'] = missing
            
            if is_valid:
                logger.info(f"Detected valid humanoid skeleton")
            else:
                logger.warning(f"Incomplete humanoid: missing {missing}")
        
        return model
    
    def supported_formats(self) -> list[str]:
        """Get list of supported file extensions."""
        return ['.glb', '.gltf', '.fbx', '.vrm']


def load_model(path: Path) -> Model3D:
    """
    Load a 3D model file.
    
    Args:
        path: Path to model
        
    Returns:
        Loaded model
    """
    importer = ModelImporter()
    return importer.load(path)


__all__ = [
    'ModelImporter',
    'Model3D',
    'MeshData',
    'Material',
    'Skeleton',
    'Bone',
    'Transform',
    'Vector3',
    'Quaternion',
    'ModelFormat',
    'GLBLoader',
    'FBXLoader',
    'AutoRigger',
    'load_model'
]
