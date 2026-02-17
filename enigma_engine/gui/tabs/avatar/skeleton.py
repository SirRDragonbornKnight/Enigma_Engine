"""
GLTF Skeletal Animation Support

Loads skeleton (skin/joints) from GLTF models and provides bone manipulation.
"""

import json
import struct
import math
from pathlib import Path
from typing import Optional
import numpy as np


class Skeleton:
    """Manages skeleton hierarchy and bone transforms for GLTF models."""
    
    def __init__(self):
        self.joints: list[dict] = []  # [{name, parent_idx, bind_matrix, local_transform}]
        self.joint_names: list[str] = []
        self.joint_map: dict[str, int] = {}  # name -> index
        
        # Vertex skinning data
        self.vertex_joints: Optional[np.ndarray] = None  # (N, 4) joint indices per vertex
        self.vertex_weights: Optional[np.ndarray] = None  # (N, 4) weights per vertex
        self.inverse_bind_matrices: Optional[np.ndarray] = None  # (J, 4, 4) bind pose inverses
        
        # Current bone transforms (modified by move_bone)
        self.bone_transforms: dict[str, dict] = {}  # name -> {pitch, yaw, roll}
        
        # Original vertices (before skinning)
        self.bind_vertices: Optional[np.ndarray] = None
        self.bind_normals: Optional[np.ndarray] = None
    
    @classmethod
    def from_gltf(cls, gltf_path: str) -> Optional['Skeleton']:
        """Load skeleton from a GLTF/GLB file."""
        path = Path(gltf_path)
        
        if path.suffix.lower() == '.glb':
            return cls._from_glb(path)
        elif path.suffix.lower() == '.gltf':
            return cls._from_gltf(path)
        return None
    
    @classmethod
    def _from_gltf(cls, path: Path) -> Optional['Skeleton']:
        """Parse GLTF JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            
            # Load binary buffer if referenced
            buffers = {}
            for i, buf_info in enumerate(data.get('buffers', [])):
                uri = buf_info.get('uri', '')
                if uri and not uri.startswith('data:'):
                    buf_path = path.parent / uri
                    if buf_path.exists():
                        with open(buf_path, 'rb') as bf:
                            buffers[i] = bf.read()
            
            return cls._parse_gltf_data(data, buffers, path.parent)
        except Exception as e:
            print(f"[Skeleton] Failed to load GLTF: {e}")
            return None
    
    @classmethod
    def _from_glb(cls, path: Path) -> Optional['Skeleton']:
        """Parse GLB binary file."""
        try:
            with open(path, 'rb') as f:
                magic = f.read(4)
                if magic != b'glTF':
                    return None
                
                version = struct.unpack('<I', f.read(4))[0]
                length = struct.unpack('<I', f.read(4))[0]
                
                # Read JSON chunk
                chunk_length = struct.unpack('<I', f.read(4))[0]
                chunk_type = f.read(4)
                json_data = json.loads(f.read(chunk_length))
                
                # Read binary chunk if present
                buffers = {}
                if f.tell() < length:
                    chunk_length = struct.unpack('<I', f.read(4))[0]
                    chunk_type = f.read(4)
                    if chunk_type == b'BIN\x00':
                        buffers[0] = f.read(chunk_length)
                
                return cls._parse_gltf_data(json_data, buffers, path.parent)
        except Exception as e:
            print(f"[Skeleton] Failed to load GLB: {e}")
            return None
    
    @classmethod
    def _parse_gltf_data(cls, data: dict, buffers: dict, base_path: Path) -> Optional['Skeleton']:
        """Parse GLTF data structure to extract skeleton."""
        skins = data.get('skins', [])
        if not skins:
            print("[Skeleton] No skins found in model")
            return None
        
        skeleton = cls()
        skin = skins[0]  # Use first skin
        
        nodes = data.get('nodes', [])
        accessors = data.get('accessors', [])
        buffer_views = data.get('bufferViews', [])
        
        # Get joint indices from skin
        joint_indices = skin.get('joints', [])
        if not joint_indices:
            return None
        
        # Build joint list with names
        for j_idx in joint_indices:
            if j_idx < len(nodes):
                node = nodes[j_idx]
                name = node.get('name', f'joint_{j_idx}')
                skeleton.joint_names.append(name)
                skeleton.joint_map[name] = len(skeleton.joints)
                
                # Extract local transform
                translation = node.get('translation', [0, 0, 0])
                rotation = node.get('rotation', [0, 0, 0, 1])  # Quaternion (x,y,z,w)
                scale = node.get('scale', [1, 1, 1])
                
                # Find parent index
                parent_idx = -1
                for pi, pnode in enumerate(nodes):
                    if 'children' in pnode and j_idx in pnode['children']:
                        # Is this parent also a joint?
                        try:
                            parent_idx = joint_indices.index(pi)
                        except ValueError:
                            parent_idx = -1
                        break
                
                skeleton.joints.append({
                    'name': name,
                    'parent_idx': parent_idx,
                    'translation': translation,
                    'rotation': rotation,
                    'scale': scale,
                    'node_idx': j_idx
                })
        
        # Load inverse bind matrices
        ibm_accessor_idx = skin.get('inverseBindMatrices')
        if ibm_accessor_idx is not None and ibm_accessor_idx < len(accessors):
            ibm_data = cls._read_accessor(accessors[ibm_accessor_idx], buffer_views, buffers)
            if ibm_data is not None:
                # Reshape to (num_joints, 4, 4)
                # GLTF stores matrices column-major, numpy is row-major - transpose each
                ibm_raw = ibm_data.reshape(-1, 4, 4)
                skeleton.inverse_bind_matrices = np.array([m.T for m in ibm_raw])
        
        # Initialize bone transforms to identity
        for name in skeleton.joint_names:
            skeleton.bone_transforms[name] = {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        print(f"[Skeleton] Loaded {len(skeleton.joints)} joints")
        return skeleton
    
    @staticmethod
    def _read_accessor(accessor: dict, buffer_views: list, buffers: dict) -> Optional[np.ndarray]:
        """Read data from a GLTF accessor."""
        view_idx = accessor.get('bufferView')
        if view_idx is None:
            return None
        
        view = buffer_views[view_idx]
        buf_idx = view.get('buffer', 0)
        
        if buf_idx not in buffers:
            return None
        
        buffer = buffers[buf_idx]
        byte_offset = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
        count = accessor['count']
        
        # Component type
        comp_type = accessor['componentType']
        comp_map = {
            5120: np.int8, 5121: np.uint8,
            5122: np.int16, 5123: np.uint16,
            5125: np.uint32, 5126: np.float32
        }
        dtype = comp_map.get(comp_type, np.float32)
        
        # Element size
        type_sizes = {
            'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
            'MAT2': 4, 'MAT3': 9, 'MAT4': 16
        }
        elem_size = type_sizes.get(accessor['type'], 1)
        
        total_elements = count * elem_size
        byte_length = total_elements * np.dtype(dtype).itemsize
        
        arr = np.frombuffer(buffer, dtype=dtype, count=total_elements, offset=byte_offset)
        return arr.copy()
    
    def load_vertex_weights(self, gltf_path: str, mesh_idx: int = None) -> bool:
        """Load vertex joint weights from mesh primitives.
        
        Args:
            gltf_path: Path to GLTF/GLB file
            mesh_idx: Specific mesh index, or None to search all meshes
        """
        path = Path(gltf_path)
        
        try:
            if path.suffix.lower() == '.glb':
                with open(path, 'rb') as f:
                    f.seek(12)  # Skip header
                    chunk_length = struct.unpack('<I', f.read(4))[0]
                    f.read(4)  # chunk type
                    data = json.loads(f.read(chunk_length))
                    
                    f.read(8)  # Next chunk header
                    buffers = {0: f.read()}
            else:
                with open(path) as f:
                    data = json.load(f)
                buffers = {}
                for i, buf_info in enumerate(data.get('buffers', [])):
                    uri = buf_info.get('uri', '')
                    if uri and not uri.startswith('data:'):
                        buf_path = path.parent / uri
                        if buf_path.exists():
                            with open(buf_path, 'rb') as bf:
                                buffers[i] = bf.read()
            
            meshes = data.get('meshes', [])
            accessors = data.get('accessors', [])
            buffer_views = data.get('bufferViews', [])
            
            # Determine which meshes to check
            if mesh_idx is not None:
                if mesh_idx >= len(meshes):
                    return False
                mesh_indices = [mesh_idx]
            else:
                # Search all meshes for weights
                mesh_indices = range(len(meshes))

            # Accumulate weights from ALL primitives
            all_joints = []
            all_weights = []
            
            for mi in mesh_indices:
                mesh = meshes[mi]
                for prim in mesh.get('primitives', []):
                    attrs = prim.get('attributes', {})
                    
                    joints_idx = attrs.get('JOINTS_0')
                    weights_idx = attrs.get('WEIGHTS_0')
                    
                    if joints_idx is not None and weights_idx is not None:
                        joints_data = self._read_accessor(accessors[joints_idx], buffer_views, buffers)
                        weights_data = self._read_accessor(accessors[weights_idx], buffer_views, buffers)
                        
                        if joints_data is not None and weights_data is not None:
                            joints_arr = joints_data.reshape(-1, 4).astype(np.int32)
                            weights_arr = weights_data.reshape(-1, 4).astype(np.float32)
                            all_joints.append(joints_arr)
                            all_weights.append(weights_arr)
                            mesh_name = mesh.get('name', f'mesh_{mi}')
                            print(f"[Skeleton] Found weights for {len(joints_arr)} vertices in {mesh_name}")
            
            if all_joints:
                # Concatenate all primitives
                self.vertex_joints = np.vstack(all_joints)
                self.vertex_weights = np.vstack(all_weights)
                print(f"[Skeleton] Total: Loaded weights for {len(self.vertex_joints)} vertices from {len(all_joints)} primitives")
                return True
            
            return False
        except Exception as e:
            print(f"[Skeleton] Failed to load vertex weights: {e}")
            return False
    
    def set_bind_pose(self, vertices: np.ndarray, normals: Optional[np.ndarray] = None):
        """Store the original bind pose vertices."""
        self.bind_vertices = vertices.copy()
        self.bind_normals = normals.copy() if normals is not None else None
    
    def move_bone(self, bone_name: str, pitch: float = 0, yaw: float = 0, roll: float = 0):
        """Set rotation for a bone (in degrees).
        
        Supports training data format names like:
        head, spine, neck, left_upper_arm, right_forearm, etc.
        """
        matched_name = self._find_bone(bone_name)
        
        if matched_name:
            self.bone_transforms[matched_name] = {
                'pitch': pitch,
                'yaw': yaw, 
                'roll': roll
            }
            print(f"[Skeleton] Matched '{bone_name}' -> '{matched_name}' (pitch={pitch}, yaw={yaw}, roll={roll})")
            return True
        
        print(f"[Skeleton] No match for '{bone_name}'")
        return False
    
    def _find_bone(self, bone_name: str) -> Optional[str]:
        """Find actual bone name from simple/training name.
        
        Maps names like 'left_upper_arm' to 'mixamorig:LeftArm'
        Tries exact match first, then aliases, then partial match.
        """
        # Exact match first - AI can use actual bone names directly
        if bone_name in self.joint_names:
            return bone_name
        
        bone_lower = bone_name.lower().replace('_', '').replace(' ', '')
        
        # Mapping from training data names to skeleton patterns
        bone_aliases = {
            # Head/Neck
            'head': ['head'],
            'neck': ['neck'],
            # Spine
            'spine': ['spine', 'spine1', 'spine2'],
            # Arms - upper
            'leftupperarm': ['leftarm', 'leftshoulder', 'l_arm', 'larm', 'left_arm'],
            'rightupperarm': ['rightarm', 'rightshoulder', 'r_arm', 'rarm', 'right_arm'],
            'leftshoulder': ['leftshoulder', 'l_shoulder'],
            'rightshoulder': ['rightshoulder', 'r_shoulder'],
            # Arms - forearm
            'leftforearm': ['leftforearm', 'l_forearm', 'lforearm', 'left_elbow'],
            'rightforearm': ['rightforearm', 'r_forearm', 'rforearm', 'right_elbow'],
            # Hands
            'lefthand': ['lefthand', 'l_hand', 'lhand', 'left_wrist'],
            'righthand': ['righthand', 'r_hand', 'rhand', 'right_wrist'],
            # Legs - upper  
            'leftupperleg': ['leftupleg', 'lefthip', 'l_thigh', 'leftthigh'],
            'rightupperleg': ['rightupleg', 'righthip', 'r_thigh', 'rightthigh'],
            # Legs - lower
            'leftlowerleg': ['leftleg', 'l_calf', 'leftcalf', 'left_knee'],
            'rightlowerleg': ['rightleg', 'r_calf', 'rightcalf', 'right_knee'],
            # Feet
            'leftfoot': ['leftfoot', 'l_foot', 'lfoot', 'left_ankle'],
            'rightfoot': ['rightfoot', 'r_foot', 'rfoot', 'right_ankle'],
        }
        
        # Check if input matches an alias key
        patterns_to_check = bone_aliases.get(bone_lower, [bone_lower])
        
        # Search all skeleton bones
        for joint_name in self.joint_names:
            joint_lower = joint_name.lower().replace('_', '').replace(':', '').replace(' ', '')
            
            for pattern in patterns_to_check:
                if pattern in joint_lower or joint_lower.endswith(pattern):
                    return joint_name
        
        # Fallback: direct partial match
        for joint_name in self.joint_names:
            if bone_lower in joint_name.lower():
                return joint_name
        
        return None
    
    def get_available_bone_mappings(self) -> dict:
        """Get mapping of simple names to actual skeleton bones."""
        simple_names = [
            'head', 'neck', 'spine',
            'left_upper_arm', 'right_upper_arm',
            'left_forearm', 'right_forearm',
            'left_hand', 'right_hand',
            'left_upper_leg', 'right_upper_leg',
            'left_lower_leg', 'right_lower_leg',
            'left_foot', 'right_foot'
        ]
        
        mappings = {}
        for name in simple_names:
            actual = self._find_bone(name)
            if actual:
                mappings[name] = actual
        return mappings
    
    def apply_skinning(self, vertices: np.ndarray) -> np.ndarray:
        """Apply bone transforms to vertices using linear blend skinning.
        
        Handles mismatch between vertex count and weight count by only
        skinning the first N vertices where N = len(vertex_weights).
        """
        if self.vertex_joints is None or self.vertex_weights is None:
            return vertices
        
        if self.inverse_bind_matrices is None:
            return vertices
        
        # Handle vertex count mismatch
        num_weighted = len(self.vertex_weights)
        num_vertices = len(vertices)
        
        if num_weighted > num_vertices:
            # More weights than vertices - truncate weights
            num_weighted = num_vertices
        
        # Start with copy of original vertices (preserve all)
        result = vertices.copy()
        
        # Only process vertices that have weights
        weighted_verts = vertices[:num_weighted].copy()
        weighted_result = np.zeros_like(weighted_verts)
        
        # Track total weight per vertex to handle partial weighting
        total_weights = np.zeros(num_weighted, dtype=np.float32)
        
        # Compute current bone matrices
        bone_matrices = self._compute_bone_matrices()
        
        # Convert to homogeneous coordinates
        v_homo = np.hstack([weighted_verts, np.ones((num_weighted, 1))])
        
        for i in range(4):  # Up to 4 bone influences per vertex
            joint_indices = self.vertex_joints[:num_weighted, i]
            weights = self.vertex_weights[:num_weighted, i:i+1]
            
            # Skip zero weights
            mask = weights.flatten() > 0.001
            if not np.any(mask):
                continue
            
            # Accumulate total weights
            total_weights += weights.flatten() * mask
            
            for j_idx in np.unique(joint_indices[mask]):
                vert_mask = (joint_indices == j_idx) & mask
                if not np.any(vert_mask):
                    continue
                
                # Get transform matrix
                bone_matrix = bone_matrices[j_idx]
                ibm = self.inverse_bind_matrices[j_idx]
                
                # Combined transform
                transform = bone_matrix @ ibm
                
                # Apply to vertices
                v_transformed = (transform @ v_homo[vert_mask].T).T[:, :3]
                weighted_result[vert_mask] += weights[vert_mask] * v_transformed
        
        # Only update vertices that have significant weight
        # Vertices with zero/negligible weights keep original position
        has_weight = total_weights > 0.01
        
        # For partially weighted vertices, blend with original
        for i in range(num_weighted):
            if has_weight[i]:
                w = total_weights[i]
                if w < 0.99:
                    # Blend: skinned * w + original * (1-w)
                    result[i] = weighted_result[i] + weighted_verts[i] * (1.0 - w)
                else:
                    result[i] = weighted_result[i]
            # else: keep original (already in result from copy)
        
        return result
    
    def _compute_bone_matrices(self) -> np.ndarray:
        """Compute world transform matrices for all bones.
        
        Strategy: Start from bind pose (inverse of IBM) and apply user rotations.
        This avoids scale/transform mismatches from GLTF export.
        """
        num_joints = len(self.joints)
        matrices = np.zeros((num_joints, 4, 4), dtype=np.float32)
        
        # Get bind pose world matrices by inverting IBM
        # At rest, bone_matrix @ IBM = I, so bone_matrix = inv(IBM)
        if self.inverse_bind_matrices is not None:
            for i in range(min(num_joints, len(self.inverse_bind_matrices))):
                try:
                    matrices[i] = np.linalg.inv(self.inverse_bind_matrices[i])
                except np.linalg.LinAlgError:
                    matrices[i] = np.eye(4, dtype=np.float32)
        else:
            for i in range(num_joints):
                matrices[i] = np.eye(4, dtype=np.float32)
        
        # Apply user rotations to bind pose
        for i, joint in enumerate(self.joints):
            name = joint['name']
            
            # Get user rotation if any
            transforms = self.bone_transforms.get(name, {})
            if transforms.get('pitch', 0) != 0 or transforms.get('yaw', 0) != 0 or transforms.get('roll', 0) != 0:
                user_rot = self._euler_to_matrix(transforms)
                # Apply rotation in bone's local space
                matrices[i] = matrices[i] @ user_rot
        
        return matrices
    
    def _joint_local_matrix(self, joint: dict) -> np.ndarray:
        """Build local transform matrix for a joint."""
        mat = np.eye(4, dtype=np.float32)
        
        # Translation
        t = joint.get('translation', [0, 0, 0])
        mat[0, 3] = t[0]
        mat[1, 3] = t[1]
        mat[2, 3] = t[2]
        
        # Rotation (quaternion xyzw)
        q = joint.get('rotation', [0, 0, 0, 1])
        rot_mat = self._quat_to_matrix(q[0], q[1], q[2], q[3])
        mat[:3, :3] = rot_mat
        
        # Scale
        s = joint.get('scale', [1, 1, 1])
        scale_mat = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float32)
        
        return mat @ scale_mat
    
    @staticmethod
    def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        mat = np.zeros((3, 3), dtype=np.float32)
        
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        mat[0, 0] = 1 - 2*(yy + zz)
        mat[0, 1] = 2*(xy - wz)
        mat[0, 2] = 2*(xz + wy)
        
        mat[1, 0] = 2*(xy + wz)
        mat[1, 1] = 1 - 2*(xx + zz)
        mat[1, 2] = 2*(yz - wx)
        
        mat[2, 0] = 2*(xz - wy)
        mat[2, 1] = 2*(yz + wx)
        mat[2, 2] = 1 - 2*(xx + yy)
        
        return mat
    
    def _euler_to_matrix(self, transform: dict) -> np.ndarray:
        """Convert pitch/yaw/roll to 4x4 rotation matrix."""
        pitch = math.radians(transform.get('pitch', 0))
        yaw = math.radians(transform.get('yaw', 0))
        roll = math.radians(transform.get('roll', 0))
        
        # Rotation matrices
        cx, sx = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cz, sz = math.cos(roll), math.sin(roll)
        
        # Combined rotation (yaw * pitch * roll order)
        mat = np.eye(4, dtype=np.float32)
        
        mat[0, 0] = cy * cz
        mat[0, 1] = sx * sy * cz - cx * sz
        mat[0, 2] = cx * sy * cz + sx * sz
        
        mat[1, 0] = cy * sz
        mat[1, 1] = sx * sy * sz + cx * cz
        mat[1, 2] = cx * sy * sz - sx * cz
        
        mat[2, 0] = -sy
        mat[2, 1] = sx * cy
        mat[2, 2] = cx * cy
        
        return mat
    
    def get_bone_names(self) -> list[str]:
        """Get list of all bone names."""
        return self.joint_names.copy()
    
    def get_bone_info(self) -> dict:
        """Get bone info for AI context injection."""
        bones = []
        for name in self.joint_names:
            transform = self.bone_transforms.get(name, {})
            bones.append({
                'name': name,
                'pitch': transform.get('pitch', 0),
                'yaw': transform.get('yaw', 0),
                'roll': transform.get('roll', 0)
            })
        
        return {
            'skeleton_loaded': True,
            'num_bones': len(self.joints),
            'available_bones': bones,
            'has_weights': self.vertex_weights is not None
        }
    
    def reset_pose(self):
        """Reset all bones to bind pose."""
        for name in self.joint_names:
            self.bone_transforms[name] = {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
