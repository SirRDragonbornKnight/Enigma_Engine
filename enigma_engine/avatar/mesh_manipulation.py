"""
Mesh Manipulation System

Provides vertex-level manipulation, morph targets, and blend shapes for
advanced avatar deformation and animation.

Features:
- Vertex-level manipulation (stretch, squash, pull)
- Morph targets / blend shapes
- Region-based scaling (scale body regions)
- Smooth deformation with weight painting
- Export to common mesh formats

FILE: enigma_engine/avatar/mesh_manipulation.py
TYPE: Avatar Morphing
MAIN CLASSES: MeshManipulator, MorphTarget, BlendShape, MeshRegion
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Vertex:
    """A 3D vertex with position, normal, and UV coordinates."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # Normal vector
    nx: float = 0.0
    ny: float = 1.0
    nz: float = 0.0
    
    # UV coordinates
    u: float = 0.0
    v: float = 0.0
    
    # Bone weights for skinning
    bone_indices: List[int] = field(default_factory=list)
    bone_weights: List[float] = field(default_factory=list)
    
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def set_position(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z
    
    def normal(self) -> Tuple[float, float, float]:
        return (self.nx, self.ny, self.nz)
    
    def distance_to(self, other: 'Vertex') -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    def lerp(self, target: 'Vertex', t: float) -> 'Vertex':
        """Linear interpolation to another vertex."""
        return Vertex(
            x=self.x + (target.x - self.x) * t,
            y=self.y + (target.y - self.y) * t,
            z=self.z + (target.z - self.z) * t,
            nx=self.nx + (target.nx - self.nx) * t,
            ny=self.ny + (target.ny - self.ny) * t,
            nz=self.nz + (target.nz - self.nz) * t,
            u=self.u,
            v=self.v,
            bone_indices=self.bone_indices.copy(),
            bone_weights=self.bone_weights.copy(),
        )
    
    def copy(self) -> 'Vertex':
        return Vertex(
            x=self.x, y=self.y, z=self.z,
            nx=self.nx, ny=self.ny, nz=self.nz,
            u=self.u, v=self.v,
            bone_indices=self.bone_indices.copy(),
            bone_weights=self.bone_weights.copy(),
        )


@dataclass
class Face:
    """A triangle face defined by vertex indices."""
    v1: int
    v2: int
    v3: int
    material_index: int = 0


class MeshRegion(Enum):
    """Predefined body regions for manipulation."""
    HEAD = "head"
    FACE = "face"
    NECK = "neck"
    TORSO = "torso"
    CHEST = "chest"
    WAIST = "waist"
    HIPS = "hips"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    LEFT_LEG = "left_leg"
    RIGHT_LEG = "right_leg"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
    HAIR = "hair"
    CUSTOM = "custom"


@dataclass
class RegionDefinition:
    """Defines a mesh region with vertex groups and bounds."""
    name: str
    region_type: MeshRegion
    vertex_indices: List[int] = field(default_factory=list)
    
    # Bounding box (for quick region detection)
    min_x: float = -float('inf')
    max_x: float = float('inf')
    min_y: float = -float('inf')
    max_y: float = float('inf')
    min_z: float = -float('inf')
    max_z: float = float('inf')
    
    # Weight painting (per-vertex influence)
    weights: Dict[int, float] = field(default_factory=dict)
    
    def contains_vertex(self, vertex: Vertex) -> bool:
        """Check if vertex is within region bounds."""
        return (
            self.min_x <= vertex.x <= self.max_x and
            self.min_y <= vertex.y <= self.max_y and
            self.min_z <= vertex.z <= self.max_z
        )
    
    def get_weight(self, vertex_index: int) -> float:
        """Get influence weight for a vertex (0-1)."""
        return self.weights.get(vertex_index, 1.0 if vertex_index in self.vertex_indices else 0.0)


@dataclass
class MorphTarget:
    """
    A morph target (shape key) storing vertex deltas.
    
    Morph targets store the displacement from base mesh to target shape,
    allowing blending between shapes.
    """
    name: str
    
    # Vertex deltas: {vertex_index: (dx, dy, dz)}
    deltas: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    
    # Current blend weight (0-1)
    weight: float = 0.0
    
    # Optional: target normals
    normal_deltas: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    
    def set_delta(self, vertex_index: int, dx: float, dy: float, dz: float):
        """Set displacement for a vertex."""
        self.deltas[vertex_index] = (dx, dy, dz)
    
    def get_delta(self, vertex_index: int) -> Tuple[float, float, float]:
        """Get displacement for a vertex (returns (0,0,0) if not set)."""
        return self.deltas.get(vertex_index, (0.0, 0.0, 0.0))
    
    def apply_to_vertex(self, vertex: Vertex, vertex_index: int) -> Vertex:
        """Apply morph to a vertex based on current weight."""
        if self.weight == 0.0 or vertex_index not in self.deltas:
            return vertex
        
        dx, dy, dz = self.deltas[vertex_index]
        result = vertex.copy()
        result.x += dx * self.weight
        result.y += dy * self.weight
        result.z += dz * self.weight
        
        if vertex_index in self.normal_deltas:
            dnx, dny, dnz = self.normal_deltas[vertex_index]
            result.nx += dnx * self.weight
            result.ny += dny * self.weight
            result.nz += dnz * self.weight
            # Renormalize
            length = math.sqrt(result.nx**2 + result.ny**2 + result.nz**2)
            if length > 0.0001:
                result.nx /= length
                result.ny /= length
                result.nz /= length
        
        return result


@dataclass
class BlendShape:
    """
    A blend shape combining multiple morph targets.
    
    Allows complex expressions/deformations by blending multiple
    morph targets with individual weights.
    """
    name: str
    morph_targets: List[MorphTarget] = field(default_factory=list)
    
    # Master weight multiplier
    weight: float = 1.0
    
    def add_target(self, target: MorphTarget):
        """Add a morph target to this blend shape."""
        self.morph_targets.append(target)
    
    def set_target_weight(self, target_name: str, weight: float):
        """Set weight for a specific target."""
        for target in self.morph_targets:
            if target.name == target_name:
                target.weight = max(0.0, min(1.0, weight))
                break
    
    def apply_to_vertices(self, vertices: List[Vertex]) -> List[Vertex]:
        """Apply all morph targets to vertices."""
        result = [v.copy() for v in vertices]
        
        for target in self.morph_targets:
            effective_weight = target.weight * self.weight
            if effective_weight == 0.0:
                continue
            
            for idx in target.deltas:
                if idx < len(result):
                    dx, dy, dz = target.deltas[idx]
                    result[idx].x += dx * effective_weight
                    result[idx].y += dy * effective_weight
                    result[idx].z += dz * effective_weight
        
        return result


class MeshManipulator:
    """
    Advanced mesh manipulation with morph targets and region-based editing.
    
    Provides tools for:
    - Direct vertex manipulation
    - Morph target creation and blending
    - Region-based scaling and deformation
    - Soft selection with falloff
    - Mesh export to common formats
    """
    
    def __init__(self):
        # Base mesh data
        self._vertices: List[Vertex] = []
        self._faces: List[Face] = []
        self._original_vertices: List[Vertex] = []
        
        # Morph targets and blend shapes
        self._morph_targets: Dict[str, MorphTarget] = {}
        self._blend_shapes: Dict[str, BlendShape] = {}
        
        # Regions
        self._regions: Dict[MeshRegion, RegionDefinition] = {}
        
        # State
        self._modified_vertices: List[Vertex] = []
        self._dirty = False
        
        # Animation
        self._morph_animations: Dict[str, Dict] = {}  # name -> {target, start, end, duration}
        
        # Callbacks
        self._on_mesh_changed: List[Callable[[], None]] = []
    
    # ========================================================================
    # Mesh Loading
    # ========================================================================
    
    def load_mesh(self, vertices: List[Vertex], faces: List[Face]):
        """Load mesh data."""
        self._vertices = vertices
        self._original_vertices = [v.copy() for v in vertices]
        self._faces = faces
        self._modified_vertices = [v.copy() for v in vertices]
        self._dirty = True
        
        # Auto-detect regions based on vertex positions
        self._auto_detect_regions()
    
    def load_from_obj(self, obj_path: str) -> bool:
        """Load mesh from OBJ file."""
        try:
            path = Path(obj_path)
            if not path.exists():
                logger.error(f"OBJ file not found: {obj_path}")
                return False
            
            vertices = []
            faces = []
            
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'v':
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append(Vertex(x=x, y=y, z=z))
                    
                    elif parts[0] == 'vn':
                        nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                        if vertices:
                            # Assign to last vertex (simplified)
                            vertices[-1].nx = nx
                            vertices[-1].ny = ny
                            vertices[-1].nz = nz
                    
                    elif parts[0] == 'vt':
                        u, v = float(parts[1]), float(parts[2])
                        if vertices:
                            vertices[-1].u = u
                            vertices[-1].v = v
                    
                    elif parts[0] == 'f':
                        # Parse face (handle v/vt/vn format)
                        indices = []
                        for p in parts[1:]:
                            idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                            indices.append(idx)
                        
                        # Triangulate if needed
                        for i in range(1, len(indices) - 1):
                            faces.append(Face(indices[0], indices[i], indices[i+1]))
            
            self.load_mesh(vertices, faces)
            logger.info(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load OBJ: {e}")
            return False
    
    def load_from_trimesh(self, mesh: Any) -> bool:
        """Load mesh from trimesh object."""
        try:
            vertices = []
            for i, pos in enumerate(mesh.vertices):
                v = Vertex(x=pos[0], y=pos[1], z=pos[2])
                if mesh.vertex_normals is not None and i < len(mesh.vertex_normals):
                    v.nx, v.ny, v.nz = mesh.vertex_normals[i]
                vertices.append(v)
            
            faces = [Face(f[0], f[1], f[2]) for f in mesh.faces]
            
            self.load_mesh(vertices, faces)
            return True
            
        except Exception as e:
            logger.error(f"Failed to load trimesh: {e}")
            return False
    
    # ========================================================================
    # Vertex Manipulation
    # ========================================================================
    
    def move_vertex(self, index: int, dx: float, dy: float, dz: float):
        """Move a single vertex by delta."""
        if 0 <= index < len(self._modified_vertices):
            v = self._modified_vertices[index]
            v.x += dx
            v.y += dy
            v.z += dz
            self._dirty = True
    
    def set_vertex_position(self, index: int, x: float, y: float, z: float):
        """Set absolute position of a vertex."""
        if 0 <= index < len(self._modified_vertices):
            self._modified_vertices[index].set_position(x, y, z)
            self._dirty = True
    
    def move_vertices(
        self,
        indices: List[int],
        dx: float,
        dy: float,
        dz: float,
        falloff: float = 0.0,
        center: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Move multiple vertices with optional falloff.
        
        Args:
            indices: Vertex indices to move
            dx, dy, dz: Movement delta
            falloff: Falloff radius (0 = no falloff, all vertices move equally)
            center: Center point for falloff calculation
        """
        if not indices:
            return
        
        # Calculate center if not provided
        if center is None and falloff > 0:
            sum_x = sum_y = sum_z = 0.0
            for idx in indices:
                if 0 <= idx < len(self._modified_vertices):
                    v = self._modified_vertices[idx]
                    sum_x += v.x
                    sum_y += v.y
                    sum_z += v.z
            n = len(indices)
            center = (sum_x / n, sum_y / n, sum_z / n)
        
        for idx in indices:
            if 0 <= idx < len(self._modified_vertices):
                v = self._modified_vertices[idx]
                
                # Calculate weight based on falloff
                weight = 1.0
                if falloff > 0 and center:
                    dist = math.sqrt(
                        (v.x - center[0])**2 +
                        (v.y - center[1])**2 +
                        (v.z - center[2])**2
                    )
                    weight = max(0.0, 1.0 - dist / falloff)
                
                v.x += dx * weight
                v.y += dy * weight
                v.z += dz * weight
        
        self._dirty = True
    
    def scale_region(
        self,
        region: MeshRegion,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        scale_z: float = 1.0,
        pivot: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Scale vertices in a region around a pivot point.
        
        Args:
            region: The region to scale
            scale_x, scale_y, scale_z: Scale factors for each axis
            pivot: Pivot point for scaling (default: region center)
        """
        region_def = self._regions.get(region)
        if not region_def or not region_def.vertex_indices:
            logger.warning(f"Region not found or empty: {region}")
            return
        
        # Calculate pivot if not provided
        if pivot is None:
            sum_x = sum_y = sum_z = 0.0
            for idx in region_def.vertex_indices:
                if 0 <= idx < len(self._modified_vertices):
                    v = self._modified_vertices[idx]
                    sum_x += v.x
                    sum_y += v.y
                    sum_z += v.z
            n = len(region_def.vertex_indices)
            pivot = (sum_x / n, sum_y / n, sum_z / n)
        
        # Scale vertices
        for idx in region_def.vertex_indices:
            if 0 <= idx < len(self._modified_vertices):
                v = self._modified_vertices[idx]
                weight = region_def.get_weight(idx)
                
                # Scale relative to pivot with weight
                effective_scale_x = 1.0 + (scale_x - 1.0) * weight
                effective_scale_y = 1.0 + (scale_y - 1.0) * weight
                effective_scale_z = 1.0 + (scale_z - 1.0) * weight
                
                v.x = pivot[0] + (v.x - pivot[0]) * effective_scale_x
                v.y = pivot[1] + (v.y - pivot[1]) * effective_scale_y
                v.z = pivot[2] + (v.z - pivot[2]) * effective_scale_z
        
        self._dirty = True
    
    def stretch_region(
        self,
        region: MeshRegion,
        direction: Tuple[float, float, float],
        amount: float,
    ):
        """Stretch a region along a direction."""
        dx, dy, dz = direction
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 0.0001:
            return
        
        # Normalize direction
        dx, dy, dz = dx/length, dy/length, dz/length
        
        region_def = self._regions.get(region)
        if not region_def:
            return
        
        for idx in region_def.vertex_indices:
            if 0 <= idx < len(self._modified_vertices):
                v = self._modified_vertices[idx]
                weight = region_def.get_weight(idx)
                
                # Project vertex onto direction and stretch
                proj = v.x * dx + v.y * dy + v.z * dz
                
                v.x += dx * proj * amount * weight
                v.y += dy * proj * amount * weight
                v.z += dz * proj * amount * weight
        
        self._dirty = True
    
    def reset_to_original(self):
        """Reset all vertices to original positions."""
        self._modified_vertices = [v.copy() for v in self._original_vertices]
        self._dirty = True
    
    def reset_region(self, region: MeshRegion):
        """Reset a region to original positions."""
        region_def = self._regions.get(region)
        if not region_def:
            return
        
        for idx in region_def.vertex_indices:
            if 0 <= idx < len(self._modified_vertices) and idx < len(self._original_vertices):
                self._modified_vertices[idx] = self._original_vertices[idx].copy()
        
        self._dirty = True
    
    # ========================================================================
    # Morph Targets
    # ========================================================================
    
    def create_morph_target(self, name: str) -> MorphTarget:
        """Create a new morph target from current deformation."""
        target = MorphTarget(name=name)
        
        for i, (original, modified) in enumerate(zip(self._original_vertices, self._modified_vertices)):
            dx = modified.x - original.x
            dy = modified.y - original.y
            dz = modified.z - original.z
            
            # Only store non-zero deltas
            if abs(dx) > 0.0001 or abs(dy) > 0.0001 or abs(dz) > 0.0001:
                target.set_delta(i, dx, dy, dz)
        
        self._morph_targets[name] = target
        logger.info(f"Created morph target '{name}' with {len(target.deltas)} vertex deltas")
        return target
    
    def create_morph_from_mesh(self, name: str, target_vertices: List[Vertex]) -> MorphTarget:
        """Create morph target from another mesh's vertex positions."""
        target = MorphTarget(name=name)
        
        for i, (original, modified) in enumerate(zip(self._original_vertices, target_vertices)):
            dx = modified.x - original.x
            dy = modified.y - original.y
            dz = modified.z - original.z
            
            if abs(dx) > 0.0001 or abs(dy) > 0.0001 or abs(dz) > 0.0001:
                target.set_delta(i, dx, dy, dz)
        
        self._morph_targets[name] = target
        return target
    
    def set_morph_weight(self, name: str, weight: float):
        """Set the weight of a morph target."""
        target = self._morph_targets.get(name)
        if target:
            target.weight = max(0.0, min(1.0, weight))
            self._apply_morphs()
    
    def animate_morph(
        self,
        name: str,
        target_weight: float,
        duration: float = 0.5,
    ):
        """Animate a morph target to a weight over time."""
        target = self._morph_targets.get(name)
        if not target:
            return
        
        self._morph_animations[name] = {
            "start_weight": target.weight,
            "target_weight": target_weight,
            "start_time": time.time(),
            "duration": duration,
        }
    
    def update_animations(self) -> bool:
        """Update morph animations. Returns True if any are active."""
        if not self._morph_animations:
            return False
        
        current_time = time.time()
        completed = []
        
        for name, anim in self._morph_animations.items():
            elapsed = current_time - anim["start_time"]
            progress = min(1.0, elapsed / anim["duration"])
            
            # Ease in-out
            if progress < 0.5:
                t = 2 * progress * progress
            else:
                t = 1 - 2 * (1 - progress) * (1 - progress)
            
            weight = anim["start_weight"] + (anim["target_weight"] - anim["start_weight"]) * t
            
            target = self._morph_targets.get(name)
            if target:
                target.weight = weight
            
            if progress >= 1.0:
                completed.append(name)
        
        for name in completed:
            del self._morph_animations[name]
        
        self._apply_morphs()
        return bool(self._morph_animations)
    
    def _apply_morphs(self):
        """Apply all morph targets to get modified vertices."""
        # Start from original
        self._modified_vertices = [v.copy() for v in self._original_vertices]
        
        # Apply each morph
        for target in self._morph_targets.values():
            if target.weight == 0.0:
                continue
            
            for idx, (dx, dy, dz) in target.deltas.items():
                if 0 <= idx < len(self._modified_vertices):
                    v = self._modified_vertices[idx]
                    v.x += dx * target.weight
                    v.y += dy * target.weight
                    v.z += dz * target.weight
        
        self._dirty = True
        self._notify_changed()
    
    def get_morph_target(self, name: str) -> Optional[MorphTarget]:
        """Get a morph target by name."""
        return self._morph_targets.get(name)
    
    def list_morph_targets(self) -> List[str]:
        """List all morph target names."""
        return list(self._morph_targets.keys())
    
    # ========================================================================
    # Blend Shapes
    # ========================================================================
    
    def create_blend_shape(self, name: str, targets: List[str]) -> BlendShape:
        """Create a blend shape from multiple morph targets."""
        blend = BlendShape(name=name)
        
        for target_name in targets:
            target = self._morph_targets.get(target_name)
            if target:
                blend.add_target(target)
        
        self._blend_shapes[name] = blend
        return blend
    
    def set_blend_shape_weight(self, name: str, weight: float):
        """Set master weight for a blend shape."""
        blend = self._blend_shapes.get(name)
        if blend:
            blend.weight = max(0.0, min(1.0, weight))
            self._apply_morphs()
    
    # ========================================================================
    # Regions
    # ========================================================================
    
    def define_region(
        self,
        region: MeshRegion,
        vertex_indices: List[int],
        weights: Optional[Dict[int, float]] = None,
    ):
        """Define a mesh region manually."""
        definition = RegionDefinition(
            name=region.value,
            region_type=region,
            vertex_indices=vertex_indices,
        )
        
        if weights:
            definition.weights = weights
        
        # Calculate bounds
        if vertex_indices and self._vertices:
            xs = [self._vertices[i].x for i in vertex_indices if i < len(self._vertices)]
            ys = [self._vertices[i].y for i in vertex_indices if i < len(self._vertices)]
            zs = [self._vertices[i].z for i in vertex_indices if i < len(self._vertices)]
            
            if xs:
                definition.min_x = min(xs)
                definition.max_x = max(xs)
                definition.min_y = min(ys)
                definition.max_y = max(ys)
                definition.min_z = min(zs)
                definition.max_z = max(zs)
        
        self._regions[region] = definition
    
    def _auto_detect_regions(self):
        """Auto-detect mesh regions based on vertex positions."""
        if not self._vertices:
            return
        
        # Get mesh bounds
        min_y = min(v.y for v in self._vertices)
        max_y = max(v.y for v in self._vertices)
        height = max_y - min_y
        
        if height < 0.0001:
            return
        
        # Define regions based on Y position (assuming Y is up)
        region_bounds = {
            MeshRegion.HEAD: (0.75, 1.0),      # Top 25%
            MeshRegion.NECK: (0.70, 0.78),     # Just below head
            MeshRegion.TORSO: (0.45, 0.72),    # Upper body
            MeshRegion.HIPS: (0.35, 0.48),     # Hip area
            MeshRegion.LEFT_LEG: (0.0, 0.38),  # Lower body (will filter by X)
            MeshRegion.RIGHT_LEG: (0.0, 0.38),
        }
        
        center_x = sum(v.x for v in self._vertices) / len(self._vertices)
        
        for region, (y_min_pct, y_max_pct) in region_bounds.items():
            y_min = min_y + height * y_min_pct
            y_max = min_y + height * y_max_pct
            
            indices = []
            for i, v in enumerate(self._vertices):
                if y_min <= v.y <= y_max:
                    # Filter left/right for legs
                    if region == MeshRegion.LEFT_LEG and v.x > center_x:
                        continue
                    if region == MeshRegion.RIGHT_LEG and v.x <= center_x:
                        continue
                    indices.append(i)
            
            if indices:
                self.define_region(region, indices)
    
    def get_region(self, region: MeshRegion) -> Optional[RegionDefinition]:
        """Get a region definition."""
        return self._regions.get(region)
    
    def list_regions(self) -> List[MeshRegion]:
        """List defined regions."""
        return list(self._regions.keys())
    
    # ========================================================================
    # Export
    # ========================================================================
    
    def export_to_obj(self, path: str) -> bool:
        """Export mesh to OBJ file."""
        try:
            with open(path, 'w') as f:
                f.write("# Exported from Enigma AI Engine Mesh Manipulator\n")
                
                # Write vertices
                for v in self._modified_vertices:
                    f.write(f"v {v.x} {v.y} {v.z}\n")
                
                # Write normals
                for v in self._modified_vertices:
                    f.write(f"vn {v.nx} {v.ny} {v.nz}\n")
                
                # Write UVs
                for v in self._modified_vertices:
                    f.write(f"vt {v.u} {v.v}\n")
                
                # Write faces (1-indexed)
                for face in self._faces:
                    f.write(f"f {face.v1+1} {face.v2+1} {face.v3+1}\n")
            
            logger.info(f"Exported mesh to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export OBJ: {e}")
            return False
    
    def export_to_trimesh(self) -> Any:
        """Export to trimesh object."""
        try:
            import numpy as np
            import trimesh
            
            vertices = np.array([[v.x, v.y, v.z] for v in self._modified_vertices])
            faces = np.array([[f.v1, f.v2, f.v3] for f in self._faces])
            
            return trimesh.Trimesh(vertices=vertices, faces=faces)
            
        except ImportError:
            logger.error("trimesh not available for export")
            return None
    
    def save_morph_targets(self, path: str) -> bool:
        """Save all morph targets to JSON."""
        try:
            data = {}
            for name, target in self._morph_targets.items():
                data[name] = {
                    "deltas": {str(k): v for k, v in target.deltas.items()},
                    "normal_deltas": {str(k): v for k, v in target.normal_deltas.items()},
                }
            
            Path(path).write_text(json.dumps(data, indent=2))
            return True
            
        except Exception as e:
            logger.error(f"Failed to save morph targets: {e}")
            return False
    
    def load_morph_targets(self, path: str) -> bool:
        """Load morph targets from JSON."""
        try:
            data = json.loads(Path(path).read_text())
            
            for name, target_data in data.items():
                target = MorphTarget(name=name)
                target.deltas = {int(k): tuple(v) for k, v in target_data["deltas"].items()}
                if "normal_deltas" in target_data:
                    target.normal_deltas = {int(k): tuple(v) for k, v in target_data["normal_deltas"].items()}
                self._morph_targets[name] = target
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load morph targets: {e}")
            return False
    
    # ========================================================================
    # Callbacks
    # ========================================================================
    
    def on_mesh_changed(self, callback: Callable[[], None]):
        """Register callback for mesh changes."""
        self._on_mesh_changed.append(callback)
    
    def _notify_changed(self):
        """Notify listeners of mesh changes."""
        for callback in self._on_mesh_changed:
            try:
                callback()
            except Exception as e:
                logger.error(f"Mesh change callback error: {e}")
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def vertices(self) -> List[Vertex]:
        """Get current (modified) vertices."""
        return self._modified_vertices
    
    @property
    def original_vertices(self) -> List[Vertex]:
        """Get original vertices."""
        return self._original_vertices
    
    @property
    def faces(self) -> List[Face]:
        """Get faces."""
        return self._faces
    
    @property
    def vertex_count(self) -> int:
        return len(self._vertices)
    
    @property
    def face_count(self) -> int:
        return len(self._faces)
    
    @property
    def is_dirty(self) -> bool:
        return self._dirty
    
    def mark_clean(self):
        """Mark mesh as clean (after rendering)."""
        self._dirty = False


# Global instance
_mesh_manipulator: Optional[MeshManipulator] = None


def get_mesh_manipulator() -> MeshManipulator:
    """Get or create the global mesh manipulator instance."""
    global _mesh_manipulator
    if _mesh_manipulator is None:
        _mesh_manipulator = MeshManipulator()
    return _mesh_manipulator


__all__ = [
    'MeshManipulator',
    'Vertex',
    'Face',
    'MeshRegion',
    'RegionDefinition',
    'MorphTarget',
    'BlendShape',
    'get_mesh_manipulator',
]
