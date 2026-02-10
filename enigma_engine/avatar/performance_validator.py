"""
Avatar Performance Validator

Check avatar models for performance issues like high polygon counts,
large textures, and excessive bone counts.

FILE: enigma_engine/avatar/performance_validator.py
TYPE: Avatar
MAIN CLASSES: AvatarPerformanceValidator, PerformanceReport, PerformancePreset
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceLevel(Enum):
    """Performance rating levels."""
    EXCELLENT = "excellent"  # Can run on mobile/low-end
    GOOD = "good"           # Desktop without issues
    ACCEPTABLE = "acceptable"  # Works but may have issues
    POOR = "poor"           # Performance problems likely
    CRITICAL = "critical"   # Will cause serious issues


class DeviceTarget(Enum):
    """Target device categories."""
    MOBILE = "mobile"
    LOW_END_PC = "low_end_pc"
    MID_RANGE_PC = "mid_range_pc"
    HIGH_END_PC = "high_end_pc"
    VR = "vr"


@dataclass
class PerformancePreset:
    """Performance limits for a target device."""
    name: str
    device: DeviceTarget
    
    # Polygon limits
    max_triangles: int
    max_vertices: int
    
    # Texture limits
    max_texture_size: int  # Single dimension (e.g., 2048)
    max_texture_memory_mb: float
    max_textures: int
    
    # Skinning limits
    max_bones: int
    max_bone_influences: int
    
    # Material limits
    max_materials: int
    max_blend_shapes: int
    
    # Other
    recommended_fps: int = 30


# Standard presets
PRESETS = {
    DeviceTarget.MOBILE: PerformancePreset(
        name="Mobile",
        device=DeviceTarget.MOBILE,
        max_triangles=10000,
        max_vertices=15000,
        max_texture_size=1024,
        max_texture_memory_mb=32,
        max_textures=4,
        max_bones=75,
        max_bone_influences=2,
        max_materials=4,
        max_blend_shapes=30,
        recommended_fps=30
    ),
    DeviceTarget.LOW_END_PC: PerformancePreset(
        name="Low-End PC",
        device=DeviceTarget.LOW_END_PC,
        max_triangles=32000,
        max_vertices=50000,
        max_texture_size=2048,
        max_texture_memory_mb=128,
        max_textures=8,
        max_bones=128,
        max_bone_influences=4,
        max_materials=8,
        max_blend_shapes=64,
        recommended_fps=30
    ),
    DeviceTarget.MID_RANGE_PC: PerformancePreset(
        name="Mid-Range PC",
        device=DeviceTarget.MID_RANGE_PC,
        max_triangles=70000,
        max_vertices=100000,
        max_texture_size=4096,
        max_texture_memory_mb=512,
        max_textures=16,
        max_bones=256,
        max_bone_influences=4,
        max_materials=16,
        max_blend_shapes=128,
        recommended_fps=60
    ),
    DeviceTarget.HIGH_END_PC: PerformancePreset(
        name="High-End PC",
        device=DeviceTarget.HIGH_END_PC,
        max_triangles=150000,
        max_vertices=200000,
        max_texture_size=8192,
        max_texture_memory_mb=2048,
        max_textures=32,
        max_bones=512,
        max_bone_influences=8,
        max_materials=32,
        max_blend_shapes=256,
        recommended_fps=90
    ),
    DeviceTarget.VR: PerformancePreset(
        name="VR",
        device=DeviceTarget.VR,
        max_triangles=32000,
        max_vertices=50000,
        max_texture_size=2048,
        max_texture_memory_mb=128,
        max_textures=8,
        max_bones=128,
        max_bone_influences=4,
        max_materials=8,
        max_blend_shapes=64,
        recommended_fps=72
    )
}


@dataclass
class PerformanceIssue:
    """A detected performance issue."""
    category: str
    severity: PerformanceLevel
    message: str
    current_value: Any
    limit_value: Any
    suggestion: str


@dataclass
class ModelStats:
    """Statistics about a model."""
    triangles: int = 0
    vertices: int = 0
    bones: int = 0
    max_bone_influences: int = 0
    materials: int = 0
    blend_shapes: int = 0
    textures: list[dict[str, Any]] = field(default_factory=list)
    texture_memory_mb: float = 0
    mesh_count: int = 0
    has_physics: bool = False
    draw_calls_estimate: int = 0
    file_size_mb: float = 0


@dataclass
class PerformanceReport:
    """Complete performance analysis report."""
    model_path: str
    stats: ModelStats
    target_device: DeviceTarget
    overall_rating: PerformanceLevel
    issues: list[PerformanceIssue]
    passed: bool
    recommendations: list[str]
    estimated_fps: Optional[int] = None


class AvatarPerformanceValidator:
    """
    Validate avatar model performance and provide optimization suggestions.
    """
    
    def __init__(self, target_device: DeviceTarget = DeviceTarget.MID_RANGE_PC):
        self.target_device = target_device
        self.preset = PRESETS[target_device]
    
    def set_target_device(self, device: DeviceTarget):
        """Change target device."""
        self.target_device = device
        self.preset = PRESETS[device]
    
    def analyze_model(
        self,
        model_path: str,
        model_data: Optional[dict[str, Any]] = None
    ) -> PerformanceReport:
        """
        Analyze a model file for performance issues.
        
        Args:
            model_path: Path to the model file
            model_data: Pre-parsed model data (optional)
        
        Returns:
            PerformanceReport with analysis results
        """
        path = Path(model_path)
        
        # Get model stats
        if model_data:
            stats = self._extract_stats_from_data(model_data)
        else:
            stats = self._extract_stats_from_file(path)
        
        stats.file_size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
        
        # Check against limits
        issues = self._check_limits(stats)
        
        # Determine overall rating
        overall_rating = self._calculate_overall_rating(issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, stats)
        
        # Estimate FPS
        estimated_fps = self._estimate_fps(stats)
        
        return PerformanceReport(
            model_path=str(model_path),
            stats=stats,
            target_device=self.target_device,
            overall_rating=overall_rating,
            issues=issues,
            passed=overall_rating in (PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD),
            recommendations=recommendations,
            estimated_fps=estimated_fps
        )
    
    def _extract_stats_from_file(self, path: Path) -> ModelStats:
        """Extract stats from model file."""
        stats = ModelStats()
        
        suffix = path.suffix.lower()
        
        if suffix == '.vrm':
            stats = self._parse_vrm(path)
        elif suffix in ('.glb', '.gltf'):
            stats = self._parse_gltf(path)
        elif suffix == '.fbx':
            stats = self._parse_fbx(path)
        else:
            logger.warning(f"Unknown format: {suffix}, using estimates")
            stats = self._estimate_from_file_size(path)
        
        return stats
    
    def _parse_vrm(self, path: Path) -> ModelStats:
        """Parse VRM file for statistics."""
        try:
            import struct
            
            with open(path, 'rb') as f:
                # Read GLB header
                magic = f.read(4)
                if magic != b'glTF':
                    return self._estimate_from_file_size(path)
                
                version = struct.unpack('<I', f.read(4))[0]
                length = struct.unpack('<I', f.read(4))[0]
                
                # Read JSON chunk
                chunk_length = struct.unpack('<I', f.read(4))[0]
                chunk_type = struct.unpack('<I', f.read(4))[0]
                
                if chunk_type == 0x4E4F534A:  # JSON
                    json_data = json.loads(f.read(chunk_length).decode('utf-8'))
                    return self._extract_stats_from_data(json_data)
        except Exception as e:
            logger.debug(f"Error parsing VRM: {e}")
        
        return self._estimate_from_file_size(path)
    
    def _parse_gltf(self, path: Path) -> ModelStats:
        """Parse GLTF/GLB file for statistics."""
        try:
            if path.suffix.lower() == '.gltf':
                with open(path) as f:
                    data = json.load(f)
                    return self._extract_stats_from_data(data)
            else:
                return self._parse_vrm(path)  # GLB uses same format
        except Exception as e:
            logger.debug(f"Error parsing GLTF: {e}")
        
        return self._estimate_from_file_size(path)
    
    def _parse_fbx(self, path: Path) -> ModelStats:
        """Parse FBX file (requires external library)."""
        # FBX requires specialized parsing
        return self._estimate_from_file_size(path)
    
    def _extract_stats_from_data(self, data: dict[str, Any]) -> ModelStats:
        """Extract stats from parsed model data."""
        stats = ModelStats()
        
        # Count vertices and triangles from accessors/primitives
        if 'meshes' in data:
            for mesh in data['meshes']:
                stats.mesh_count += 1
                
                if 'primitives' in mesh:
                    for prim in mesh['primitives']:
                        # Count triangles (assuming index buffer)
                        if 'indices' in prim and 'accessors' in data:
                            idx = prim['indices']
                            if idx < len(data['accessors']):
                                acc = data['accessors'][idx]
                                count = acc.get('count', 0)
                                stats.triangles += count // 3
                        
                        # Count vertices from POSITION accessor
                        if 'attributes' in prim:
                            pos_idx = prim['attributes'].get('POSITION')
                            if pos_idx is not None and 'accessors' in data:
                                if pos_idx < len(data['accessors']):
                                    acc = data['accessors'][pos_idx]
                                    stats.vertices += acc.get('count', 0)
        
        # Count materials
        if 'materials' in data:
            stats.materials = len(data['materials'])
            stats.draw_calls_estimate = stats.materials
        
        # Count bones from skin
        if 'skins' in data:
            for skin in data['skins']:
                if 'joints' in skin:
                    stats.bones = max(stats.bones, len(skin['joints']))
        
        # Count blend shapes
        if 'meshes' in data:
            for mesh in data['meshes']:
                if 'extras' in mesh:
                    target_names = mesh['extras'].get('targetNames', [])
                    stats.blend_shapes += len(target_names)
                
                for prim in mesh.get('primitives', []):
                    if 'targets' in prim:
                        stats.blend_shapes = max(stats.blend_shapes, len(prim['targets']))
        
        # Count textures and estimate memory
        if 'images' in data:
            for img in data['images']:
                tex_info = {
                    'name': img.get('name', 'unknown'),
                    'uri': img.get('uri', ''),
                    'mimeType': img.get('mimeType', '')
                }
                stats.textures.append(tex_info)
                
                # Estimate texture memory (assume 2048x2048 RGBA)
                stats.texture_memory_mb += 16  # 2048*2048*4 bytes
        
        return stats
    
    def _estimate_from_file_size(self, path: Path) -> ModelStats:
        """Estimate stats based on file size."""
        stats = ModelStats()
        
        if not path.exists():
            return stats
        
        file_size = path.stat().st_size
        
        # Very rough estimates
        stats.triangles = file_size // 100
        stats.vertices = stats.triangles // 2
        stats.bones = min(128, file_size // 100000)
        stats.materials = min(16, file_size // 500000)
        
        return stats
    
    def _check_limits(self, stats: ModelStats) -> list[PerformanceIssue]:
        """Check stats against preset limits."""
        issues = []
        preset = self.preset
        
        # Triangle count
        if stats.triangles > preset.max_triangles:
            severity = self._get_severity(stats.triangles, preset.max_triangles)
            issues.append(PerformanceIssue(
                category="Polygons",
                severity=severity,
                message=f"Triangle count ({stats.triangles:,}) exceeds limit ({preset.max_triangles:,})",
                current_value=stats.triangles,
                limit_value=preset.max_triangles,
                suggestion="Use mesh decimation to reduce polygon count"
            ))
        
        # Vertex count
        if stats.vertices > preset.max_vertices:
            severity = self._get_severity(stats.vertices, preset.max_vertices)
            issues.append(PerformanceIssue(
                category="Vertices",
                severity=severity,
                message=f"Vertex count ({stats.vertices:,}) exceeds limit ({preset.max_vertices:,})",
                current_value=stats.vertices,
                limit_value=preset.max_vertices,
                suggestion="Merge duplicate vertices and remove hidden geometry"
            ))
        
        # Bone count
        if stats.bones > preset.max_bones:
            severity = self._get_severity(stats.bones, preset.max_bones)
            issues.append(PerformanceIssue(
                category="Bones",
                severity=severity,
                message=f"Bone count ({stats.bones}) exceeds limit ({preset.max_bones})",
                current_value=stats.bones,
                limit_value=preset.max_bones,
                suggestion="Reduce skeleton complexity or remove unused bones"
            ))
        
        # Material count
        if stats.materials > preset.max_materials:
            severity = self._get_severity(stats.materials, preset.max_materials)
            issues.append(PerformanceIssue(
                category="Materials",
                severity=severity,
                message=f"Material count ({stats.materials}) exceeds limit ({preset.max_materials})",
                current_value=stats.materials,
                limit_value=preset.max_materials,
                suggestion="Merge materials using texture atlases"
            ))
        
        # Blend shape count
        if stats.blend_shapes > preset.max_blend_shapes:
            severity = self._get_severity(stats.blend_shapes, preset.max_blend_shapes)
            issues.append(PerformanceIssue(
                category="Blend Shapes",
                severity=severity,
                message=f"Blend shape count ({stats.blend_shapes}) exceeds limit ({preset.max_blend_shapes})",
                current_value=stats.blend_shapes,
                limit_value=preset.max_blend_shapes,
                suggestion="Remove unused blend shapes or combine similar ones"
            ))
        
        # Texture memory
        if stats.texture_memory_mb > preset.max_texture_memory_mb:
            severity = self._get_severity(stats.texture_memory_mb, preset.max_texture_memory_mb)
            issues.append(PerformanceIssue(
                category="Texture Memory",
                severity=severity,
                message=f"Texture memory ({stats.texture_memory_mb:.1f}MB) exceeds limit ({preset.max_texture_memory_mb}MB)",
                current_value=stats.texture_memory_mb,
                limit_value=preset.max_texture_memory_mb,
                suggestion="Reduce texture resolution or use compression"
            ))
        
        # Texture count
        if len(stats.textures) > preset.max_textures:
            severity = self._get_severity(len(stats.textures), preset.max_textures)
            issues.append(PerformanceIssue(
                category="Texture Count",
                severity=severity,
                message=f"Texture count ({len(stats.textures)}) exceeds limit ({preset.max_textures})",
                current_value=len(stats.textures),
                limit_value=preset.max_textures,
                suggestion="Combine textures using atlases"
            ))
        
        return issues
    
    def _get_severity(self, current: float, limit: float) -> PerformanceLevel:
        """Determine severity based on how much limit is exceeded."""
        ratio = current / limit
        
        if ratio <= 1.0:
            return PerformanceLevel.EXCELLENT
        elif ratio <= 1.5:
            return PerformanceLevel.ACCEPTABLE
        elif ratio <= 2.0:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    def _calculate_overall_rating(self, issues: list[PerformanceIssue]) -> PerformanceLevel:
        """Calculate overall performance rating."""
        if not issues:
            return PerformanceLevel.EXCELLENT
        
        severities = [issue.severity for issue in issues]
        
        if PerformanceLevel.CRITICAL in severities:
            return PerformanceLevel.CRITICAL
        elif PerformanceLevel.POOR in severities:
            return PerformanceLevel.POOR
        elif PerformanceLevel.ACCEPTABLE in severities:
            return PerformanceLevel.ACCEPTABLE
        else:
            return PerformanceLevel.GOOD
    
    def _generate_recommendations(
        self,
        issues: list[PerformanceIssue],
        stats: ModelStats
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Add issue-specific suggestions
        for issue in issues:
            recommendations.append(f"[{issue.category}] {issue.suggestion}")
        
        # General recommendations
        if stats.triangles > 50000:
            recommendations.append("Consider creating LOD (Level of Detail) versions")
        
        if stats.materials > 4:
            recommendations.append("Batch materials where possible to reduce draw calls")
        
        if stats.texture_memory_mb > 64:
            recommendations.append("Use GPU texture compression (BC7, ETC2, ASTC)")
        
        if stats.bones > 128:
            recommendations.append("Use GPU skinning if available")
        
        if stats.blend_shapes > 50:
            recommendations.append("Group blend shapes by region for selective updates")
        
        return recommendations
    
    def _estimate_fps(self, stats: ModelStats) -> int:
        """Estimate FPS based on model complexity."""
        # Very rough estimation
        base_fps = self.preset.recommended_fps
        
        # Penalty for high poly count
        poly_ratio = stats.triangles / self.preset.max_triangles
        base_fps = int(base_fps / max(1, poly_ratio))
        
        # Penalty for many materials
        mat_ratio = stats.materials / max(1, self.preset.max_materials)
        base_fps = int(base_fps / max(1, mat_ratio * 0.5))
        
        # Penalty for blend shapes
        if stats.blend_shapes > 50:
            base_fps = int(base_fps * 0.9)
        
        return max(10, min(base_fps, 144))
    
    def quick_check(self, model_path: str) -> tuple[bool, str]:
        """
        Quick pass/fail check for a model.
        
        Args:
            model_path: Path to model file
        
        Returns:
            Tuple of (passed, summary_message)
        """
        report = self.analyze_model(model_path)
        
        if report.passed:
            return True, f"Model passes for {self.preset.name}"
        else:
            critical_issues = [i for i in report.issues if i.severity == PerformanceLevel.CRITICAL]
            if critical_issues:
                return False, f"CRITICAL: {critical_issues[0].message}"
            
            return False, f"{len(report.issues)} performance issues detected"
    
    def to_dict(self, report: PerformanceReport) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "model_path": report.model_path,
            "target_device": report.target_device.value,
            "overall_rating": report.overall_rating.value,
            "passed": report.passed,
            "estimated_fps": report.estimated_fps,
            "stats": {
                "triangles": report.stats.triangles,
                "vertices": report.stats.vertices,
                "bones": report.stats.bones,
                "materials": report.stats.materials,
                "blend_shapes": report.stats.blend_shapes,
                "texture_count": len(report.stats.textures),
                "texture_memory_mb": report.stats.texture_memory_mb
            },
            "issues": [
                {
                    "category": i.category,
                    "severity": i.severity.value,
                    "message": i.message,
                    "suggestion": i.suggestion
                }
                for i in report.issues
            ],
            "recommendations": report.recommendations
        }


def validate_avatar(
    model_path: str,
    target: DeviceTarget = DeviceTarget.MID_RANGE_PC
) -> PerformanceReport:
    """
    Validate an avatar model for performance.
    
    Args:
        model_path: Path to model file
        target: Target device
    
    Returns:
        PerformanceReport with analysis
    """
    validator = AvatarPerformanceValidator(target)
    return validator.analyze_model(model_path)
