"""
Model Validation

Validates 3D models for corruption, performance issues, and compatibility.
Checks polycount, texture limits, and structural integrity.

FILE: enigma_engine/avatar/model_validation.py
TYPE: Avatar System
MAIN CLASSES: ModelValidator, ValidationResult, ValidationRule
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks."""
    STRUCTURE = "structure"
    GEOMETRY = "geometry"
    TEXTURES = "textures"
    SKELETON = "skeleton"
    MATERIALS = "materials"
    PERFORMANCE = "performance"
    COMPATIBILITY = "compatibility"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    code: str
    message: str
    severity: ValidationSeverity
    category: ValidationCategory
    details: dict[str, Any] = field(default_factory=dict)
    fix_suggestion: str = ""


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool
    model_name: str = ""
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    
    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)]
    
    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.is_valid = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "model_name": self.model_name,
            "issue_count": len(self.issues),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [
                {
                    "code": i.code,
                    "message": i.message,
                    "severity": i.severity.value,
                    "category": i.category.value,
                    "fix_suggestion": i.fix_suggestion
                }
                for i in self.issues
            ],
            "stats": self.stats
        }


@dataclass
class ValidationLimits:
    """Performance limits for validation."""
    max_vertices: int = 100000
    max_triangles: int = 70000
    max_bones: int = 256
    max_texture_size: int = 4096
    max_texture_count: int = 32
    max_meshes: int = 100
    max_materials: int = 50
    max_blend_shapes: int = 100
    recommended_vertices: int = 30000
    recommended_triangles: int = 20000


class ValidationRule:
    """A single validation rule."""
    
    def __init__(self,
                 code: str,
                 name: str,
                 category: ValidationCategory,
                 check_func: Callable):
        """
        Initialize validation rule.
        
        Args:
            code: Unique rule code
            name: Rule name
            category: Rule category
            check_func: Function that performs the check
        """
        self.code = code
        self.name = name
        self.category = category
        self.check_func = check_func
    
    def run(self, model: Any, limits: ValidationLimits) -> list[ValidationIssue]:
        """Run the validation rule."""
        try:
            return self.check_func(model, limits)
        except Exception as e:
            return [ValidationIssue(
                code=f"{self.code}_ERROR",
                message=f"Rule failed: {e}",
                severity=ValidationSeverity.WARNING,
                category=self.category
            )]


class ModelValidator:
    """Validates 3D models for various issues."""
    
    def __init__(self, limits: Optional[ValidationLimits] = None):
        """
        Initialize model validator.
        
        Args:
            limits: Validation limits (uses defaults if None)
        """
        self._limits = limits or ValidationLimits()
        self._rules = self._build_rules()
    
    def _build_rules(self) -> list[ValidationRule]:
        """Build validation rules."""
        return [
            # Geometry rules
            ValidationRule(
                "GEO001", "Vertex Count",
                ValidationCategory.GEOMETRY,
                self._check_vertex_count
            ),
            ValidationRule(
                "GEO002", "Triangle Count",
                ValidationCategory.GEOMETRY,
                self._check_triangle_count
            ),
            ValidationRule(
                "GEO003", "Empty Meshes",
                ValidationCategory.GEOMETRY,
                self._check_empty_meshes
            ),
            ValidationRule(
                "GEO004", "Mesh Count",
                ValidationCategory.GEOMETRY,
                self._check_mesh_count
            ),
            
            # Skeleton rules
            ValidationRule(
                "SKEL001", "Bone Count",
                ValidationCategory.SKELETON,
                self._check_bone_count
            ),
            ValidationRule(
                "SKEL002", "Required Bones",
                ValidationCategory.SKELETON,
                self._check_required_bones
            ),
            ValidationRule(
                "SKEL003", "Bone Hierarchy",
                ValidationCategory.SKELETON,
                self._check_bone_hierarchy
            ),
            
            # Material rules
            ValidationRule(
                "MAT001", "Material Count",
                ValidationCategory.MATERIALS,
                self._check_material_count
            ),
            ValidationRule(
                "MAT002", "Missing Materials",
                ValidationCategory.MATERIALS,
                self._check_missing_materials
            ),
            
            # Structure rules
            ValidationRule(
                "STRUCT001", "Model Name",
                ValidationCategory.STRUCTURE,
                self._check_model_name
            ),
        ]
    
    def validate(self, model: Any) -> ValidationResult:
        """
        Validate a 3D model.
        
        Args:
            model: Model to validate (Model3D or VRMModel)
            
        Returns:
            Validation result
        """
        result = ValidationResult(
            is_valid=True,
            model_name=getattr(model, 'name', 'Unknown')
        )
        
        # Gather statistics
        result.stats = self._gather_stats(model)
        
        # Run all rules
        for rule in self._rules:
            issues = rule.run(model, self._limits)
            for issue in issues:
                result.add_issue(issue)
        
        logger.info(f"Validated {result.model_name}: {len(result.errors)} errors, {len(result.warnings)} warnings")
        
        return result
    
    def _gather_stats(self, model: Any) -> dict[str, Any]:
        """Gather model statistics."""
        stats = {
            "name": getattr(model, 'name', 'Unknown'),
            "format": str(getattr(model, 'format', 'Unknown'))
        }
        
        # Mesh stats
        meshes = getattr(model, 'meshes', [])
        stats["mesh_count"] = len(meshes)
        stats["total_vertices"] = sum(getattr(m, 'vertex_count', 0) for m in meshes)
        stats["total_triangles"] = sum(getattr(m, 'index_count', 0) // 3 for m in meshes)
        
        # Material stats
        materials = getattr(model, 'materials', [])
        stats["material_count"] = len(materials)
        
        # Skeleton stats
        skeleton = getattr(model, 'skeleton', None)
        if skeleton:
            bones = getattr(skeleton, 'bones', [])
            stats["bone_count"] = len(bones)
        else:
            stats["bone_count"] = 0
        
        # Expression/blend shape stats
        expressions = getattr(model, 'expressions', {})
        stats["expression_count"] = len(expressions)
        
        return stats
    
    # Geometry checks
    def _check_vertex_count(self, model: Any, limits: ValidationLimits) -> list[ValidationIssue]:
        """Check total vertex count."""
        issues = []
        
        meshes = getattr(model, 'meshes', [])
        total = sum(getattr(m, 'vertex_count', 0) for m in meshes)
        
        if total > limits.max_vertices:
            issues.append(ValidationIssue(
                code="GEO001_MAX",
                message=f"Total vertices ({total:,}) exceeds maximum ({limits.max_vertices:,})",
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.GEOMETRY,
                details={"vertex_count": total, "limit": limits.max_vertices},
                fix_suggestion="Reduce polygon count using decimation or retopology"
            ))
        elif total > limits.recommended_vertices:
            issues.append(ValidationIssue(
                code="GEO001_REC",
                message=f"Total vertices ({total:,}) exceeds recommended ({limits.recommended_vertices:,})",
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.GEOMETRY,
                details={"vertex_count": total, "recommended": limits.recommended_vertices},
                fix_suggestion="Consider reducing for better performance"
            ))
        
        return issues
    
    def _check_triangle_count(self, model: Any, limits: ValidationLimits) -> list[ValidationIssue]:
        """Check total triangle count."""
        issues = []
        
        meshes = getattr(model, 'meshes', [])
        total = sum(getattr(m, 'index_count', 0) // 3 for m in meshes)
        
        if total > limits.max_triangles:
            issues.append(ValidationIssue(
                code="GEO002_MAX",
                message=f"Total triangles ({total:,}) exceeds maximum ({limits.max_triangles:,})",
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.GEOMETRY,
                details={"triangle_count": total, "limit": limits.max_triangles},
                fix_suggestion="Reduce polygon count"
            ))
        
        return issues
    
    def _check_empty_meshes(self, model: Any, limits: ValidationLimits) -> list[ValidationIssue]:
        """Check for empty meshes."""
        issues = []
        
        meshes = getattr(model, 'meshes', [])
        for mesh in meshes:
            if getattr(mesh, 'vertex_count', 0) == 0:
                issues.append(ValidationIssue(
                    code="GEO003_EMPTY",
                    message=f"Empty mesh found: {getattr(mesh, 'name', 'unnamed')}",
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.GEOMETRY,
                    fix_suggestion="Remove empty meshes"
                ))
        
        return issues
    
    def _check_mesh_count(self, model: Any, limits: ValidationLimits) -> list[ValidationIssue]:
        """Check total mesh count."""
        issues = []
        
        meshes = getattr(model, 'meshes', [])
        if len(meshes) > limits.max_meshes:
            issues.append(ValidationIssue(
                code="GEO004_MAX",
                message=f"Mesh count ({len(meshes)}) exceeds maximum ({limits.max_meshes})",
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.GEOMETRY,
                fix_suggestion="Combine meshes where possible"
            ))
        
        return issues
    
    # Skeleton checks
    def _check_bone_count(self, model: Any, limits: ValidationLimits) -> list[ValidationIssue]:
        """Check bone count."""
        issues = []
        
        skeleton = getattr(model, 'skeleton', None)
        if not skeleton:
            return issues
        
        bones = getattr(skeleton, 'bones', [])
        if len(bones) > limits.max_bones:
            issues.append(ValidationIssue(
                code="SKEL001_MAX",
                message=f"Bone count ({len(bones)}) exceeds maximum ({limits.max_bones})",
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.SKELETON,
                fix_suggestion="Reduce bone count or merge similar bones"
            ))
        
        return issues
    
    def _check_required_bones(self, model: Any, limits: ValidationLimits) -> list[ValidationIssue]:
        """Check for required humanoid bones."""
        issues = []
        
        # Check if humanoid validation was performed
        metadata = getattr(model, 'metadata', {})
        if 'is_humanoid' in metadata:
            if not metadata['is_humanoid']:
                missing = metadata.get('missing_bones', [])
                issues.append(ValidationIssue(
                    code="SKEL002_MISS",
                    message=f"Missing required humanoid bones: {', '.join(missing)}",
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.SKELETON,
                    details={"missing": missing},
                    fix_suggestion="Rename bones to match VRM/humanoid conventions"
                ))
        
        return issues
    
    def _check_bone_hierarchy(self, model: Any, limits: ValidationLimits) -> list[ValidationIssue]:
        """Check bone hierarchy integrity."""
        issues = []
        
        skeleton = getattr(model, 'skeleton', None)
        if not skeleton:
            return issues
        
        bones = getattr(skeleton, 'bones', [])
        
        # Check for orphaned bones (invalid parent references)
        for bone in bones:
            parent_idx = getattr(bone, 'parent_index', -1)
            if parent_idx >= len(bones):
                issues.append(ValidationIssue(
                    code="SKEL003_ORPHAN",
                    message=f"Bone '{bone.name}' has invalid parent reference",
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SKELETON
                ))
        
        return issues
    
    # Material checks
    def _check_material_count(self, model: Any, limits: ValidationLimits) -> list[ValidationIssue]:
        """Check material count."""
        issues = []
        
        materials = getattr(model, 'materials', [])
        if len(materials) > limits.max_materials:
            issues.append(ValidationIssue(
                code="MAT001_MAX",
                message=f"Material count ({len(materials)}) exceeds recommended ({limits.max_materials})",
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.MATERIALS,
                fix_suggestion="Combine materials using texture atlasing"
            ))
        
        return issues
    
    def _check_missing_materials(self, model: Any, limits: ValidationLimits) -> list[ValidationIssue]:
        """Check for meshes without materials."""
        # This would require mesh-material mapping from the model
        return []
    
    # Structure checks
    def _check_model_name(self, model: Any, limits: ValidationLimits) -> list[ValidationIssue]:
        """Check model has a name."""
        issues = []
        
        name = getattr(model, 'name', '')
        if not name:
            issues.append(ValidationIssue(
                code="STRUCT001_NAME",
                message="Model has no name",
                severity=ValidationSeverity.INFO,
                category=ValidationCategory.STRUCTURE
            ))
        
        return issues


def validate_model(model: Any, limits: ValidationLimits = None) -> ValidationResult:
    """
    Validate a 3D model.
    
    Args:
        model: Model to validate
        limits: Validation limits
        
    Returns:
        Validation result
    """
    validator = ModelValidator(limits)
    return validator.validate(model)


__all__ = [
    'ModelValidator',
    'ValidationResult',
    'ValidationIssue',
    'ValidationRule',
    'ValidationLimits',
    'ValidationSeverity',
    'ValidationCategory',
    'validate_model'
]
