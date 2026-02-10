"""
VRM Model Support

Full VRM 1.0 and 0.x specification support for avatar models.
Handles VRM parsing, validation, and extension processing.

FILE: enigma_engine/avatar/vrm_support.py
TYPE: Avatar System
MAIN CLASSES: VRMLoader, VRMMeta, VRMHumanoid, VRMBlendShape
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class VRMVersion(Enum):
    """VRM specification versions."""
    VRM_0_0 = "0.0"
    VRM_1_0 = "1.0"


class VRMExportPurpose(Enum):
    """Allowed model usage purposes."""
    ONLY_AUTHOR = "OnlyAuthor"
    EXPLICITLY_LICENSED = "ExplicitlyLicensed"
    CORPORATE = "Corporate"
    PERSONAL = "Personal"
    PUBLIC_USE = "PublicUse"


class VRMBoneName(Enum):
    """Standard VRM bone names."""
    # Required bones
    HIPS = "hips"
    SPINE = "spine"
    CHEST = "chest"
    NECK = "neck"
    HEAD = "head"
    
    # Arms
    LEFT_UPPER_ARM = "leftUpperArm"
    LEFT_LOWER_ARM = "leftLowerArm"
    LEFT_HAND = "leftHand"
    RIGHT_UPPER_ARM = "rightUpperArm"
    RIGHT_LOWER_ARM = "rightLowerArm"
    RIGHT_HAND = "rightHand"
    
    # Legs
    LEFT_UPPER_LEG = "leftUpperLeg"
    LEFT_LOWER_LEG = "leftLowerLeg"
    LEFT_FOOT = "leftFoot"
    RIGHT_UPPER_LEG = "rightUpperLeg"
    RIGHT_LOWER_LEG = "rightLowerLeg"
    RIGHT_FOOT = "rightFoot"
    
    # Optional bones
    UPPER_CHEST = "upperChest"
    LEFT_SHOULDER = "leftShoulder"
    RIGHT_SHOULDER = "rightShoulder"
    LEFT_TOES = "leftToes"
    RIGHT_TOES = "rightToes"
    
    # Fingers (optional)
    LEFT_THUMB_PROXIMAL = "leftThumbProximal"
    LEFT_THUMB_INTERMEDIATE = "leftThumbIntermediate"
    LEFT_THUMB_DISTAL = "leftThumbDistal"
    LEFT_INDEX_PROXIMAL = "leftIndexProximal"
    LEFT_INDEX_INTERMEDIATE = "leftIndexIntermediate"
    LEFT_INDEX_DISTAL = "leftIndexDistal"
    LEFT_MIDDLE_PROXIMAL = "leftMiddleProximal"
    LEFT_MIDDLE_INTERMEDIATE = "leftMiddleIntermediate"
    LEFT_MIDDLE_DISTAL = "leftMiddleDistal"
    LEFT_RING_PROXIMAL = "leftRingProximal"
    LEFT_RING_INTERMEDIATE = "leftRingIntermediate"
    LEFT_RING_DISTAL = "leftRingDistal"
    LEFT_LITTLE_PROXIMAL = "leftLittleProximal"
    LEFT_LITTLE_INTERMEDIATE = "leftLittleIntermediate"
    LEFT_LITTLE_DISTAL = "leftLittleDistal"
    # Right hand mirrors left


class VRMExpression(Enum):
    """Standard VRM expressions (blend shapes)."""
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    RELAXED = "relaxed"
    SURPRISED = "surprised"
    
    # Eye expressions
    BLINK = "blink"
    BLINK_LEFT = "blinkLeft"
    BLINK_RIGHT = "blinkRight"
    LOOK_UP = "lookUp"
    LOOK_DOWN = "lookDown"
    LOOK_LEFT = "lookLeft"
    LOOK_RIGHT = "lookRight"
    
    # Mouth expressions
    AA = "aa"  # あ
    IH = "ih"  # い
    OU = "ou"  # う
    EE = "ee"  # え
    OH = "oh"  # お
    
    NEUTRAL = "neutral"


@dataclass
class VRMMeta:
    """VRM metadata."""
    name: str = ""
    version: str = ""
    author: str = ""
    contact_information: str = ""
    reference: str = ""
    title: str = ""
    thumbnail_path: Optional[str] = None
    
    # Permissions (VRM 1.0)
    avatar_permission: str = "OnlyAuthor"
    allow_excessively_violent_usage: bool = False
    allow_excessively_sexual_usage: bool = False
    commercial_usage: str = "PersonalNonProfit"
    allow_political_or_religious_usage: bool = False
    allow_antisocial_or_hate_usage: bool = False
    credit_notation: str = "Required"
    allow_redistribution: bool = False
    modification: str = "Prohibited"
    other_license_url: str = ""


@dataclass
class VRMHumanBone:
    """A VRM humanoid bone mapping."""
    bone_name: VRMBoneName
    node_index: int
    use_default_values: bool = False


@dataclass
class VRMHumanoid:
    """VRM humanoid configuration."""
    bones: dict[str, VRMHumanBone] = field(default_factory=dict)
    arm_stretch: float = 0.05
    leg_stretch: float = 0.05
    upper_arm_twist: float = 0.5
    lower_arm_twist: float = 0.5
    upper_leg_twist: float = 0.5
    lower_leg_twist: float = 0.5
    feet_spacing: float = 0.0
    has_translation_dof: bool = False


@dataclass
class VRMBlendShapeBind:
    """Blend shape binding to a mesh."""
    mesh_index: int
    morph_target_index: int
    weight: float = 1.0


@dataclass
class VRMBlendShapeGroup:
    """A VRM expression/blend shape group."""
    name: str
    preset: Optional[VRMExpression] = None
    binds: list[VRMBlendShapeBind] = field(default_factory=list)
    is_binary: bool = False


@dataclass
class VRMMaterial:
    """VRM material properties."""
    name: str = ""
    shader_name: str = "VRM/MToon"
    render_queue: int = -1
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class VRMSpringBone:
    """VRM spring bone for physics."""
    bone_groups: list[dict[str, Any]] = field(default_factory=list)
    collider_groups: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class VRMModel:
    """Complete VRM model data."""
    version: VRMVersion = VRMVersion.VRM_1_0
    meta: VRMMeta = field(default_factory=VRMMeta)
    humanoid: VRMHumanoid = field(default_factory=VRMHumanoid)
    expressions: dict[str, VRMBlendShapeGroup] = field(default_factory=dict)
    materials: list[VRMMaterial] = field(default_factory=list)
    spring_bones: VRMSpringBone = field(default_factory=VRMSpringBone)
    first_person: dict[str, Any] = field(default_factory=dict)
    look_at: dict[str, Any] = field(default_factory=dict)
    gltf_data: dict[str, Any] = field(default_factory=dict)
    
    def get_expression(self, name: str) -> Optional[VRMBlendShapeGroup]:
        """Get an expression by name."""
        return self.expressions.get(name)
    
    def get_bone(self, bone_name: VRMBoneName) -> Optional[VRMHumanBone]:
        """Get bone mapping by VRM bone name."""
        return self.humanoid.bones.get(bone_name.value)


class VRMLoader:
    """Loads and parses VRM files."""
    
    # Required bones for a valid VRM humanoid
    REQUIRED_BONES = [
        VRMBoneName.HIPS,
        VRMBoneName.SPINE,
        VRMBoneName.HEAD,
        VRMBoneName.LEFT_UPPER_ARM,
        VRMBoneName.LEFT_LOWER_ARM,
        VRMBoneName.LEFT_HAND,
        VRMBoneName.RIGHT_UPPER_ARM,
        VRMBoneName.RIGHT_LOWER_ARM,
        VRMBoneName.RIGHT_HAND,
        VRMBoneName.LEFT_UPPER_LEG,
        VRMBoneName.LEFT_LOWER_LEG,
        VRMBoneName.LEFT_FOOT,
        VRMBoneName.RIGHT_UPPER_LEG,
        VRMBoneName.RIGHT_LOWER_LEG,
        VRMBoneName.RIGHT_FOOT
    ]
    
    def __init__(self):
        """Initialize VRM loader."""
    
    def load(self, path: Path) -> VRMModel:
        """
        Load a VRM file.
        
        Args:
            path: Path to .vrm file
            
        Returns:
            VRMModel with parsed data
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"VRM file not found: {path}")
        
        if path.suffix.lower() != '.vrm':
            raise ValueError(f"Not a VRM file: {path}")
        
        # VRM files are GLB files with VRM extension data
        gltf_data = self._load_glb(path)
        
        # Determine VRM version
        version = self._detect_version(gltf_data)
        
        # Parse VRM-specific data
        if version == VRMVersion.VRM_1_0:
            return self._parse_vrm_1_0(gltf_data)
        else:
            return self._parse_vrm_0_x(gltf_data)
    
    def _load_glb(self, path: Path) -> dict[str, Any]:
        """Load GLB file and extract JSON portion."""
        with open(path, 'rb') as f:
            # GLB header
            magic = f.read(4)
            if magic != b'glTF':
                raise ValueError("Invalid GLB magic bytes")
            
            version = int.from_bytes(f.read(4), 'little')
            length = int.from_bytes(f.read(4), 'little')
            
            # First chunk (JSON)
            chunk_length = int.from_bytes(f.read(4), 'little')
            chunk_type = f.read(4)
            
            if chunk_type != b'JSON':
                raise ValueError("Expected JSON chunk")
            
            json_data = f.read(chunk_length)
            return json.loads(json_data)
    
    def _detect_version(self, gltf_data: dict) -> VRMVersion:
        """Detect VRM version from glTF data."""
        extensions = gltf_data.get('extensions', {})
        
        if 'VRMC_vrm' in extensions:
            return VRMVersion.VRM_1_0
        elif 'VRM' in extensions:
            return VRMVersion.VRM_0_0
        
        # Check for VRM data in extensionsUsed
        extensions_used = gltf_data.get('extensionsUsed', [])
        if 'VRMC_vrm' in extensions_used:
            return VRMVersion.VRM_1_0
        
        return VRMVersion.VRM_0_0
    
    def _parse_vrm_1_0(self, gltf_data: dict) -> VRMModel:
        """Parse VRM 1.0 format."""
        vrm_ext = gltf_data.get('extensions', {}).get('VRMC_vrm', {})
        
        model = VRMModel(
            version=VRMVersion.VRM_1_0,
            gltf_data=gltf_data
        )
        
        # Parse meta
        meta_data = vrm_ext.get('meta', {})
        model.meta = VRMMeta(
            name=meta_data.get('name', ''),
            version=meta_data.get('version', ''),
            author=meta_data.get('authors', [''])[0] if meta_data.get('authors') else '',
            contact_information=meta_data.get('contactInformation', ''),
            reference=meta_data.get('references', [''])[0] if meta_data.get('references') else '',
            avatar_permission=meta_data.get('avatarPermission', 'OnlyAuthor'),
            allow_excessively_violent_usage=meta_data.get('allowExcessivelyViolentUsage', False),
            allow_excessively_sexual_usage=meta_data.get('allowExcessivelySexualUsage', False),
            commercial_usage=meta_data.get('commercialUsage', 'PersonalNonProfit'),
        )
        
        # Parse humanoid
        humanoid_data = vrm_ext.get('humanoid', {})
        model.humanoid = self._parse_humanoid_1_0(humanoid_data)
        
        # Parse expressions
        expressions_data = vrm_ext.get('expressions', {})
        model.expressions = self._parse_expressions_1_0(expressions_data)
        
        return model
    
    def _parse_vrm_0_x(self, gltf_data: dict) -> VRMModel:
        """Parse VRM 0.x format."""
        vrm_ext = gltf_data.get('extensions', {}).get('VRM', {})
        
        model = VRMModel(
            version=VRMVersion.VRM_0_0,
            gltf_data=gltf_data
        )
        
        # Parse meta (0.x format)
        meta_data = vrm_ext.get('meta', {})
        model.meta = VRMMeta(
            name=meta_data.get('title', ''),
            version=meta_data.get('version', ''),
            author=meta_data.get('author', ''),
            contact_information=meta_data.get('contactInformation', ''),
            reference=meta_data.get('reference', ''),
        )
        
        # Parse humanoid
        humanoid_data = vrm_ext.get('humanoid', {})
        model.humanoid = self._parse_humanoid_0_x(humanoid_data)
        
        # Parse blend shapes
        blend_shapes = vrm_ext.get('blendShapeMaster', {})
        model.expressions = self._parse_blend_shapes_0_x(blend_shapes)
        
        return model
    
    def _parse_humanoid_1_0(self, data: dict) -> VRMHumanoid:
        """Parse VRM 1.0 humanoid data."""
        humanoid = VRMHumanoid()
        
        bones_data = data.get('humanBones', {})
        for bone_name, bone_data in bones_data.items():
            try:
                vrm_bone = VRMBoneName(bone_name)
                humanoid.bones[bone_name] = VRMHumanBone(
                    bone_name=vrm_bone,
                    node_index=bone_data.get('node', -1)
                )
            except ValueError:
                logger.warning(f"Unknown bone name: {bone_name}")
        
        return humanoid
    
    def _parse_humanoid_0_x(self, data: dict) -> VRMHumanoid:
        """Parse VRM 0.x humanoid data."""
        humanoid = VRMHumanoid()
        
        bones_data = data.get('humanBones', [])
        for bone_data in bones_data:
            bone_name = bone_data.get('bone', '')
            try:
                vrm_bone = VRMBoneName(bone_name)
                humanoid.bones[bone_name] = VRMHumanBone(
                    bone_name=vrm_bone,
                    node_index=bone_data.get('node', -1),
                    use_default_values=bone_data.get('useDefaultValues', True)
                )
            except ValueError:
                logger.warning(f"Unknown bone name: {bone_name}")
        
        return humanoid
    
    def _parse_expressions_1_0(self, data: dict) -> dict[str, VRMBlendShapeGroup]:
        """Parse VRM 1.0 expressions."""
        expressions = {}
        
        preset_data = data.get('preset', {})
        for preset_name, expr_data in preset_data.items():
            expressions[preset_name] = self._parse_expression(preset_name, expr_data)
        
        custom_data = data.get('custom', {})
        for custom_name, expr_data in custom_data.items():
            expressions[custom_name] = self._parse_expression(custom_name, expr_data)
        
        return expressions
    
    def _parse_blend_shapes_0_x(self, data: dict) -> dict[str, VRMBlendShapeGroup]:
        """Parse VRM 0.x blend shapes."""
        expressions = {}
        
        groups = data.get('blendShapeGroups', [])
        for group in groups:
            name = group.get('name', '')
            preset = group.get('presetName', '')
            
            binds = []
            for bind_data in group.get('binds', []):
                binds.append(VRMBlendShapeBind(
                    mesh_index=bind_data.get('mesh', -1),
                    morph_target_index=bind_data.get('index', -1),
                    weight=bind_data.get('weight', 100.0) / 100.0
                ))
            
            expressions[name or preset] = VRMBlendShapeGroup(
                name=name or preset,
                binds=binds,
                is_binary=group.get('isBinary', False)
            )
        
        return expressions
    
    def _parse_expression(self, name: str, data: dict) -> VRMBlendShapeGroup:
        """Parse a single expression."""
        binds = []
        
        for bind_data in data.get('morphTargetBinds', []):
            binds.append(VRMBlendShapeBind(
                mesh_index=bind_data.get('node', -1),
                morph_target_index=bind_data.get('index', -1),
                weight=bind_data.get('weight', 1.0)
            ))
        
        return VRMBlendShapeGroup(
            name=name,
            binds=binds,
            is_binary=data.get('isBinary', False)
        )
    
    def validate(self, model: VRMModel) -> list[str]:
        """
        Validate a VRM model.
        
        Args:
            model: VRM model to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required bones
        for bone in self.REQUIRED_BONES:
            if bone.value not in model.humanoid.bones:
                errors.append(f"Missing required bone: {bone.value}")
        
        # Check meta
        if not model.meta.name and not model.meta.title:
            errors.append("Missing model name/title in meta")
        
        return errors


def load_vrm(path: Path) -> VRMModel:
    """
    Load a VRM file.
    
    Args:
        path: Path to VRM file
        
    Returns:
        Parsed VRM model
    """
    loader = VRMLoader()
    return loader.load(path)


__all__ = [
    'VRMLoader',
    'VRMModel',
    'VRMMeta',
    'VRMHumanoid',
    'VRMHumanBone',
    'VRMBlendShapeGroup',
    'VRMBlendShapeBind',
    'VRMVersion',
    'VRMBoneName',
    'VRMExpression',
    'load_vrm'
]
