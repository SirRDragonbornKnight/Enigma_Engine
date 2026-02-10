"""
================================================================================
AVATAR MODEL MANAGER - MANAGE AND EXPAND AVATAR COLLECTION
================================================================================

Easy management of avatar models - import from various sources, convert formats,
and organize your avatar library.

FILE: enigma_engine/avatar/model_manager.py
TYPE: Avatar Management
MAIN CLASSES: AvatarModelManager, AvatarModel, AvatarSource

FEATURES:
    - Import avatars from various sources (VRoid, Ready Player Me, MakeHuman)
    - Support multiple formats (VRM, FBX, GLB, GLTF)
    - Auto-detect rigging and animations
    - Thumbnail generation
    - Category organization
    - Model validation

USAGE:
    from enigma_engine.avatar.model_manager import AvatarModelManager
    
    manager = AvatarModelManager()
    
    # Import a new avatar
    avatar = manager.import_avatar("path/to/avatar.vrm")
    
    # List all avatars
    avatars = manager.list_avatars()
    
    # Get avatar by category
    anime_avatars = manager.get_by_category("anime")
"""

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from ..config import CONFIG

logger = logging.getLogger(__name__)


class AvatarFormat(Enum):
    """Supported avatar formats."""
    VRM = "vrm"           # VRoid format (recommended)
    GLTF = "gltf"         # Standard 3D format
    GLB = "glb"           # Binary GLTF
    FBX = "fbx"           # Autodesk format
    OBJ = "obj"           # Basic mesh
    DAE = "dae"           # Collada
    BLEND = "blend"       # Blender native
    UNKNOWN = "unknown"


class AvatarCategory(Enum):
    """Avatar categories for organization."""
    ANIME = "anime"
    REALISTIC = "realistic"
    CARTOON = "cartoon"
    FANTASY = "fantasy"
    SCIFI = "scifi"
    ANIMAL = "animal"
    ROBOT = "robot"
    CUSTOM = "custom"
    OTHER = "other"


class AvatarSource(Enum):
    """Known avatar sources."""
    VROID = "vroid"              # VRoid Studio/Hub
    READY_PLAYER_ME = "rpm"      # Ready Player Me
    MAKEHUMAN = "makehuman"      # MakeHuman
    MIXAMO = "mixamo"            # Adobe Mixamo
    SKETCHFAB = "sketchfab"      # Sketchfab
    LOCAL = "local"              # Local file
    CUSTOM = "custom"            # Custom creation
    UNKNOWN = "unknown"


@dataclass
class AvatarRigging:
    """Avatar rigging information."""
    
    is_rigged: bool = False
    bone_count: int = 0
    has_humanoid_rig: bool = False
    has_face_rig: bool = False
    has_finger_bones: bool = False
    rig_type: str = "unknown"  # humanoid, quadruped, custom
    blend_shapes: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'is_rigged': self.is_rigged,
            'bone_count': self.bone_count,
            'has_humanoid_rig': self.has_humanoid_rig,
            'has_face_rig': self.has_face_rig,
            'has_finger_bones': self.has_finger_bones,
            'rig_type': self.rig_type,
            'blend_shapes': self.blend_shapes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AvatarRigging':
        return cls(
            is_rigged=data.get('is_rigged', False),
            bone_count=data.get('bone_count', 0),
            has_humanoid_rig=data.get('has_humanoid_rig', False),
            has_face_rig=data.get('has_face_rig', False),
            has_finger_bones=data.get('has_finger_bones', False),
            rig_type=data.get('rig_type', 'unknown'),
            blend_shapes=data.get('blend_shapes', []),
        )


@dataclass
class AvatarModel:
    """Avatar model metadata."""
    
    id: str
    name: str
    format: AvatarFormat
    category: AvatarCategory
    source: AvatarSource
    
    file_path: Path
    thumbnail_path: Optional[Path] = None
    
    description: str = ""
    author: str = ""
    license: str = ""
    version: str = "1.0"
    
    rigging: AvatarRigging = field(default_factory=AvatarRigging)
    
    # Technical details
    vertex_count: int = 0
    triangle_count: int = 0
    texture_count: int = 0
    material_count: int = 0
    file_size: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)
    custom_data: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'format': self.format.value,
            'category': self.category.value,
            'source': self.source.value,
            'file_path': str(self.file_path),
            'thumbnail_path': str(self.thumbnail_path) if self.thumbnail_path else None,
            'description': self.description,
            'author': self.author,
            'license': self.license,
            'version': self.version,
            'rigging': self.rigging.to_dict(),
            'vertex_count': self.vertex_count,
            'triangle_count': self.triangle_count,
            'texture_count': self.texture_count,
            'material_count': self.material_count,
            'file_size': self.file_size,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags,
            'custom_data': self.custom_data,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AvatarModel':
        return cls(
            id=data['id'],
            name=data['name'],
            format=AvatarFormat(data.get('format', 'unknown')),
            category=AvatarCategory(data.get('category', 'other')),
            source=AvatarSource(data.get('source', 'unknown')),
            file_path=Path(data['file_path']),
            thumbnail_path=Path(data['thumbnail_path']) if data.get('thumbnail_path') else None,
            description=data.get('description', ''),
            author=data.get('author', ''),
            license=data.get('license', ''),
            version=data.get('version', '1.0'),
            rigging=AvatarRigging.from_dict(data.get('rigging', {})),
            vertex_count=data.get('vertex_count', 0),
            triangle_count=data.get('triangle_count', 0),
            texture_count=data.get('texture_count', 0),
            material_count=data.get('material_count', 0),
            file_size=data.get('file_size', 0),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            tags=data.get('tags', []),
            custom_data=data.get('custom_data', {}),
        )


class AvatarModelManager:
    """
    Manage avatar model library.
    
    Features:
    - Import avatars from files
    - Organize by category
    - Auto-detect format and source
    - Generate thumbnails
    - Export/share avatars
    """
    
    # Format extension mapping
    FORMAT_MAP = {
        '.vrm': AvatarFormat.VRM,
        '.gltf': AvatarFormat.GLTF,
        '.glb': AvatarFormat.GLB,
        '.fbx': AvatarFormat.FBX,
        '.obj': AvatarFormat.OBJ,
        '.dae': AvatarFormat.DAE,
        '.blend': AvatarFormat.BLEND,
    }
    
    def __init__(self, models_dir: Path = None):
        """
        Initialize avatar model manager.
        
        Args:
            models_dir: Directory for avatar models
        """
        self.models_dir = models_dir or Path(CONFIG.get("data_dir", "data")) / "avatar" / "models"
        self.thumbnails_dir = self.models_dir.parent / "thumbnails"
        self.registry_path = self.models_dir.parent / "avatar_registry.json"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        
        # Load registry
        self.avatars: dict[str, AvatarModel] = {}
        self._load_registry()
        
        logger.info(f"Avatar model manager initialized with {len(self.avatars)} avatars")
    
    def _load_registry(self):
        """Load avatar registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                
                for avatar_data in data.get('avatars', []):
                    try:
                        avatar = AvatarModel.from_dict(avatar_data)
                        self.avatars[avatar.id] = avatar
                    except Exception as e:
                        logger.warning(f"Could not load avatar: {e}")
            except Exception as e:
                logger.error(f"Error loading avatar registry: {e}")
        
        # Scan for unregistered avatars
        self._scan_for_new_avatars()
    
    def _save_registry(self):
        """Save avatar registry to disk."""
        data = {
            'version': '1.0',
            'updated_at': datetime.now().isoformat(),
            'avatars': [a.to_dict() for a in self.avatars.values()]
        }
        
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _scan_for_new_avatars(self):
        """Scan models directory for unregistered avatars."""
        registered_paths = {a.file_path for a in self.avatars.values()}
        
        for ext in self.FORMAT_MAP:
            for file_path in self.models_dir.glob(f"**/*{ext}"):
                if file_path not in registered_paths:
                    try:
                        avatar = self._create_avatar_from_file(file_path)
                        self.avatars[avatar.id] = avatar
                        logger.info(f"Auto-registered avatar: {avatar.name}")
                    except Exception as e:
                        logger.warning(f"Could not register {file_path}: {e}")
        
        self._save_registry()
    
    def _generate_id(self, file_path: Path) -> str:
        """Generate unique ID for avatar."""
        content = f"{file_path.name}{file_path.stat().st_size}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _detect_format(self, file_path: Path) -> AvatarFormat:
        """Detect avatar format from file extension."""
        ext = file_path.suffix.lower()
        return self.FORMAT_MAP.get(ext, AvatarFormat.UNKNOWN)
    
    def _detect_source(self, file_path: Path, content: bytes = None) -> AvatarSource:
        """Detect avatar source from file metadata."""
        name_lower = file_path.stem.lower()
        
        # Check filename patterns
        if 'vroid' in name_lower or file_path.suffix.lower() == '.vrm':
            return AvatarSource.VROID
        if 'rpm' in name_lower or 'readyplayerme' in name_lower:
            return AvatarSource.READY_PLAYER_ME
        if 'makehuman' in name_lower or 'mhx' in name_lower:
            return AvatarSource.MAKEHUMAN
        if 'mixamo' in name_lower:
            return AvatarSource.MIXAMO
        
        return AvatarSource.LOCAL
    
    def _detect_category(self, file_path: Path, name: str) -> AvatarCategory:
        """Detect avatar category from name and location."""
        name_lower = name.lower()
        path_str = str(file_path).lower()
        
        # Check path for category folders
        for category in AvatarCategory:
            if category.value in path_str:
                return category
        
        # Check name patterns
        anime_keywords = ['anime', 'vtuber', 'chibi', 'manga', 'kawaii']
        if any(kw in name_lower for kw in anime_keywords):
            return AvatarCategory.ANIME
        
        fantasy_keywords = ['dragon', 'elf', 'demon', 'angel', 'fairy', 'wolf']
        if any(kw in name_lower for kw in fantasy_keywords):
            return AvatarCategory.FANTASY
        
        robot_keywords = ['robot', 'mech', 'android', 'cyborg']
        if any(kw in name_lower for kw in robot_keywords):
            return AvatarCategory.ROBOT
        
        return AvatarCategory.OTHER
    
    def _analyze_model(self, file_path: Path) -> tuple[AvatarRigging, dict]:
        """Analyze model file for technical details."""
        rigging = AvatarRigging()
        stats = {
            'vertex_count': 0,
            'triangle_count': 0,
            'texture_count': 0,
            'material_count': 0,
        }
        
        # Try to analyze with pygltflib for GLTF/GLB/VRM
        try:
            import pygltflib
            if file_path.suffix.lower() in ['.gltf', '.glb', '.vrm']:
                gltf = pygltflib.GLTF2().load(str(file_path))
                
                # Count meshes
                for mesh in gltf.meshes or []:
                    for primitive in mesh.primitives or []:
                        stats['triangle_count'] += 1000  # Estimate
                
                # Check for rigging
                if gltf.skins:
                    rigging.is_rigged = True
                    if gltf.skins[0].joints:
                        rigging.bone_count = len(gltf.skins[0].joints)
                        rigging.has_humanoid_rig = rigging.bone_count > 15
                
                # Count materials
                stats['material_count'] = len(gltf.materials) if gltf.materials else 0
                stats['texture_count'] = len(gltf.textures) if gltf.textures else 0
                
        except ImportError:
            logger.debug("pygltflib not available for model analysis")
        except Exception as e:
            logger.debug(f"Could not analyze model: {e}")
        
        return rigging, stats
    
    def _create_avatar_from_file(self, file_path: Path) -> AvatarModel:
        """Create avatar model entry from file."""
        avatar_format = self._detect_format(file_path)
        source = self._detect_source(file_path)
        name = file_path.stem.replace('_', ' ').replace('-', ' ').title()
        category = self._detect_category(file_path, name)
        
        # Analyze model
        rigging, stats = self._analyze_model(file_path)
        
        return AvatarModel(
            id=self._generate_id(file_path),
            name=name,
            format=avatar_format,
            category=category,
            source=source,
            file_path=file_path,
            rigging=rigging,
            file_size=file_path.stat().st_size,
            **stats
        )
    
    def import_avatar(
        self,
        source_path: Path,
        name: str = None,
        category: AvatarCategory = None,
        copy_file: bool = True
    ) -> AvatarModel:
        """
        Import an avatar from a file.
        
        Args:
            source_path: Path to avatar file
            name: Custom name (uses filename if not provided)
            category: Category to assign (auto-detected if not provided)
            copy_file: Whether to copy file to models directory
        
        Returns:
            Imported AvatarModel
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Avatar file not found: {source_path}")
        
        avatar_format = self._detect_format(source_path)
        if avatar_format == AvatarFormat.UNKNOWN:
            raise ValueError(f"Unsupported avatar format: {source_path.suffix}")
        
        # Determine destination
        if copy_file:
            dest_dir = self.models_dir / (category.value if category else 'imported')
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / source_path.name
            
            # Handle duplicates
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{source_path.stem}_{counter}{source_path.suffix}"
                counter += 1
            
            shutil.copy2(source_path, dest_path)
            file_path = dest_path
        else:
            file_path = source_path
        
        # Create avatar entry
        avatar = self._create_avatar_from_file(file_path)
        
        # Apply custom settings
        if name:
            avatar.name = name
        if category:
            avatar.category = category
        
        # Register
        self.avatars[avatar.id] = avatar
        self._save_registry()
        
        logger.info(f"Imported avatar: {avatar.name} ({avatar.format.value})")
        return avatar
    
    def import_from_url(
        self,
        url: str,
        name: str = None,
        category: AvatarCategory = None
    ) -> Optional[AvatarModel]:
        """
        Import avatar from URL.
        
        Args:
            url: URL to avatar file
            name: Custom name
            category: Category to assign
        
        Returns:
            Imported AvatarModel or None if failed
        """
        try:
            import urllib.request

            # Parse filename from URL
            from urllib.parse import urlparse
            parsed = urlparse(url)
            filename = Path(parsed.path).name or "downloaded_avatar.glb"
            
            # Download to temp
            temp_path = self.models_dir.parent / "temp" / filename
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading avatar from {url}")
            urllib.request.urlretrieve(url, temp_path)
            
            # Import
            avatar = self.import_avatar(temp_path, name, category)
            
            # Cleanup temp
            temp_path.unlink()
            
            return avatar
            
        except Exception as e:
            logger.error(f"Failed to import from URL: {e}")
            return None
    
    def get_avatar(self, avatar_id: str) -> Optional[AvatarModel]:
        """Get avatar by ID."""
        return self.avatars.get(avatar_id)
    
    def get_by_name(self, name: str) -> Optional[AvatarModel]:
        """Get avatar by name."""
        name_lower = name.lower()
        for avatar in self.avatars.values():
            if avatar.name.lower() == name_lower:
                return avatar
        return None
    
    def list_avatars(self) -> list[AvatarModel]:
        """List all avatars."""
        return list(self.avatars.values())
    
    def get_by_category(self, category: AvatarCategory) -> list[AvatarModel]:
        """Get avatars by category."""
        if isinstance(category, str):
            category = AvatarCategory(category)
        return [a for a in self.avatars.values() if a.category == category]
    
    def get_by_format(self, avatar_format: AvatarFormat) -> list[AvatarModel]:
        """Get avatars by format."""
        if isinstance(avatar_format, str):
            avatar_format = AvatarFormat(avatar_format)
        return [a for a in self.avatars.values() if a.format == avatar_format]
    
    def search(self, query: str) -> list[AvatarModel]:
        """Search avatars by name, tags, or description."""
        query_lower = query.lower()
        results = []
        
        for avatar in self.avatars.values():
            if (query_lower in avatar.name.lower() or
                query_lower in avatar.description.lower() or
                any(query_lower in tag.lower() for tag in avatar.tags)):
                results.append(avatar)
        
        return results
    
    def update_avatar(self, avatar_id: str, **updates):
        """Update avatar metadata."""
        avatar = self.avatars.get(avatar_id)
        if not avatar:
            raise ValueError(f"Avatar not found: {avatar_id}")
        
        for key, value in updates.items():
            if hasattr(avatar, key):
                setattr(avatar, key, value)
        
        self._save_registry()
    
    def delete_avatar(self, avatar_id: str, delete_file: bool = False):
        """
        Delete avatar from registry.
        
        Args:
            avatar_id: Avatar ID
            delete_file: Also delete the model file
        """
        avatar = self.avatars.get(avatar_id)
        if not avatar:
            return
        
        if delete_file and avatar.file_path.exists():
            avatar.file_path.unlink()
        
        del self.avatars[avatar_id]
        self._save_registry()
        
        logger.info(f"Deleted avatar: {avatar.name}")
    
    def get_stats(self) -> dict:
        """Get library statistics."""
        by_category = {}
        by_format = {}
        total_size = 0
        
        for avatar in self.avatars.values():
            cat = avatar.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            
            fmt = avatar.format.value
            by_format[fmt] = by_format.get(fmt, 0) + 1
            
            total_size += avatar.file_size
        
        return {
            'total_avatars': len(self.avatars),
            'by_category': by_category,
            'by_format': by_format,
            'total_size_mb': total_size / (1024 * 1024),
        }
    
    def export_list(self, output_path: Path = None) -> str:
        """Export avatar list as JSON."""
        output_path = output_path or self.models_dir.parent / "avatar_list.json"
        
        data = {
            'exported_at': datetime.now().isoformat(),
            'stats': self.get_stats(),
            'avatars': [
                {
                    'id': a.id,
                    'name': a.name,
                    'category': a.category.value,
                    'format': a.format.value,
                    'source': a.source.value,
                    'has_rig': a.rigging.is_rigged,
                }
                for a in self.avatars.values()
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(output_path)


# Global instance
_manager: Optional[AvatarModelManager] = None


def get_avatar_manager() -> AvatarModelManager:
    """Get or create global avatar model manager."""
    global _manager
    if _manager is None:
        _manager = AvatarModelManager()
    return _manager


# Sample avatars that can be downloaded
SAMPLE_AVATAR_URLS = {
    "vroid_sample": {
        "url": "https://hub.vroid.com/sample_avatars/sample.vrm",
        "name": "VRoid Sample Avatar",
        "category": "anime",
        "description": "Sample anime-style avatar from VRoid Hub"
    },
    "rpm_robot": {
        "url": "https://models.readyplayer.me/sample-robot.glb",
        "name": "RPM Robot",
        "category": "robot",
        "description": "Robot avatar from Ready Player Me"
    },
}
