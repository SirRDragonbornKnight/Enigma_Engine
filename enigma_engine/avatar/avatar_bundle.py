"""
Avatar Bundle System (.forgeavatar format)

Provides a shareable avatar format that bundles:
- Avatar images (PNG/GIF for different emotions)
- Configuration (type, mode, emotion mappings)
- Metadata (name, author, description)
- Optional animations and 3D models

Bundle Structure:
    my_avatar.forgeavatar (ZIP file)
    ├── manifest.json       # Metadata and config
    ├── base.png            # Default avatar image
    ├── emotions/           # Emotion-specific images
    │   ├── happy.png
    │   ├── sad.png
    │   └── ...
    ├── animations/         # Optional GIF/sprite sheets
    │   ├── idle.gif
    │   ├── talking.gif
    │   └── ...
    └── models/             # Optional 3D models
        └── avatar.glb
"""

import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class AvatarBundleVersion(Enum):
    """Bundle format versions."""
    V1 = "1.0"
    CURRENT = "1.0"


@dataclass
class AvatarManifest:
    """Manifest describing the avatar bundle contents."""
    
    # Metadata
    name: str = "Unnamed Avatar"
    author: str = "Unknown"
    description: str = ""
    version: str = "1.0"
    bundle_version: str = AvatarBundleVersion.CURRENT.value
    
    # Avatar settings
    avatar_type: str = "HUMAN"  # HUMAN, ANIMAL, ROBOT, FANTASY, ABSTRACT, CUSTOM
    default_mode: str = "PNG_BOUNCE"  # PNG_BOUNCE, ANIMATED_2D, SKELETAL_3D
    
    # File references (relative paths within bundle)
    base_image: str = "base.png"
    emotion_images: dict[str, str] = field(default_factory=dict)
    animation_files: dict[str, str] = field(default_factory=dict)
    model_file: Optional[str] = None
    
    # Emotion mappings (for non-human types)
    emotion_mapping: dict[str, str] = field(default_factory=dict)
    
    # Display settings
    default_size: list[int] = field(default_factory=lambda: [256, 256])
    anchor_point: list[float] = field(default_factory=lambda: [0.5, 1.0])  # Center bottom
    
    # Tags for discovery
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'AvatarManifest':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class AvatarBundle:
    """
    Handler for .forgeavatar bundle files.
    
    Usage:
        # Create a bundle
        bundle = AvatarBundle()
        bundle.manifest.name = "Happy Bot"
        bundle.manifest.author = "Me"
        bundle.set_base_image("path/to/base.png")
        bundle.add_emotion("happy", "path/to/happy.png")
        bundle.save("happy_bot.forgeavatar")
        
        # Load a bundle
        bundle = AvatarBundle.load("happy_bot.forgeavatar")
        print(bundle.manifest.name)
        base_img = bundle.get_base_image()  # Returns bytes
    """
    
    def __init__(self, manifest: Optional[AvatarManifest] = None):
        self.manifest = manifest or AvatarManifest()
        self._files: dict[str, bytes] = {}  # path -> content
        self._temp_dir: Optional[Path] = None
    
    def set_base_image(self, path: str):
        """Set the base avatar image from a file path."""
        path = Path(path)
        if path.exists():
            self._files["base.png"] = path.read_bytes()
            self.manifest.base_image = "base.png"
    
    def set_base_image_bytes(self, data: bytes, filename: str = "base.png"):
        """Set the base avatar image from bytes."""
        self._files[filename] = data
        self.manifest.base_image = filename
    
    def add_emotion(self, emotion: str, path: str):
        """Add an emotion-specific image."""
        path = Path(path)
        if path.exists():
            emotion_path = f"emotions/{emotion}.png"
            self._files[emotion_path] = path.read_bytes()
            self.manifest.emotion_images[emotion] = emotion_path
    
    def add_emotion_bytes(self, emotion: str, data: bytes):
        """Add an emotion-specific image from bytes."""
        emotion_path = f"emotions/{emotion}.png"
        self._files[emotion_path] = data
        self.manifest.emotion_images[emotion] = emotion_path
    
    def add_animation(self, name: str, path: str):
        """Add an animation file (GIF or sprite sheet)."""
        path = Path(path)
        if path.exists():
            ext = path.suffix
            anim_path = f"animations/{name}{ext}"
            self._files[anim_path] = path.read_bytes()
            self.manifest.animation_files[name] = anim_path
    
    def add_animation_bytes(self, name: str, data: bytes, extension: str = ".gif"):
        """Add an animation from bytes."""
        anim_path = f"animations/{name}{extension}"
        self._files[anim_path] = data
        self.manifest.animation_files[name] = anim_path
    
    def set_model(self, path: str):
        """Set a 3D model file (GLB/GLTF)."""
        path = Path(path)
        if path.exists():
            ext = path.suffix
            model_path = f"models/avatar{ext}"
            self._files[model_path] = path.read_bytes()
            self.manifest.model_file = model_path
    
    def save(self, output_path: str):
        """Save the bundle to a .forgeavatar file."""
        output_path = Path(output_path)
        
        # Ensure .forgeavatar extension
        if output_path.suffix != '.forgeavatar':
            output_path = output_path.with_suffix('.forgeavatar')
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Write manifest
            manifest_json = json.dumps(self.manifest.to_dict(), indent=2)
            zf.writestr("manifest.json", manifest_json)
            
            # Write all files
            for path, content in self._files.items():
                zf.writestr(path, content)
        
        return output_path
    
    @classmethod
    def load(cls, bundle_path: str) -> 'AvatarBundle':
        """Load a bundle from a .forgeavatar file."""
        bundle_path = Path(bundle_path)
        
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {bundle_path}")
        
        bundle = cls()
        
        with zipfile.ZipFile(bundle_path, 'r') as zf:
            # Read manifest
            if "manifest.json" in zf.namelist():
                manifest_data = json.loads(zf.read("manifest.json"))
                bundle.manifest = AvatarManifest.from_dict(manifest_data)
            
            # Read all files into memory
            for name in zf.namelist():
                if name != "manifest.json":
                    bundle._files[name] = zf.read(name)
        
        return bundle
    
    def get_base_image(self) -> Optional[bytes]:
        """Get the base image bytes."""
        return self._files.get(self.manifest.base_image)
    
    def get_emotion_image(self, emotion: str) -> Optional[bytes]:
        """Get an emotion-specific image."""
        path = self.manifest.emotion_images.get(emotion)
        return self._files.get(path) if path else None
    
    def get_animation(self, name: str) -> Optional[bytes]:
        """Get an animation file."""
        path = self.manifest.animation_files.get(name)
        return self._files.get(path) if path else None
    
    def get_model(self) -> Optional[bytes]:
        """Get the 3D model file."""
        if self.manifest.model_file:
            return self._files.get(self.manifest.model_file)
        return None
    
    def list_emotions(self) -> list[str]:
        """List available emotions."""
        return list(self.manifest.emotion_images.keys())
    
    def list_animations(self) -> list[str]:
        """List available animations."""
        return list(self.manifest.animation_files.keys())
    
    def extract_to(self, directory: str) -> Path:
        """Extract bundle contents to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Write manifest
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest.to_dict(), indent=2))
        
        # Write files
        for path, content in self._files.items():
            file_path = directory / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)
        
        return directory
    
    def get_temp_directory(self) -> Path:
        """Extract to a temporary directory and return path."""
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="forgeavatar_"))
            self.extract_to(self._temp_dir)
        return self._temp_dir
    
    def cleanup_temp(self):
        """Clean up temporary directory if created."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup_temp()
        except Exception:
            pass  # Ignore cleanup errors during shutdown


class AvatarBundleCreator:
    """
    Helper class to create avatar bundles from various sources.
    """
    
    @staticmethod
    def from_single_image(
        image_path: str,
        name: str = "Custom Avatar",
        avatar_type: str = "HUMAN"
    ) -> AvatarBundle:
        """Create a simple bundle from a single image."""
        bundle = AvatarBundle()
        bundle.manifest.name = name
        bundle.manifest.avatar_type = avatar_type
        bundle.manifest.default_mode = "PNG_BOUNCE"
        bundle.set_base_image(image_path)
        return bundle
    
    @staticmethod
    def from_emotion_folder(
        folder_path: str,
        name: str = "Custom Avatar",
        avatar_type: str = "HUMAN"
    ) -> AvatarBundle:
        """
        Create a bundle from a folder of emotion images.
        
        Expected folder structure:
            folder/
            ├── base.png (or default.png or neutral.png)
            ├── happy.png
            ├── sad.png
            └── ...
        """
        folder = Path(folder_path)
        bundle = AvatarBundle()
        bundle.manifest.name = name
        bundle.manifest.avatar_type = avatar_type
        bundle.manifest.default_mode = "PNG_BOUNCE"
        
        # Find base image
        base_names = ["base.png", "default.png", "neutral.png", "idle.png"]
        for base_name in base_names:
            base_path = folder / base_name
            if base_path.exists():
                bundle.set_base_image(str(base_path))
                break
        
        # Standard emotion names to look for
        emotions = [
            "happy", "sad", "surprised", "thinking", "confused",
            "neutral", "angry", "excited", "curious", "worried",
            "proud", "embarrassed", "loving", "mischievous", "focused"
        ]
        
        for emotion in emotions:
            for ext in [".png", ".gif", ".jpg"]:
                emotion_path = folder / f"{emotion}{ext}"
                if emotion_path.exists():
                    bundle.add_emotion(emotion, str(emotion_path))
                    break
        
        return bundle
    
    @staticmethod
    def from_animation_folder(
        folder_path: str,
        name: str = "Animated Avatar",
        avatar_type: str = "HUMAN"
    ) -> AvatarBundle:
        """
        Create a bundle from a folder with animations.
        
        Expected folder structure:
            folder/
            ├── base.png
            ├── idle.gif
            ├── talking.gif
            ├── wave.gif
            └── ...
        """
        folder = Path(folder_path)
        bundle = AvatarBundle()
        bundle.manifest.name = name
        bundle.manifest.avatar_type = avatar_type
        bundle.manifest.default_mode = "ANIMATED_2D"
        
        # Find base image
        for ext in [".png", ".jpg"]:
            base_path = folder / f"base{ext}"
            if base_path.exists():
                bundle.set_base_image(str(base_path))
                break
        
        # Find animations
        for gif_path in folder.glob("*.gif"):
            anim_name = gif_path.stem
            bundle.add_animation(anim_name, str(gif_path))
        
        return bundle
    
    @staticmethod
    def from_3d_model(
        model_path: str,
        base_image_path: Optional[str] = None,
        name: str = "3D Avatar",
        avatar_type: str = "HUMAN"
    ) -> AvatarBundle:
        """Create a bundle from a 3D model (GLB/GLTF)."""
        bundle = AvatarBundle()
        bundle.manifest.name = name
        bundle.manifest.avatar_type = avatar_type
        bundle.manifest.default_mode = "SKELETAL_3D"
        bundle.set_model(model_path)
        
        if base_image_path:
            bundle.set_base_image(base_image_path)
        
        return bundle


def install_avatar_bundle(bundle_path: str, avatars_dir: Optional[str] = None) -> Path:
    """
    Install an avatar bundle to the avatars directory.
    
    Args:
        bundle_path: Path to .forgeavatar file
        avatars_dir: Target directory (default: data/avatar/installed/)
        
    Returns:
        Path to installed avatar directory
    """
    if avatars_dir is None:
        # Default installation directory
        from ..config import CONFIG
        avatars_dir = Path(CONFIG["data_dir"]) / "avatar" / "installed"
    
    avatars_dir = Path(avatars_dir)
    avatars_dir.mkdir(parents=True, exist_ok=True)
    
    # Load bundle
    bundle = AvatarBundle.load(bundle_path)
    
    # Create avatar directory
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in bundle.manifest.name)
    avatar_dir = avatars_dir / safe_name
    
    # Handle duplicates
    counter = 1
    original_dir = avatar_dir
    while avatar_dir.exists():
        avatar_dir = original_dir.parent / f"{original_dir.name}_{counter}"
        counter += 1
    
    # Extract
    bundle.extract_to(avatar_dir)
    
    return avatar_dir


def list_installed_avatars(avatars_dir: Optional[str] = None) -> list[dict[str, Any]]:
    """
    List all installed avatars.
    
    Returns:
        List of avatar info dicts with name, path, type, etc.
    """
    if avatars_dir is None:
        from ..config import CONFIG
        avatars_dir = Path(CONFIG["data_dir"]) / "avatar" / "installed"
    
    avatars_dir = Path(avatars_dir)
    avatars = []
    
    if not avatars_dir.exists():
        return avatars
    
    for avatar_path in avatars_dir.iterdir():
        if avatar_path.is_dir():
            manifest_path = avatar_path / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest_data = json.loads(manifest_path.read_text())
                    manifest = AvatarManifest.from_dict(manifest_data)
                    avatars.append({
                        "name": manifest.name,
                        "path": str(avatar_path),
                        "type": manifest.avatar_type,
                        "mode": manifest.default_mode,
                        "author": manifest.author,
                        "description": manifest.description,
                        "tags": manifest.tags,
                    })
                except Exception:
                    pass  # Intentionally silent
    
    return avatars
