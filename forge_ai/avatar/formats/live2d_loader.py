"""
Live2D Model Loader

Live2D is a 2D animation technology that creates animated 
2D characters from layered images. Popular for VTubers.

Live2D models consist of:
- .moc3: Compiled model data
- .model3.json: Model definition
- Texture images (PNG)
- Motion files (.motion3.json)
- Expression files (.exp3.json)

NOTE: Live2D Cubism SDK has licensing requirements for commercial use.
This loader provides basic model info extraction and integration stubs.
Full rendering requires the official Live2D SDK or compatible library.

Optional dependencies:
- Pillow: Texture processing
- json: Model file parsing
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Check for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Live2D availability (basic parsing always works, rendering needs SDK)
LIVE2D_AVAILABLE = True  # Basic model info extraction


@dataclass
class Live2DExpression:
    """A Live2D expression definition."""
    name: str
    file_path: str = ""
    fade_in_time: float = 1.0
    fade_out_time: float = 1.0


@dataclass
class Live2DMotion:
    """A Live2D motion (animation)."""
    name: str
    file_path: str = ""
    group: str = ""  # idle, tap, etc.
    fade_in_time: float = 1.0
    fade_out_time: float = 1.0


@dataclass
class Live2DModel:
    """
    A Live2D model definition.
    
    Contains metadata and references to model files,
    but not the actual rendering data (requires Live2D SDK).
    """
    
    # Metadata
    name: str = ""
    version: int = 3  # Cubism version (3 or 4)
    
    # File references
    moc_file: str = ""
    texture_files: List[str] = field(default_factory=list)
    physics_file: str = ""
    pose_file: str = ""
    user_data_file: str = ""
    
    # Expressions and motions
    expressions: Dict[str, Live2DExpression] = field(default_factory=dict)
    motions: Dict[str, List[Live2DMotion]] = field(default_factory=dict)
    
    # Model directory
    base_path: str = ""
    
    @property
    def has_expressions(self) -> bool:
        """Check if model has expressions."""
        return len(self.expressions) > 0
    
    @property
    def has_motions(self) -> bool:
        """Check if model has motions."""
        return any(len(m) > 0 for m in self.motions.values())
    
    def get_expression_names(self) -> List[str]:
        """Get available expression names."""
        return list(self.expressions.keys())
    
    def get_motion_groups(self) -> List[str]:
        """Get motion group names."""
        return list(self.motions.keys())
    
    def get_texture_paths(self) -> List[Path]:
        """Get full paths to texture files."""
        base = Path(self.base_path)
        return [base / tex for tex in self.texture_files]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "moc_file": self.moc_file,
            "texture_count": len(self.texture_files),
            "expression_count": len(self.expressions),
            "motion_groups": self.get_motion_groups(),
            "base_path": self.base_path,
        }


class Live2DLoader:
    """
    Loader for Live2D Cubism models.
    
    This provides basic model info extraction and file parsing.
    Full rendering requires the Live2D Cubism SDK.
    
    Usage:
        loader = Live2DLoader()
        model = loader.load("model/model.model3.json")
        if model:
            print(f"Model: {model.name}")
            print(f"Expressions: {model.get_expression_names()}")
    """
    
    def __init__(self):
        """Initialize Live2D loader."""
        self._cache: Dict[str, Live2DModel] = {}
    
    @staticmethod
    def is_available() -> bool:
        """Check if loader is available (basic always is)."""
        return LIVE2D_AVAILABLE
    
    @staticmethod
    def get_requirements() -> List[str]:
        """Get recommended packages."""
        return ["Pillow (for textures)", "Live2D Cubism SDK (for rendering)"]
    
    def load(self, filepath: str) -> Optional[Live2DModel]:
        """
        Load a Live2D model definition.
        
        Args:
            filepath: Path to .model3.json file
            
        Returns:
            Live2DModel or None
        """
        path = Path(filepath)
        
        # Handle different input types
        if path.suffix == '.model3':
            # Direct .model3.json reference
            model_json_path = path.with_suffix('.model3.json')
        elif path.suffix == '.json' and 'model3' in path.stem:
            model_json_path = path
        elif path.is_dir():
            # Look for model3.json in directory
            candidates = list(path.glob("*.model3.json"))
            if not candidates:
                print(f"[Live2DLoader] No model3.json found in {path}")
                return None
            model_json_path = candidates[0]
        else:
            print(f"[Live2DLoader] Unsupported input: {filepath}")
            return None
        
        if not model_json_path.exists():
            print(f"[Live2DLoader] File not found: {model_json_path}")
            return None
        
        # Check cache
        cache_key = str(model_json_path.absolute())
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            return self._load_model(model_json_path)
        except Exception as e:
            print(f"[Live2DLoader] Error loading {filepath}: {e}")
            return None
    
    def _load_model(self, model_json_path: Path) -> Optional[Live2DModel]:
        """Internal model loading."""
        with open(model_json_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        model = Live2DModel()
        model.base_path = str(model_json_path.parent)
        model.name = model_json_path.stem.replace('.model3', '')
        
        # Parse version
        model.version = model_data.get('Version', 3)
        
        # File references
        file_refs = model_data.get('FileReferences', {})
        
        # MOC file
        model.moc_file = file_refs.get('Moc', '')
        
        # Textures
        model.texture_files = file_refs.get('Textures', [])
        
        # Physics
        model.physics_file = file_refs.get('Physics', '')
        
        # Pose
        model.pose_file = file_refs.get('Pose', '')
        
        # User data
        model.user_data_file = file_refs.get('UserData', '')
        
        # Expressions
        for expr in file_refs.get('Expressions', []):
            name = expr.get('Name', '')
            if name:
                model.expressions[name] = Live2DExpression(
                    name=name,
                    file_path=expr.get('File', ''),
                    fade_in_time=expr.get('FadeInTime', 1.0),
                    fade_out_time=expr.get('FadeOutTime', 1.0)
                )
        
        # Motions
        for group_name, motions in file_refs.get('Motions', {}).items():
            motion_list = []
            for i, motion in enumerate(motions):
                motion_list.append(Live2DMotion(
                    name=f"{group_name}_{i}",
                    file_path=motion.get('File', ''),
                    group=group_name,
                    fade_in_time=motion.get('FadeInTime', 1.0),
                    fade_out_time=motion.get('FadeOutTime', 1.0)
                ))
            model.motions[group_name] = motion_list
        
        # Cache and return
        self._cache[str(model_json_path.absolute())] = model
        return model
    
    def get_model_preview(self, model: Live2DModel) -> Optional[Any]:
        """
        Get a preview image for the model (first texture).
        
        Args:
            model: Live2DModel to preview
            
        Returns:
            PIL Image or None
        """
        if not PIL_AVAILABLE:
            print("[Live2DLoader] Pillow not installed for preview")
            return None
        
        textures = model.get_texture_paths()
        if not textures:
            return None
        
        # Load first texture as preview
        tex_path = textures[0]
        if tex_path.exists():
            from PIL import Image as PILImage
            return PILImage.open(tex_path)
        
        return None
    
    def clear_cache(self):
        """Clear model cache."""
        self._cache.clear()
    
    def get_cached_models(self) -> List[str]:
        """Get cached model paths."""
        return list(self._cache.keys())


def get_live2d_loader() -> Live2DLoader:
    """Get or create Live2D loader instance."""
    return Live2DLoader()


# Common Live2D expression presets
LIVE2D_EXPRESSION_PRESETS = {
    "default": "neutral expression",
    "angry": "angry/frustrated expression",
    "happy": "happy/smiling expression",  
    "sad": "sad/disappointed expression",
    "surprised": "surprised expression",
    "wink_l": "left eye wink",
    "wink_r": "right eye wink",
    "thinking": "thinking/pondering expression",
}
