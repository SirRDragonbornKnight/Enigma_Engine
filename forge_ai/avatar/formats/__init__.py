"""
Avatar Format Loaders

Support for various avatar formats:
- VRM: 3D humanoid models (VTuber standard)
- Live2D: 2D animated models (.moc3)
- Custom sprite sheets
"""

from .vrm_loader import VRMLoader, VRMModel, VRM_AVAILABLE
from .live2d_loader import Live2DLoader, Live2DModel, LIVE2D_AVAILABLE
from .sprite_sheet import SpriteSheetLoader, SpriteSheet

__all__ = [
    'VRMLoader',
    'VRMModel', 
    'VRM_AVAILABLE',
    'Live2DLoader',
    'Live2DModel',
    'LIVE2D_AVAILABLE',
    'SpriteSheetLoader',
    'SpriteSheet',
]
