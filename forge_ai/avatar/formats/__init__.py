"""
Avatar Format Loaders

Support for various avatar formats:
- VRM: 3D humanoid models (VTuber standard)
- Live2D: 2D animated models (.moc3)
- Custom sprite sheets
"""

from .live2d_loader import LIVE2D_AVAILABLE, Live2DLoader, Live2DModel
from .sprite_sheet import SpriteSheet, SpriteSheetLoader
from .vrm_loader import VRM_AVAILABLE, VRMLoader, VRMModel

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
