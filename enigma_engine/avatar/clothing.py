"""
Avatar Clothing System for Enigma AI Engine

Manage avatar clothing and accessories.

Features:
- Clothing inventory
- Outfit presets
- Layer management
- Color customization
- Fit adjustment

Usage:
    from enigma_engine.avatar.clothing import ClothingSystem
    
    clothing = ClothingSystem()
    
    # Add items
    clothing.add_item("tshirt_blue", category="top", color="#0066cc")
    clothing.add_item("jeans", category="bottom")
    
    # Create outfit
    outfit = clothing.create_outfit("casual", ["tshirt_blue", "jeans"])
    
    # Apply to avatar
    clothing.apply_outfit("casual")
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class ClothingCategory(Enum):
    """Clothing categories."""
    HEAD = "head"  # Hats, hair accessories
    FACE = "face"  # Glasses, masks
    TOP = "top"  # Shirts, jackets
    BOTTOM = "bottom"  # Pants, skirts
    FEET = "feet"  # Shoes, socks
    HANDS = "hands"  # Gloves
    FULL_BODY = "full_body"  # Dresses, suits
    ACCESSORY = "accessory"  # Jewelry, bags
    OVERLAY = "overlay"  # Capes, coats


class ClothingLayer(Enum):
    """Clothing layers for proper ordering."""
    SKIN = 0  # Base skin/body
    UNDERWEAR = 1
    BASE = 2  # T-shirts, pants
    MIDDLE = 3  # Sweaters, vests
    OUTER = 4  # Jackets, coats
    ACCESSORY = 5  # Jewelry, bags
    OVERLAY = 6  # Capes


@dataclass
class ColorPalette:
    """Color customization options."""
    primary: str = "#FFFFFF"  # Main color (hex)
    secondary: str = "#000000"  # Accent color
    tertiary: str = "#808080"  # Third color
    pattern: Optional[str] = None  # Pattern texture path
    metallic: float = 0.0  # Metallic property 0-1
    roughness: float = 0.5  # Surface roughness 0-1


@dataclass
class FitParameters:
    """Clothing fit parameters."""
    scale: float = 1.0  # Overall scale
    tightness: float = 0.5  # 0=loose, 1=tight
    length: float = 1.0  # Length multiplier
    
    # Per-axis adjustment
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    
    # Position offset
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0


@dataclass
class ClothingItem:
    """A clothing item."""
    id: str
    name: str
    category: ClothingCategory
    layer: ClothingLayer = ClothingLayer.BASE
    
    # Assets
    model_path: Optional[str] = None  # 3D model
    texture_path: Optional[str] = None  # Texture
    thumbnail_path: Optional[str] = None  # Preview image
    
    # Customization
    colors: ColorPalette = field(default_factory=ColorPalette)
    fit: FitParameters = field(default_factory=FitParameters)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    # Compatibility
    compatible_bodies: List[str] = field(default_factory=lambda: ["default"])
    conflicts_with: List[str] = field(default_factory=list)  # IDs of conflicting items
    requires: List[str] = field(default_factory=list)  # Required items
    
    # State
    visible: bool = True
    locked: bool = False  # Can't be removed


@dataclass
class Outfit:
    """A preset outfit."""
    id: str
    name: str
    items: List[str] = field(default_factory=list)  # Item IDs
    description: str = ""
    thumbnail_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class ClothingSystem:
    """Manages avatar clothing."""
    
    def __init__(
        self,
        wardrobe_dir: str = "data/avatar/wardrobe",
        outfits_file: str = "data/avatar/outfits.json"
    ):
        """
        Initialize clothing system.
        
        Args:
            wardrobe_dir: Directory for clothing assets
            outfits_file: File to save outfit presets
        """
        self.wardrobe_dir = Path(wardrobe_dir)
        self.outfits_file = Path(outfits_file)
        
        # Ensure directories exist
        self.wardrobe_dir.mkdir(parents=True, exist_ok=True)
        self.outfits_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Item storage
        self._items: Dict[str, ClothingItem] = {}
        self._outfits: Dict[str, Outfit] = {}
        
        # Currently worn items
        self._worn_items: Set[str] = set()
        
        # Callbacks
        self._on_change_callbacks: List[Callable] = []
        
        # Load saved data
        self._load_outfits()
        
        logger.info(f"ClothingSystem initialized with {len(self._items)} items")
    
    def add_item(
        self,
        item_id: str,
        name: Optional[str] = None,
        category: Union[str, ClothingCategory] = ClothingCategory.TOP,
        layer: Union[str, ClothingLayer] = ClothingLayer.BASE,
        model_path: Optional[str] = None,
        texture_path: Optional[str] = None,
        color: Optional[str] = None,
        **kwargs
    ) -> ClothingItem:
        """
        Add a clothing item to wardrobe.
        
        Args:
            item_id: Unique item identifier
            name: Display name
            category: Clothing category
            layer: Clothing layer
            model_path: Path to 3D model
            texture_path: Path to texture
            color: Primary color hex
            
        Returns:
            Created clothing item
        """
        # Convert string to enum
        if isinstance(category, str):
            category = ClothingCategory(category)
        if isinstance(layer, str):
            layer = ClothingLayer[layer.upper()]
        
        # Create colors
        colors = ColorPalette()
        if color:
            colors.primary = color
        
        item = ClothingItem(
            id=item_id,
            name=name or item_id,
            category=category,
            layer=layer,
            model_path=model_path,
            texture_path=texture_path,
            colors=colors
        )
        
        # Apply additional kwargs
        for key, value in kwargs.items():
            if hasattr(item, key):
                setattr(item, key, value)
        
        self._items[item_id] = item
        logger.info(f"Added clothing item: {item_id}")
        
        return item
    
    def remove_item(self, item_id: str) -> bool:
        """Remove item from wardrobe."""
        if item_id in self._items:
            # Remove from worn if wearing
            self._worn_items.discard(item_id)
            del self._items[item_id]
            return True
        return False
    
    def get_item(self, item_id: str) -> Optional[ClothingItem]:
        """Get clothing item by ID."""
        return self._items.get(item_id)
    
    def list_items(
        self,
        category: Optional[ClothingCategory] = None,
        tags: Optional[List[str]] = None
    ) -> List[ClothingItem]:
        """
        List clothing items.
        
        Args:
            category: Filter by category
            tags: Filter by tags
            
        Returns:
            List of matching items
        """
        items = list(self._items.values())
        
        if category:
            items = [i for i in items if i.category == category]
        
        if tags:
            items = [i for i in items if any(t in i.tags for t in tags)]
        
        return items
    
    def wear(self, item_id: str) -> bool:
        """
        Wear a clothing item.
        
        Args:
            item_id: Item to wear
            
        Returns:
            True if successful
        """
        item = self._items.get(item_id)
        if not item:
            logger.warning(f"Item not found: {item_id}")
            return False
        
        # Check conflicts
        for worn_id in list(self._worn_items):
            worn = self._items.get(worn_id)
            if not worn:
                continue
            
            # Same category conflict (except accessories)
            if worn.category == item.category and item.category != ClothingCategory.ACCESSORY:
                self._worn_items.discard(worn_id)
            
            # Explicit conflicts
            if item.id in worn.conflicts_with or worn.id in item.conflicts_with:
                self._worn_items.discard(worn_id)
        
        # Check requirements
        for req_id in item.requires:
            if req_id not in self._worn_items:
                logger.warning(f"Item {item_id} requires {req_id}")
                return False
        
        self._worn_items.add(item_id)
        self._notify_change()
        
        return True
    
    def remove(self, item_id: str) -> bool:
        """
        Remove a worn item.
        
        Args:
            item_id: Item to remove
            
        Returns:
            True if successful
        """
        item = self._items.get(item_id)
        if item and item.locked:
            logger.warning(f"Cannot remove locked item: {item_id}")
            return False
        
        if item_id in self._worn_items:
            self._worn_items.discard(item_id)
            self._notify_change()
            return True
        return False
    
    def get_worn_items(self) -> List[ClothingItem]:
        """Get currently worn items, sorted by layer."""
        items = [self._items[i] for i in self._worn_items if i in self._items]
        return sorted(items, key=lambda x: x.layer.value)
    
    def clear_all(self):
        """Remove all worn items (except locked)."""
        for item_id in list(self._worn_items):
            item = self._items.get(item_id)
            if item and not item.locked:
                self._worn_items.discard(item_id)
        self._notify_change()
    
    def create_outfit(
        self,
        outfit_id: str,
        items: List[str],
        name: Optional[str] = None,
        description: str = ""
    ) -> Outfit:
        """
        Create an outfit preset.
        
        Args:
            outfit_id: Unique outfit ID
            items: List of item IDs
            name: Display name
            description: Description
            
        Returns:
            Created outfit
        """
        outfit = Outfit(
            id=outfit_id,
            name=name or outfit_id,
            items=items,
            description=description
        )
        
        self._outfits[outfit_id] = outfit
        self._save_outfits()
        
        return outfit
    
    def save_current_outfit(
        self,
        outfit_id: str,
        name: Optional[str] = None
    ) -> Outfit:
        """Save current worn items as outfit."""
        return self.create_outfit(
            outfit_id,
            list(self._worn_items),
            name
        )
    
    def apply_outfit(self, outfit_id: str) -> bool:
        """
        Apply an outfit preset.
        
        Args:
            outfit_id: Outfit to apply
            
        Returns:
            True if successful
        """
        outfit = self._outfits.get(outfit_id)
        if not outfit:
            logger.warning(f"Outfit not found: {outfit_id}")
            return False
        
        # Clear current (except locked)
        self.clear_all()
        
        # Wear outfit items
        for item_id in outfit.items:
            self.wear(item_id)
        
        return True
    
    def delete_outfit(self, outfit_id: str) -> bool:
        """Delete an outfit preset."""
        if outfit_id in self._outfits:
            del self._outfits[outfit_id]
            self._save_outfits()
            return True
        return False
    
    def list_outfits(self, tags: Optional[List[str]] = None) -> List[Outfit]:
        """List outfit presets."""
        outfits = list(self._outfits.values())
        
        if tags:
            outfits = [o for o in outfits if any(t in o.tags for t in tags)]
        
        return outfits
    
    def customize_item(
        self,
        item_id: str,
        primary_color: Optional[str] = None,
        secondary_color: Optional[str] = None,
        fit_scale: Optional[float] = None,
        fit_tightness: Optional[float] = None
    ) -> Optional[ClothingItem]:
        """
        Customize a clothing item.
        
        Args:
            item_id: Item to customize
            primary_color: New primary color
            secondary_color: New secondary color
            fit_scale: Fit scale
            fit_tightness: Fit tightness
            
        Returns:
            Updated item or None
        """
        item = self._items.get(item_id)
        if not item:
            return None
        
        if primary_color:
            item.colors.primary = primary_color
        if secondary_color:
            item.colors.secondary = secondary_color
        if fit_scale is not None:
            item.fit.scale = fit_scale
        if fit_tightness is not None:
            item.fit.tightness = fit_tightness
        
        self._notify_change()
        
        return item
    
    def get_layer_order(self) -> List[Tuple[ClothingLayer, List[ClothingItem]]]:
        """Get worn items organized by layer."""
        worn = self.get_worn_items()
        
        layers = {}
        for item in worn:
            if item.layer not in layers:
                layers[item.layer] = []
            layers[item.layer].append(item)
        
        return sorted(layers.items(), key=lambda x: x[0].value)
    
    def on_change(self, callback: Callable):
        """Register callback for clothing changes."""
        self._on_change_callbacks.append(callback)
    
    def _notify_change(self):
        """Notify listeners of changes."""
        for callback in self._on_change_callbacks:
            try:
                callback(self.get_worn_items())
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _load_outfits(self):
        """Load outfits from file."""
        if self.outfits_file.exists():
            try:
                data = json.loads(self.outfits_file.read_text())
                for outfit_data in data.get("outfits", []):
                    outfit = Outfit(
                        id=outfit_data.get("id", ""),
                        name=outfit_data.get("name", ""),
                        items=outfit_data.get("items", []),
                        description=outfit_data.get("description", ""),
                        tags=outfit_data.get("tags", [])
                    )
                    self._outfits[outfit.id] = outfit
            except Exception as e:
                logger.error(f"Failed to load outfits: {e}")
    
    def _save_outfits(self):
        """Save outfits to file."""
        data = {
            "outfits": [
                {
                    "id": o.id,
                    "name": o.name,
                    "items": o.items,
                    "description": o.description,
                    "tags": o.tags
                }
                for o in self._outfits.values()
            ]
        }
        
        self.outfits_file.write_text(json.dumps(data, indent=2))
    
    def export_wardrobe(self, output_path: str) -> str:
        """Export wardrobe to file."""
        data = {
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "category": item.category.value,
                    "layer": item.layer.name,
                    "tags": item.tags,
                    "colors": {
                        "primary": item.colors.primary,
                        "secondary": item.colors.secondary
                    }
                }
                for item in self._items.values()
            ],
            "outfits": [
                {
                    "id": o.id,
                    "name": o.name,
                    "items": o.items
                }
                for o in self._outfits.values()
            ]
        }
        
        Path(output_path).write_text(json.dumps(data, indent=2))
        return output_path
    
    def import_wardrobe(self, input_path: str):
        """Import wardrobe from file."""
        data = json.loads(Path(input_path).read_text())
        
        for item_data in data.get("items", []):
            self.add_item(
                item_id=item_data.get("id"),
                name=item_data.get("name"),
                category=item_data.get("category", "top"),
                tags=item_data.get("tags", [])
            )
        
        for outfit_data in data.get("outfits", []):
            self.create_outfit(
                outfit_id=outfit_data.get("id"),
                items=outfit_data.get("items", []),
                name=outfit_data.get("name")
            )


# Global instance
_clothing_system: Optional[ClothingSystem] = None


def get_clothing_system() -> ClothingSystem:
    """Get or create global ClothingSystem instance."""
    global _clothing_system
    if _clothing_system is None:
        _clothing_system = ClothingSystem()
    return _clothing_system
