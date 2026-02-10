"""
Avatar Outfit System

Changeable outfits, accessories, and clothing customization.
Supports color customization and prop items.

FILE: enigma_engine/avatar/outfit_system.py
TYPE: Avatar
MAIN CLASSES: OutfitManager, Outfit, ClothingItem, Accessory
"""

import colorsys
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ClothingSlot(Enum):
    """Body slots for clothing items."""
    HEAD = "head"
    FACE = "face"
    NECK = "neck"
    TORSO = "torso"
    ARMS = "arms"
    HANDS = "hands"
    LEGS = "legs"
    FEET = "feet"
    FULL_BODY = "full_body"


class AccessorySlot(Enum):
    """Slots for accessories."""
    HAT = "hat"
    GLASSES = "glasses"
    EARRINGS = "earrings"
    NECKLACE = "necklace"
    BRACELET = "bracelet"
    RING = "ring"
    WATCH = "watch"
    BELT = "belt"
    BAG = "bag"
    WINGS = "wings"
    TAIL = "tail"


class PropSlot(Enum):
    """Slots for holdable props."""
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    BACK = "back"
    HIP = "hip"
    SHOULDER = "shoulder"


@dataclass
class Color:
    """Color with RGB and optional alpha."""
    r: float  # 0-1
    g: float  # 0-1
    b: float  # 0-1
    a: float = 1.0
    
    @classmethod
    def from_hex(cls, hex_str: str) -> 'Color':
        """Create from hex string."""
        hex_str = hex_str.lstrip('#')
        if len(hex_str) == 6:
            r, g, b = tuple(int(hex_str[i:i+2], 16) / 255 for i in (0, 2, 4))
            return cls(r, g, b)
        elif len(hex_str) == 8:
            r, g, b, a = tuple(int(hex_str[i:i+2], 16) / 255 for i in (0, 2, 4, 6))
            return cls(r, g, b, a)
        raise ValueError(f"Invalid hex color: {hex_str}")
    
    def to_hex(self) -> str:
        """Convert to hex string."""
        return f"#{int(self.r*255):02x}{int(self.g*255):02x}{int(self.b*255):02x}"
    
    def lighten(self, amount: float) -> 'Color':
        """Lighten color by amount (0-1)."""
        h, l, s = colorsys.rgb_to_hls(self.r, self.g, self.b)
        l = min(1.0, l + amount)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return Color(r, g, b, self.a)
    
    def darken(self, amount: float) -> 'Color':
        """Darken color by amount (0-1)."""
        h, l, s = colorsys.rgb_to_hls(self.r, self.g, self.b)
        l = max(0.0, l - amount)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return Color(r, g, b, self.a)
    
    def saturate(self, amount: float) -> 'Color':
        """Increase saturation by amount (0-1)."""
        h, l, s = colorsys.rgb_to_hls(self.r, self.g, self.b)
        s = min(1.0, s + amount)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return Color(r, g, b, self.a)


@dataclass
class ColorZone:
    """A colorizable zone on an item."""
    name: str
    default_color: Color
    current_color: Color = None
    
    def __post_init__(self):
        if self.current_color is None:
            self.current_color = self.default_color


@dataclass
class ClothingItem:
    """A wearable clothing item."""
    id: str
    name: str
    slot: ClothingSlot
    model_path: str
    thumbnail_path: str = ""
    
    # Customization
    color_zones: list[ColorZone] = field(default_factory=list)
    
    # Constraints
    blocks_slots: list[ClothingSlot] = field(default_factory=list)
    requires_slots: list[ClothingSlot] = field(default_factory=list)
    
    # Physics
    has_physics: bool = False
    physics_bones: list[str] = field(default_factory=list)
    
    # Metadata
    tags: list[str] = field(default_factory=list)
    
    def set_color(self, zone_name: str, color: Color):
        """Set color for a zone."""
        for zone in self.color_zones:
            if zone.name == zone_name:
                zone.current_color = color
                return
        raise ValueError(f"Zone not found: {zone_name}")
    
    def reset_colors(self):
        """Reset all colors to defaults."""
        for zone in self.color_zones:
            zone.current_color = zone.default_color


@dataclass
class Accessory:
    """An accessory item."""
    id: str
    name: str
    slot: AccessorySlot
    model_path: str
    thumbnail_path: str = ""
    
    # Positioning
    position_offset: tuple[float, float, float] = (0, 0, 0)
    rotation_offset: tuple[float, float, float] = (0, 0, 0)
    scale: float = 1.0
    
    # Customization
    color_zones: list[ColorZone] = field(default_factory=list)
    
    # Attach point
    attach_bone: str = ""
    
    tags: list[str] = field(default_factory=list)


@dataclass
class Prop:
    """A holdable prop item."""
    id: str
    name: str
    slot: PropSlot
    model_path: str
    thumbnail_path: str = ""
    
    # Animation
    idle_animation: str = ""  # Animation to play when holding
    use_animation: str = ""  # Animation when using
    
    # Attach
    attach_bone: str = ""
    grip_offset: tuple[float, float, float] = (0, 0, 0)
    grip_rotation: tuple[float, float, float] = (0, 0, 0)
    
    tags: list[str] = field(default_factory=list)


@dataclass
class Outfit:
    """A complete outfit configuration."""
    id: str
    name: str
    description: str = ""
    thumbnail_path: str = ""
    
    # Items
    clothing: dict[ClothingSlot, ClothingItem] = field(default_factory=dict)
    accessories: dict[AccessorySlot, Accessory] = field(default_factory=dict)
    props: dict[PropSlot, Prop] = field(default_factory=dict)
    
    # Avatar customization
    hair_color: Optional[Color] = None
    eye_color: Optional[Color] = None
    skin_color: Optional[Color] = None
    
    tags: list[str] = field(default_factory=list)
    
    def add_clothing(self, item: ClothingItem):
        """Add clothing item to outfit."""
        # Check for blocked slots
        for slot in item.blocks_slots:
            if slot in self.clothing:
                logger.warning(f"Removing {self.clothing[slot].name} (blocked by {item.name})")
                del self.clothing[slot]
        
        self.clothing[item.slot] = item
    
    def remove_clothing(self, slot: ClothingSlot):
        """Remove clothing from slot."""
        if slot in self.clothing:
            del self.clothing[slot]
    
    def add_accessory(self, accessory: Accessory):
        """Add accessory to outfit."""
        self.accessories[accessory.slot] = accessory
    
    def remove_accessory(self, slot: AccessorySlot):
        """Remove accessory from slot."""
        if slot in self.accessories:
            del self.accessories[slot]
    
    def add_prop(self, prop: Prop):
        """Add prop to outfit."""
        self.props[prop.slot] = prop
    
    def remove_prop(self, slot: PropSlot):
        """Remove prop from slot."""
        if slot in self.props:
            del self.props[slot]
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize outfit to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "thumbnail_path": self.thumbnail_path,
            "clothing": {
                slot.value: {
                    "id": item.id,
                    "colors": {
                        zone.name: zone.current_color.to_hex()
                        for zone in item.color_zones
                    }
                }
                for slot, item in self.clothing.items()
            },
            "accessories": {
                slot.value: {"id": acc.id}
                for slot, acc in self.accessories.items()
            },
            "props": {
                slot.value: {"id": prop.id}
                for slot, prop in self.props.items()
            },
            "hair_color": self.hair_color.to_hex() if self.hair_color else None,
            "eye_color": self.eye_color.to_hex() if self.eye_color else None,
            "skin_color": self.skin_color.to_hex() if self.skin_color else None,
            "tags": self.tags
        }


class OutfitManager:
    """
    Manages avatar outfits and customization.
    
    Handles loading, saving, and applying outfits.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize outfit manager.
        
        Args:
            data_dir: Directory for outfit data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._clothing_library: dict[str, ClothingItem] = {}
        self._accessory_library: dict[str, Accessory] = {}
        self._prop_library: dict[str, Prop] = {}
        self._outfits: dict[str, Outfit] = {}
        
        self._current_outfit: Optional[Outfit] = None
        
        self._load_library()
        self._load_outfits()
    
    def _load_library(self):
        """Load clothing/accessory library from disk."""
        library_file = self.data_dir / "library.json"
        if not library_file.exists():
            self._create_default_library()
            return
        
        with open(library_file) as f:
            data = json.load(f)
        
        for item_data in data.get("clothing", []):
            item = self._parse_clothing(item_data)
            self._clothing_library[item.id] = item
        
        for item_data in data.get("accessories", []):
            item = self._parse_accessory(item_data)
            self._accessory_library[item.id] = item
        
        for item_data in data.get("props", []):
            item = self._parse_prop(item_data)
            self._prop_library[item.id] = item
        
        logger.info(f"Loaded {len(self._clothing_library)} clothing, "
                    f"{len(self._accessory_library)} accessories, "
                    f"{len(self._prop_library)} props")
    
    def _create_default_library(self):
        """Create default item library."""
        # Default clothing
        self._clothing_library["casual_shirt"] = ClothingItem(
            id="casual_shirt",
            name="Casual Shirt",
            slot=ClothingSlot.TORSO,
            model_path="models/clothing/casual_shirt.glb",
            color_zones=[
                ColorZone("main", Color.from_hex("#ffffff")),
                ColorZone("buttons", Color.from_hex("#333333"))
            ],
            tags=["casual", "basic"]
        )
        
        self._clothing_library["jeans"] = ClothingItem(
            id="jeans",
            name="Blue Jeans",
            slot=ClothingSlot.LEGS,
            model_path="models/clothing/jeans.glb",
            color_zones=[
                ColorZone("denim", Color.from_hex("#4a6fa5"))
            ],
            tags=["casual", "basic"]
        )
        
        self._clothing_library["sneakers"] = ClothingItem(
            id="sneakers",
            name="Sneakers",
            slot=ClothingSlot.FEET,
            model_path="models/clothing/sneakers.glb",
            color_zones=[
                ColorZone("main", Color.from_hex("#ffffff")),
                ColorZone("sole", Color.from_hex("#333333")),
                ColorZone("laces", Color.from_hex("#ffffff"))
            ],
            tags=["casual", "shoes"]
        )
        
        # Default accessories
        self._accessory_library["round_glasses"] = Accessory(
            id="round_glasses",
            name="Round Glasses",
            slot=AccessorySlot.GLASSES,
            model_path="models/accessories/round_glasses.glb",
            attach_bone="head",
            position_offset=(0, 0.05, 0.08),
            color_zones=[
                ColorZone("frame", Color.from_hex("#333333")),
                ColorZone("lens", Color(0.2, 0.2, 0.2, 0.5))
            ],
            tags=["glasses", "casual"]
        )
        
        self._accessory_library["watch"] = Accessory(
            id="watch",
            name="Wristwatch",
            slot=AccessorySlot.WATCH,
            model_path="models/accessories/watch.glb",
            attach_bone="left_wrist",
            color_zones=[
                ColorZone("band", Color.from_hex("#8b4513")),
                ColorZone("face", Color.from_hex("#silver"))
            ],
            tags=["watch", "formal"]
        )
        
        # Default props
        self._prop_library["coffee_cup"] = Prop(
            id="coffee_cup",
            name="Coffee Cup",
            slot=PropSlot.RIGHT_HAND,
            model_path="models/props/coffee_cup.glb",
            attach_bone="right_hand",
            idle_animation="hold_cup",
            use_animation="drink",
            tags=["drink", "casual"]
        )
        
        self._prop_library["book"] = Prop(
            id="book",
            name="Book",
            slot=PropSlot.LEFT_HAND,
            model_path="models/props/book.glb",
            attach_bone="left_hand",
            idle_animation="hold_book",
            use_animation="read_book",
            tags=["reading", "education"]
        )
        
        self._save_library()
    
    def _save_library(self):
        """Save library to disk."""
        data = {
            "clothing": [self._serialize_clothing(c) for c in self._clothing_library.values()],
            "accessories": [self._serialize_accessory(a) for a in self._accessory_library.values()],
            "props": [self._serialize_prop(p) for p in self._prop_library.values()]
        }
        
        with open(self.data_dir / "library.json", 'w') as f:
            json.dump(data, f, indent=2)
    
    def _parse_clothing(self, data: dict) -> ClothingItem:
        """Parse clothing from dict."""
        return ClothingItem(
            id=data["id"],
            name=data["name"],
            slot=ClothingSlot(data["slot"]),
            model_path=data["model_path"],
            thumbnail_path=data.get("thumbnail_path", ""),
            color_zones=[
                ColorZone(z["name"], Color.from_hex(z["color"]))
                for z in data.get("color_zones", [])
            ],
            blocks_slots=[ClothingSlot(s) for s in data.get("blocks_slots", [])],
            tags=data.get("tags", [])
        )
    
    def _serialize_clothing(self, item: ClothingItem) -> dict:
        """Serialize clothing to dict."""
        return {
            "id": item.id,
            "name": item.name,
            "slot": item.slot.value,
            "model_path": item.model_path,
            "thumbnail_path": item.thumbnail_path,
            "color_zones": [
                {"name": z.name, "color": z.default_color.to_hex()}
                for z in item.color_zones
            ],
            "blocks_slots": [s.value for s in item.blocks_slots],
            "tags": item.tags
        }
    
    def _parse_accessory(self, data: dict) -> Accessory:
        """Parse accessory from dict."""
        return Accessory(
            id=data["id"],
            name=data["name"],
            slot=AccessorySlot(data["slot"]),
            model_path=data["model_path"],
            thumbnail_path=data.get("thumbnail_path", ""),
            attach_bone=data.get("attach_bone", ""),
            position_offset=tuple(data.get("position_offset", [0, 0, 0])),
            rotation_offset=tuple(data.get("rotation_offset", [0, 0, 0])),
            scale=data.get("scale", 1.0),
            tags=data.get("tags", [])
        )
    
    def _serialize_accessory(self, acc: Accessory) -> dict:
        """Serialize accessory to dict."""
        return {
            "id": acc.id,
            "name": acc.name,
            "slot": acc.slot.value,
            "model_path": acc.model_path,
            "thumbnail_path": acc.thumbnail_path,
            "attach_bone": acc.attach_bone,
            "position_offset": list(acc.position_offset),
            "rotation_offset": list(acc.rotation_offset),
            "scale": acc.scale,
            "tags": acc.tags
        }
    
    def _parse_prop(self, data: dict) -> Prop:
        """Parse prop from dict."""
        return Prop(
            id=data["id"],
            name=data["name"],
            slot=PropSlot(data["slot"]),
            model_path=data["model_path"],
            thumbnail_path=data.get("thumbnail_path", ""),
            attach_bone=data.get("attach_bone", ""),
            idle_animation=data.get("idle_animation", ""),
            use_animation=data.get("use_animation", ""),
            tags=data.get("tags", [])
        )
    
    def _serialize_prop(self, prop: Prop) -> dict:
        """Serialize prop to dict."""
        return {
            "id": prop.id,
            "name": prop.name,
            "slot": prop.slot.value,
            "model_path": prop.model_path,
            "thumbnail_path": prop.thumbnail_path,
            "attach_bone": prop.attach_bone,
            "idle_animation": prop.idle_animation,
            "use_animation": prop.use_animation,
            "tags": prop.tags
        }
    
    def _load_outfits(self):
        """Load saved outfits."""
        outfits_file = self.data_dir / "outfits.json"
        if not outfits_file.exists():
            return
        
        with open(outfits_file) as f:
            data = json.load(f)
        
        for outfit_data in data.get("outfits", []):
            outfit = self._parse_outfit(outfit_data)
            self._outfits[outfit.id] = outfit
    
    def _parse_outfit(self, data: dict) -> Outfit:
        """Parse outfit from dict."""
        outfit = Outfit(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            thumbnail_path=data.get("thumbnail_path", ""),
            tags=data.get("tags", [])
        )
        
        # Load clothing
        for slot_str, item_data in data.get("clothing", {}).items():
            slot = ClothingSlot(slot_str)
            item_id = item_data["id"]
            if item_id in self._clothing_library:
                item = self._clothing_library[item_id]
                # Apply colors
                for zone_name, color_hex in item_data.get("colors", {}).items():
                    item.set_color(zone_name, Color.from_hex(color_hex))
                outfit.clothing[slot] = item
        
        # Load accessories
        for slot_str, item_data in data.get("accessories", {}).items():
            slot = AccessorySlot(slot_str)
            item_id = item_data["id"]
            if item_id in self._accessory_library:
                outfit.accessories[slot] = self._accessory_library[item_id]
        
        # Load props
        for slot_str, item_data in data.get("props", {}).items():
            slot = PropSlot(slot_str)
            item_id = item_data["id"]
            if item_id in self._prop_library:
                outfit.props[slot] = self._prop_library[item_id]
        
        # Avatar colors
        if data.get("hair_color"):
            outfit.hair_color = Color.from_hex(data["hair_color"])
        if data.get("eye_color"):
            outfit.eye_color = Color.from_hex(data["eye_color"])
        if data.get("skin_color"):
            outfit.skin_color = Color.from_hex(data["skin_color"])
        
        return outfit
    
    def save_outfits(self):
        """Save all outfits to disk."""
        data = {
            "outfits": [o.to_dict() for o in self._outfits.values()]
        }
        with open(self.data_dir / "outfits.json", 'w') as f:
            json.dump(data, f, indent=2)
    
    # Public API
    
    def get_clothing(self, category: str = None) -> list[ClothingItem]:
        """Get available clothing items."""
        items = list(self._clothing_library.values())
        if category:
            items = [i for i in items if category in i.tags]
        return items
    
    def get_accessories(self, category: str = None) -> list[Accessory]:
        """Get available accessories."""
        items = list(self._accessory_library.values())
        if category:
            items = [i for i in items if category in i.tags]
        return items
    
    def get_props(self, category: str = None) -> list[Prop]:
        """Get available props."""
        items = list(self._prop_library.values())
        if category:
            items = [i for i in items if category in i.tags]
        return items
    
    def get_outfits(self) -> list[Outfit]:
        """Get saved outfits."""
        return list(self._outfits.values())
    
    def create_outfit(self, name: str) -> Outfit:
        """Create a new outfit."""
        outfit_id = name.lower().replace(" ", "_")
        outfit = Outfit(id=outfit_id, name=name)
        self._outfits[outfit_id] = outfit
        self.save_outfits()
        return outfit
    
    def delete_outfit(self, outfit_id: str):
        """Delete an outfit."""
        if outfit_id in self._outfits:
            del self._outfits[outfit_id]
            self.save_outfits()
    
    def apply_outfit(self, outfit: Outfit):
        """Apply an outfit to the avatar."""
        self._current_outfit = outfit
        logger.info(f"Applied outfit: {outfit.name}")
        # In real implementation, this would update the 3D model
    
    def get_current_outfit(self) -> Optional[Outfit]:
        """Get currently applied outfit."""
        return self._current_outfit


# Convenience function
def get_outfit_manager(data_dir: str = None) -> OutfitManager:
    """Get or create outfit manager."""
    if data_dir is None:
        data_dir = Path.home() / ".enigma_engine" / "outfits"
    return OutfitManager(Path(data_dir))


__all__ = [
    'OutfitManager',
    'Outfit',
    'ClothingItem',
    'Accessory',
    'Prop',
    'Color',
    'ColorZone',
    'ClothingSlot',
    'AccessorySlot',
    'PropSlot',
    'get_outfit_manager'
]
