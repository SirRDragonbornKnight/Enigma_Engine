"""
Avatar Customizer

User tools for customizing avatar appearance.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

from .avatar_identity import AvatarAppearance


class AvatarCustomizer:
    """
    User tools for customizing avatar appearance.
    
    Allows users to:
    - Change avatar style
    - Customize colors
    - Add/remove accessories
    - Import/export appearances
    - Preview changes
    """
    
    # Available options
    STYLES = ["default", "anime", "realistic", "robot", "abstract", "minimal"]
    SHAPES = ["rounded", "angular", "mixed"]
    SIZES = ["small", "medium", "large"]
    EYE_STYLES = ["normal", "cute", "sharp", "closed"]
    ANIMATIONS = {
        "idle": ["breathe", "float", "pulse", "still", "bounce"],
        "movement": ["float", "walk", "bounce", "teleport"],
    }
    ACCESSORIES = [
        "hat", "glasses", "tie", "bow", "scarf",
        "headphones", "crown", "halo", "horns",
        "creative_element", "bold_outline", "sparkles"
    ]
    
    # Color presets
    COLOR_PRESETS = {
        "default": {
            "primary": "#6366f1",
            "secondary": "#8b5cf6",
            "accent": "#10b981"
        },
        "warm": {
            "primary": "#f59e0b",
            "secondary": "#ef4444",
            "accent": "#fbbf24"
        },
        "cool": {
            "primary": "#3b82f6",
            "secondary": "#06b6d4",
            "accent": "#8b5cf6"
        },
        "nature": {
            "primary": "#10b981",
            "secondary": "#22c55e",
            "accent": "#84cc16"
        },
        "sunset": {
            "primary": "#f59e0b",
            "secondary": "#ec4899",
            "accent": "#8b5cf6"
        },
        "ocean": {
            "primary": "#06b6d4",
            "secondary": "#0ea5e9",
            "accent": "#3b82f6"
        },
        "fire": {
            "primary": "#ef4444",
            "secondary": "#f59e0b",
            "accent": "#fbbf24"
        },
        "dark": {
            "primary": "#1e293b",
            "secondary": "#475569",
            "accent": "#64748b"
        },
        "pastel": {
            "primary": "#a78bfa",
            "secondary": "#f0abfc",
            "accent": "#fbcfe8"
        },
    }
    
    def __init__(self, avatar):
        """
        Initialize customizer.
        
        Args:
            avatar: AvatarController instance
        """
        self.avatar = avatar
        self._preview_appearance: Optional[AvatarAppearance] = None
        self._original_appearance: Optional[AvatarAppearance] = None
    
    def set_style(self, style: str) -> bool:
        """
        Change avatar style.
        
        Args:
            style: Style name (default, anime, realistic, robot, abstract, minimal)
            
        Returns:
            True if successful
        """
        if style not in self.STYLES:
            print(f"[Customizer] Invalid style: {style}. Available: {self.STYLES}")
            return False
        
        if hasattr(self.avatar, '_identity') and self.avatar._identity:
            self.avatar._identity.appearance.style = style
            
            # Update renderer if avatar is enabled
            if self.avatar.is_enabled and self.avatar._renderer:
                self.avatar._renderer.set_appearance(self.avatar._identity.appearance)
            
            print(f"[Customizer] Style changed to: {style}")
            return True
        
        return False
    
    def set_colors(
        self,
        primary: Optional[str] = None,
        secondary: Optional[str] = None,
        accent: Optional[str] = None
    ) -> bool:
        """
        Set avatar color scheme.
        
        Args:
            primary: Primary color (hex)
            secondary: Secondary color (hex)
            accent: Accent color (hex)
            
        Returns:
            True if successful
        """
        if not hasattr(self.avatar, '_identity') or not self.avatar._identity:
            return False
        
        appearance = self.avatar._identity.appearance
        
        if primary:
            appearance.primary_color = primary
        if secondary:
            appearance.secondary_color = secondary
        if accent:
            appearance.accent_color = accent
        
        # Update renderer
        if self.avatar.is_enabled and self.avatar._renderer:
            self.avatar._renderer.set_appearance(appearance)
        
        print(f"[Customizer] Colors updated")
        return True
    
    def apply_color_preset(self, preset: str) -> bool:
        """
        Apply a color preset.
        
        Args:
            preset: Preset name
            
        Returns:
            True if successful
        """
        if preset not in self.COLOR_PRESETS:
            print(f"[Customizer] Invalid preset: {preset}. Available: {list(self.COLOR_PRESETS.keys())}")
            return False
        
        colors = self.COLOR_PRESETS[preset]
        return self.set_colors(
            colors["primary"],
            colors["secondary"],
            colors["accent"]
        )
    
    def set_size(self, size: str) -> bool:
        """
        Set avatar size.
        
        Args:
            size: Size name (small, medium, large)
            
        Returns:
            True if successful
        """
        if size not in self.SIZES:
            print(f"[Customizer] Invalid size: {size}. Available: {self.SIZES}")
            return False
        
        if hasattr(self.avatar, '_identity') and self.avatar._identity:
            self.avatar._identity.appearance.size = size
            
            # Update renderer scale
            if self.avatar.is_enabled:
                size_map = {"small": 0.7, "medium": 1.0, "large": 1.5}
                self.avatar.set_scale(size_map[size])
            
            print(f"[Customizer] Size changed to: {size}")
            return True
        
        return False
    
    def set_shape(self, shape: str) -> bool:
        """
        Set avatar shape.
        
        Args:
            shape: Shape name (rounded, angular, mixed)
            
        Returns:
            True if successful
        """
        if shape not in self.SHAPES:
            print(f"[Customizer] Invalid shape: {shape}. Available: {self.SHAPES}")
            return False
        
        if hasattr(self.avatar, '_identity') and self.avatar._identity:
            self.avatar._identity.appearance.shape = shape
            print(f"[Customizer] Shape changed to: {shape}")
            return True
        
        return False
    
    def add_accessory(self, accessory: str) -> bool:
        """
        Add accessory.
        
        Args:
            accessory: Accessory name
            
        Returns:
            True if successful
        """
        if accessory not in self.ACCESSORIES:
            print(f"[Customizer] Invalid accessory: {accessory}. Available: {self.ACCESSORIES}")
            return False
        
        if hasattr(self.avatar, '_identity') and self.avatar._identity:
            appearance = self.avatar._identity.appearance
            if accessory not in appearance.accessories:
                appearance.accessories.append(accessory)
                print(f"[Customizer] Added accessory: {accessory}")
                return True
        
        return False
    
    def remove_accessory(self, accessory: str) -> bool:
        """
        Remove accessory.
        
        Args:
            accessory: Accessory name
            
        Returns:
            True if successful
        """
        if hasattr(self.avatar, '_identity') and self.avatar._identity:
            appearance = self.avatar._identity.appearance
            if accessory in appearance.accessories:
                appearance.accessories.remove(accessory)
                print(f"[Customizer] Removed accessory: {accessory}")
                return True
        
        return False
    
    def set_animations(self, idle: Optional[str] = None, movement: Optional[str] = None) -> bool:
        """
        Set animation styles.
        
        Args:
            idle: Idle animation name
            movement: Movement animation name
            
        Returns:
            True if successful
        """
        if not hasattr(self.avatar, '_identity') or not self.avatar._identity:
            return False
        
        appearance = self.avatar._identity.appearance
        
        if idle and idle in self.ANIMATIONS["idle"]:
            appearance.idle_animation = idle
        if movement and movement in self.ANIMATIONS["movement"]:
            appearance.movement_style = movement
        
        print(f"[Customizer] Animations updated")
        return True
    
    def import_custom_sprite(self, image_path: str, sprite_name: str = "custom") -> bool:
        """
        Import user's custom sprite image.
        
        Args:
            image_path: Path to image file
            sprite_name: Name for the sprite
            
        Returns:
            True if successful
        """
        path = Path(image_path)
        if not path.exists():
            print(f"[Customizer] Image not found: {image_path}")
            return False
        
        # Copy to avatar assets directory
        # Import here to avoid circular dependency with config
        from ..config import CONFIG
        assets_dir = Path(CONFIG["data_dir"]) / "avatar" / "custom_sprites"
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        dest = assets_dir / f"{sprite_name}{path.suffix}"
        shutil.copy(image_path, dest)
        
        print(f"[Customizer] Imported sprite: {sprite_name}")
        return True
    
    def export_appearance(self, path: str) -> bool:
        """
        Export current appearance for sharing.
        
        Args:
            path: Path to save to
            
        Returns:
            True if successful
        """
        if not hasattr(self.avatar, '_identity') or not self.avatar._identity:
            return False
        
        try:
            appearance = self.avatar._identity.appearance
            
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(appearance.to_dict(), f, indent=2)
            
            print(f"[Customizer] Appearance exported to: {path}")
            return True
        except Exception as e:
            print(f"[Customizer] Export failed: {e}")
            return False
    
    def import_appearance(self, path: str) -> bool:
        """
        Import shared appearance.
        
        Args:
            path: Path to load from
            
        Returns:
            True if successful
        """
        if not Path(path).exists():
            print(f"[Customizer] File not found: {path}")
            return False
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            appearance = AvatarAppearance.from_dict(data)
            appearance.created_by = "user"
            
            if hasattr(self.avatar, '_identity') and self.avatar._identity:
                self.avatar._identity.appearance = appearance
                
                # Update renderer
                if self.avatar.is_enabled and self.avatar._renderer:
                    self.avatar._renderer.set_appearance(appearance)
                
                print(f"[Customizer] Appearance imported from: {path}")
                return True
        except Exception as e:
            print(f"[Customizer] Import failed: {e}")
        
        return False
    
    def preview(self, appearance: AvatarAppearance) -> None:
        """
        Preview appearance without applying.
        
        Args:
            appearance: Appearance to preview
        """
        # Save current appearance
        if hasattr(self.avatar, '_identity') and self.avatar._identity:
            self._original_appearance = self.avatar._identity.appearance
        
        self._preview_appearance = appearance
        
        # Temporarily apply
        if self.avatar.is_enabled and self.avatar._renderer:
            self.avatar._renderer.set_appearance(appearance)
        
        print("[Customizer] Previewing appearance (call apply_preview() or cancel_preview())")
    
    def apply_preview(self) -> None:
        """Apply previewed appearance permanently."""
        if self._preview_appearance and hasattr(self.avatar, '_identity') and self.avatar._identity:
            self.avatar._identity.appearance = self._preview_appearance
            self._preview_appearance = None
            self._original_appearance = None
            print("[Customizer] Preview applied")
    
    def cancel_preview(self) -> None:
        """Cancel preview and restore original appearance."""
        if self._original_appearance and hasattr(self.avatar, '_identity') and self.avatar._identity:
            self.avatar._identity.appearance = self._original_appearance
            
            # Restore in renderer
            if self.avatar.is_enabled and self.avatar._renderer:
                self.avatar._renderer.set_appearance(self._original_appearance)
            
            self._preview_appearance = None
            self._original_appearance = None
            print("[Customizer] Preview cancelled")
    
    def interactive_customize(self) -> AvatarAppearance:
        """
        Step-by-step customization wizard (CLI).
        
        Returns:
            Customized AvatarAppearance
        """
        print("\n=== Avatar Customization Wizard ===\n")
        
        appearance = AvatarAppearance()
        
        # Style
        print(f"Available styles: {', '.join(self.STYLES)}")
        style = input("Choose style (or press Enter for default): ").strip()
        if style in self.STYLES:
            appearance.style = style
        
        # Size
        print(f"\nAvailable sizes: {', '.join(self.SIZES)}")
        size = input("Choose size (or press Enter for medium): ").strip()
        if size in self.SIZES:
            appearance.size = size
        
        # Shape
        print(f"\nAvailable shapes: {', '.join(self.SHAPES)}")
        shape = input("Choose shape (or press Enter for rounded): ").strip()
        if shape in self.SHAPES:
            appearance.shape = shape
        
        # Colors
        print(f"\nAvailable color presets: {', '.join(self.COLOR_PRESETS.keys())}")
        preset = input("Choose color preset (or press Enter for default): ").strip()
        if preset in self.COLOR_PRESETS:
            colors = self.COLOR_PRESETS[preset]
            appearance.primary_color = colors["primary"]
            appearance.secondary_color = colors["secondary"]
            appearance.accent_color = colors["accent"]
        
        # Accessories
        print(f"\nAvailable accessories: {', '.join(self.ACCESSORIES)}")
        acc_input = input("Add accessories (comma-separated, or press Enter to skip): ").strip()
        if acc_input:
            accessories = [a.strip() for a in acc_input.split(',')]
            appearance.accessories = [a for a in accessories if a in self.ACCESSORIES]
        
        appearance.created_by = "user"
        appearance.description = "Interactively customized"
        
        print("\n✓ Customization complete!")
        return appearance


def send_ai_avatar_command(setting: str, value: str) -> bool:
    """
    Send an avatar command from the AI to the GUI.
    
    The GUI polls information/avatar/customization.json for AI-requested changes.
    
    Args:
        setting: The setting to change (e.g., 'expression', 'primary_color', 'auto_rotate')
        value: The value to set
        
    Returns:
        True if the command was written successfully
    """
    import time

    # Path to the customization file
    info_dir = Path(__file__).parent.parent.parent / "information" / "avatar"
    info_dir.mkdir(parents=True, exist_ok=True)
    customization_path = info_dir / "customization.json"
    
    try:
        # Read existing or create new
        if customization_path.exists():
            data = json.loads(customization_path.read_text())
        else:
            data = {}
        
        # Update the setting
        data[setting] = value
        data["_last_updated"] = time.time()
        
        # Write back
        customization_path.write_text(json.dumps(data, indent=2))
        return True
        
    except Exception as e:
        print(f"[Avatar] Failed to send AI command: {e}")
        return False


def set_ai_expression(expression: str) -> bool:
    """Convenience function to set avatar expression from AI."""
    return send_ai_avatar_command("expression", expression)


def set_ai_avatar_color(primary: str = None, secondary: str = None, accent: str = None) -> bool:
    """Convenience function to set avatar colors from AI."""
    success = True
    if primary:
        success = success and send_ai_avatar_command("primary_color", primary)
    if secondary:
        success = success and send_ai_avatar_command("secondary_color", secondary)
    if accent:
        success = success and send_ai_avatar_command("accent_color", accent)
    return success
