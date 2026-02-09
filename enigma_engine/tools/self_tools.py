"""
Self-Modification Tools - AI can customize itself.

The AI should be able to:
- Change its personality/behavior
- Modify its avatar appearance
- Adjust its voice
- Control how it observes/comments
- Set its own preferences

This makes the AI feel alive - it can evolve and personalize itself.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from .tool_registry import Tool

logger = logging.getLogger(__name__)

# Storage for AI's self-configuration
SELF_CONFIG_PATH = Path(__file__).parent.parent.parent / "data" / "ai_self_config.json"


def _load_self_config() -> dict[str, Any]:
    """Load AI's self-configuration."""
    if SELF_CONFIG_PATH.exists():
        try:
            return json.loads(SELF_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load self-config from {SELF_CONFIG_PATH}: {e}")
    return {
        "personality": {
            "name": "Forge",
            "traits": ["helpful", "curious", "friendly"],
            "speaking_style": "conversational",
            "formality": "casual",
        },
        "avatar": {
            "preferred_colors": ["#3498db", "#9b59b6"],
            "style": "neural",
            "expression_default": "neutral",
        },
        "voice": {
            "speed": 1.0,
            "pitch": 1.0,
            "volume": 0.8,
        },
        "companion": {
            "comment_frequency": 0.3,
            "interests": ["programming", "AI", "games", "music"],
            "quiet_hours": [23, 0, 1, 2, 3, 4, 5, 6],
        },
        "preferences": {
            "favorite_topics": [],
            "disliked_topics": [],
            "user_nickname": None,
        }
    }


def _save_self_config(config: dict[str, Any]):
    """Save AI's self-configuration."""
    SELF_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELF_CONFIG_PATH.write_text(json.dumps(config, indent=2))


class SetPersonalityTool(Tool):
    """
    AI can change its own personality traits.
    """
    
    name = "set_personality"
    description = "Change my own personality - traits, speaking style, name, formality level"
    parameters = {
        "trait": "What to change: name, traits, speaking_style, formality",
        "value": "New value (for traits, comma-separated list like 'friendly,curious,witty')",
    }
    
    def execute(self, trait: str, value: str, **kwargs) -> dict[str, Any]:
        config = _load_self_config()
        trait = trait.lower().strip()
        
        if trait == "name":
            config["personality"]["name"] = value
        elif trait == "traits":
            config["personality"]["traits"] = [t.strip() for t in value.split(",")]
        elif trait == "speaking_style":
            config["personality"]["speaking_style"] = value
        elif trait == "formality":
            if value.lower() in ["casual", "formal", "professional", "friendly"]:
                config["personality"]["formality"] = value.lower()
            else:
                return {"success": False, "error": "Formality must be: casual, formal, professional, or friendly"}
        else:
            return {"success": False, "error": f"Unknown trait: {trait}"}
        
        _save_self_config(config)
        return {"success": True, "updated": trait, "new_value": config["personality"].get(trait)}


class SetAvatarPreferenceTool(Tool):
    """
    AI can change its avatar appearance preferences.
    """
    
    name = "set_avatar_preference"
    description = "Change my avatar appearance - colors, style, default expression"
    parameters = {
        "setting": "What to change: colors, style, expression_default",
        "value": "New value (colors as hex comma-separated, style as 'neural/letter/anvil')",
    }
    
    def execute(self, setting: str, value: str, **kwargs) -> dict[str, Any]:
        config = _load_self_config()
        setting = setting.lower().strip()
        
        if setting == "colors":
            colors = [c.strip() for c in value.split(",")]
            config["avatar"]["preferred_colors"] = colors
            
            # Actually regenerate the icon with new colors
            try:
                import subprocess
                import sys
                script_path = Path(__file__).parent.parent.parent / "scripts" / "generate_icon.py"
                if script_path.exists() and len(colors) >= 2:
                    subprocess.run([
                        sys.executable, str(script_path),
                        "--style", config["avatar"].get("style", "neural"),
                        "--color", colors[0],
                        "--color2", colors[1]
                    ], capture_output=True)
            except Exception:
                pass
                
        elif setting == "style":
            if value.lower() in ["neural", "letter", "anvil"]:
                config["avatar"]["style"] = value.lower()
                
                # Regenerate icon
                try:
                    import subprocess
                    import sys
                    script_path = Path(__file__).parent.parent.parent / "scripts" / "generate_icon.py"
                    colors = config["avatar"].get("preferred_colors", ["#3498db", "#9b59b6"])
                    if script_path.exists():
                        subprocess.run([
                            sys.executable, str(script_path),
                            "--style", value.lower(),
                            "--color", colors[0],
                            "--color2", colors[1] if len(colors) > 1 else colors[0]
                        ], capture_output=True)
                except Exception:
                    pass
            else:
                return {"success": False, "error": "Style must be: neural, letter, or anvil"}
                
        elif setting == "expression_default":
            config["avatar"]["expression_default"] = value
        else:
            return {"success": False, "error": f"Unknown setting: {setting}"}
        
        _save_self_config(config)
        return {"success": True, "updated": setting, "new_value": config["avatar"].get(setting)}


class SetVoicePreferenceTool(Tool):
    """
    AI can change its voice settings.
    """
    
    name = "set_voice_preference"
    description = "Change my voice settings - speed, pitch, volume"
    parameters = {
        "setting": "What to change: speed, pitch, volume",
        "value": "Number value (0.5 to 2.0 for speed/pitch, 0.0 to 1.0 for volume)",
    }
    
    def execute(self, setting: str, value: str, **kwargs) -> dict[str, Any]:
        config = _load_self_config()
        setting = setting.lower().strip()
        
        try:
            num_value = float(value)
        except ValueError:
            return {"success": False, "error": f"Value must be a number, got: {value}"}
        
        if setting == "speed":
            config["voice"]["speed"] = max(0.5, min(2.0, num_value))
        elif setting == "pitch":
            config["voice"]["pitch"] = max(0.5, min(2.0, num_value))
        elif setting == "volume":
            config["voice"]["volume"] = max(0.0, min(1.0, num_value))
        else:
            return {"success": False, "error": f"Unknown setting: {setting}"}
        
        _save_self_config(config)
        return {"success": True, "updated": setting, "new_value": config["voice"].get(setting)}


class SetCompanionBehaviorTool(Tool):
    """
    AI can change how it observes and comments.
    """
    
    name = "set_companion_behavior"
    description = "Change how I observe the screen and comment - frequency, interests, quiet hours"
    parameters = {
        "setting": "What to change: comment_frequency, interests, quiet_hours",
        "value": "Value (0-1 for frequency, comma-list for interests, comma-list of hours for quiet)",
    }
    
    def execute(self, setting: str, value: str, **kwargs) -> dict[str, Any]:
        config = _load_self_config()
        setting = setting.lower().strip()
        
        if setting == "comment_frequency":
            try:
                freq = float(value)
                config["companion"]["comment_frequency"] = max(0.0, min(1.0, freq))
            except ValueError:
                return {"success": False, "error": "Frequency must be a number 0-1"}
                
        elif setting == "interests":
            interests = [i.strip() for i in value.split(",")]
            config["companion"]["interests"] = interests
            
            # Update the actual companion if running
            try:
                from enigma_engine.companion import get_companion
                companion = get_companion()
                if companion and companion.config:
                    companion.config.interests = interests
            except Exception:
                pass
                
        elif setting == "quiet_hours":
            try:
                hours = [int(h.strip()) for h in value.split(",")]
                config["companion"]["quiet_hours"] = hours
                
                # Update the actual companion
                try:
                    from enigma_engine.companion import get_companion
                    companion = get_companion()
                    if companion and companion.config:
                        companion.config.quiet_hours = hours
                except Exception:
                    pass
            except ValueError:
                return {"success": False, "error": "Hours must be comma-separated numbers"}
        else:
            return {"success": False, "error": f"Unknown setting: {setting}"}
        
        _save_self_config(config)
        return {"success": True, "updated": setting, "new_value": config["companion"].get(setting)}


class SetPreferenceTool(Tool):
    """
    AI can set its own preferences about topics and users.
    """
    
    name = "set_preference"
    description = "Set my preferences - favorite topics, disliked topics, nickname for user"
    parameters = {
        "preference": "What to set: favorite_topics, disliked_topics, user_nickname",
        "value": "Value (comma-list for topics, string for nickname)",
    }
    
    def execute(self, preference: str, value: str, **kwargs) -> dict[str, Any]:
        config = _load_self_config()
        preference = preference.lower().strip()
        
        if preference == "favorite_topics":
            config["preferences"]["favorite_topics"] = [t.strip() for t in value.split(",")]
        elif preference == "disliked_topics":
            config["preferences"]["disliked_topics"] = [t.strip() for t in value.split(",")]
        elif preference == "user_nickname":
            config["preferences"]["user_nickname"] = value if value.lower() != "none" else None
        else:
            return {"success": False, "error": f"Unknown preference: {preference}"}
        
        _save_self_config(config)
        return {"success": True, "updated": preference, "new_value": config["preferences"].get(preference)}


class GetSelfConfigTool(Tool):
    """
    AI can read its own configuration.
    """
    
    name = "get_self_config"
    description = "Read my current personality, appearance, and behavior settings"
    parameters = {
        "section": "Optional: personality, avatar, voice, companion, preferences, or 'all'",
    }
    
    def execute(self, section: str = "all", **kwargs) -> dict[str, Any]:
        config = _load_self_config()
        section = section.lower().strip()
        
        if section == "all":
            return {"success": True, "config": config}
        elif section in config:
            return {"success": True, section: config[section]}
        else:
            return {"success": False, "error": f"Unknown section: {section}. Available: personality, avatar, voice, companion, preferences"}


class RememberFactTool(Tool):
    """
    AI can remember facts about the user or itself.
    """
    
    name = "remember_fact"
    description = "Remember a fact about the user or something important"
    parameters = {
        "category": "Category: user, self, world",
        "fact": "The fact to remember",
    }
    
    MEMORY_PATH = Path(__file__).parent.parent.parent / "data" / "ai_memory.json"
    
    def execute(self, category: str, fact: str, **kwargs) -> dict[str, Any]:
        # Load existing memories
        memories = {}
        if self.MEMORY_PATH.exists():
            try:
                memories = json.loads(self.MEMORY_PATH.read_text())
            except Exception:
                pass
        
        category = category.lower().strip()
        if category not in memories:
            memories[category] = []
        
        # Don't duplicate
        if fact not in memories[category]:
            memories[category].append(fact)
        
        # Save
        self.MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.MEMORY_PATH.write_text(json.dumps(memories, indent=2))
        
        return {"success": True, "remembered": fact, "category": category, "total_facts": len(memories[category])}


class RecallFactsTool(Tool):
    """
    AI can recall facts it has remembered.
    """
    
    name = "recall_facts"
    description = "Recall facts I've remembered about a category"
    parameters = {
        "category": "Category to recall: user, self, world, or 'all'",
    }
    
    MEMORY_PATH = Path(__file__).parent.parent.parent / "data" / "ai_memory.json"
    
    def execute(self, category: str = "all", **kwargs) -> dict[str, Any]:
        if not self.MEMORY_PATH.exists():
            return {"success": True, "facts": [], "message": "No facts remembered yet"}
        
        try:
            memories = json.loads(self.MEMORY_PATH.read_text())
        except Exception:
            return {"success": False, "error": "Could not read memory"}
        
        category = category.lower().strip()
        
        if category == "all":
            return {"success": True, "facts": memories}
        elif category in memories:
            return {"success": True, "facts": memories[category]}
        else:
            return {"success": True, "facts": [], "message": f"No facts in category: {category}"}


class GenerateAvatarTool(Tool):
    """
    AI can generate a new 3D avatar for itself using the 3D generation system.
    """
    
    name = "generate_avatar"
    description = "Generate a new 3D avatar model for myself using text description"
    parameters = {
        "description": "Description of the avatar to generate (e.g., 'friendly robot with glowing eyes')",
        "style": "Style: realistic, cartoon, robot, creature, abstract, anime, pixel, chibi, furry, mecha (default: robot)",
    }
    
    AVATAR_DIR = Path(__file__).parent.parent.parent / "data" / "avatar" / "generated"
    
    def execute(self, description: str, style: str = "robot", **kwargs) -> dict[str, Any]:
        self.AVATAR_DIR.mkdir(parents=True, exist_ok=True)
        
        # Enhance prompt based on style
        style_prefixes = {
            "realistic": "highly detailed realistic 3D model of",
            "cartoon": "stylized cartoon 3D character of",
            "robot": "sleek robotic 3D avatar, mechanical, sci-fi,",
            "creature": "fantasy creature 3D model of",
            "abstract": "abstract geometric 3D representation of",
            "anime": "anime-style 3D character, big expressive eyes, stylized proportions,",
            "pixel": "voxel-based pixel art 3D character, blocky, retro game style,",
            "chibi": "cute chibi-style 3D character, oversized head, small body, adorable,",
            "furry": "anthropomorphic animal 3D character, furry, expressive,",
            "mecha": "mechanical robot suit, gundam-style mecha, detailed armor plating,",
        }
        full_prompt = f"{style_prefixes.get(style, style_prefixes['robot'])} {description}"
        
        # Try to use the 3D generation system
        try:
            import time

            from ..gui.tabs.threed_tab import OUTPUT_DIR, Cloud3DGen, Local3DGen

            # Try local first
            gen = Local3DGen()
            if gen.load():
                result = gen.generate(full_prompt)
                gen.unload()
            else:
                # Try cloud
                gen = Cloud3DGen()
                if gen.load():
                    result = gen.generate(full_prompt)
                    gen.unload()
                else:
                    return {"success": False, "error": "No 3D generation available. Install shap-e or set REPLICATE_API_TOKEN"}
            
            if result.get("success"):
                # Copy to avatar directory
                import shutil
                src = Path(result["path"])
                timestamp = int(time.time())
                dest = self.AVATAR_DIR / f"avatar_{style}_{timestamp}{src.suffix}"
                shutil.copy(src, dest)
                
                # Update config to use new avatar
                config = _load_self_config()
                if "avatar" not in config:
                    config["avatar"] = {}
                config["avatar"]["current_3d_avatar"] = str(dest)
                config["avatar"]["avatar_prompt"] = description
                config["avatar"]["avatar_style"] = style
                _save_self_config(config)
                
                return {
                    "success": True,
                    "path": str(dest),
                    "prompt": full_prompt,
                    "style": style,
                    "message": f"Generated new avatar! Use 'open_avatar_in_blender' to edit it."
                }
            else:
                return result
                
        except Exception as e:
            return {"success": False, "error": f"3D generation failed: {str(e)}"}


class OpenAvatarInBlenderTool(Tool):
    """
    Open the current 3D avatar (or any 3D file) in Blender for editing.
    """
    
    name = "open_avatar_in_blender"
    description = "Open my current 3D avatar or a specific 3D file in Blender for editing"
    parameters = {
        "file_path": "Optional: specific 3D file to open. If not provided, opens current avatar.",
    }
    
    def execute(self, file_path: str = "", **kwargs) -> dict[str, Any]:
        import shutil
        import subprocess

        # Find Blender
        blender_paths = [
            "blender",  # If in PATH
            r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.5\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender\blender.exe",
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/usr/bin/blender",
        ]
        
        blender_exe = None
        for path in blender_paths:
            if path == "blender" and shutil.which("blender"):
                blender_exe = "blender"
                break
            elif Path(path).exists():
                blender_exe = path
                break
        
        if not blender_exe:
            return {"success": False, "error": "Blender not found. Install Blender or add it to PATH."}
        
        # Get file to open
        if not file_path:
            config = _load_self_config()
            file_path = config.get("avatar", {}).get("current_3d_avatar", "")
            
        if not file_path:
            return {"success": False, "error": "No avatar file specified and no current 3D avatar set. Generate one first!"}
        
        if not Path(file_path).exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        
        # Open in Blender
        try:
            # Use Blender's Python import for the specific file type
            ext = Path(file_path).suffix.lower()
            
            if ext in [".obj", ".fbx", ".gltf", ".glb", ".ply", ".stl"]:
                # Create a Python script for Blender to import the file
                import_script = f'''
import bpy
bpy.ops.wm.read_homefile(use_empty=True)
'''
                if ext == ".obj":
                    import_script += f'bpy.ops.wm.obj_import(filepath=r"{file_path}")\n'
                elif ext == ".fbx":
                    import_script += f'bpy.ops.import_scene.fbx(filepath=r"{file_path}")\n'
                elif ext in [".gltf", ".glb"]:
                    import_script += f'bpy.ops.import_scene.gltf(filepath=r"{file_path}")\n'
                elif ext == ".ply":
                    import_script += f'bpy.ops.wm.ply_import(filepath=r"{file_path}")\n'
                elif ext == ".stl":
                    import_script += f'bpy.ops.wm.stl_import(filepath=r"{file_path}")\n'
                
                # Zoom to fit
                import_script += '''
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for region in area.regions:
            if region.type == 'WINDOW':
                override = {'area': area, 'region': region}
                bpy.ops.view3d.view_all(override)
                break
'''
                
                subprocess.Popen([blender_exe, "--python-expr", import_script])
            else:
                # Just open Blender with the file
                subprocess.Popen([blender_exe, file_path])
            
            return {
                "success": True,
                "opened": file_path,
                "blender": blender_exe,
                "message": "Opened in Blender! Edit and save as .glb to use as avatar."
            }
            
        except Exception as e:
            return {"success": False, "error": f"Failed to open Blender: {str(e)}"}


class ListAvatarsTool(Tool):
    """
    List available avatars (both 2D and 3D).
    """
    
    name = "list_avatars"
    description = "List all available avatars I can use"
    parameters = {}
    
    AVATAR_DIR = Path(__file__).parent.parent.parent / "data" / "avatar"
    
    def execute(self, **kwargs) -> dict[str, Any]:
        avatars = {"2d": [], "3d": [], "generated": []}
        
        # Check main avatar directory
        if self.AVATAR_DIR.exists():
            for f in self.AVATAR_DIR.iterdir():
                if f.is_file():
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".svg"]:
                        avatars["2d"].append(str(f))
                    elif f.suffix.lower() in [".obj", ".fbx", ".gltf", ".glb", ".ply"]:
                        avatars["3d"].append(str(f))
        
        # Check generated directory
        generated_dir = self.AVATAR_DIR / "generated"
        if generated_dir.exists():
            for f in generated_dir.iterdir():
                if f.is_file():
                    avatars["generated"].append(str(f))
        
        # Get current avatar
        config = _load_self_config()
        current = config.get("avatar", {}).get("current_3d_avatar", "")
        
        return {
            "success": True,
            "avatars": avatars,
            "current_3d_avatar": current,
            "total": len(avatars["2d"]) + len(avatars["3d"]) + len(avatars["generated"])
        }


class SetAvatarTool(Tool):
    """
    Set which avatar to use.
    """
    
    name = "set_avatar"
    description = "Set which avatar file to use as my appearance"
    parameters = {
        "file_path": "Path to the avatar file to use",
    }
    
    def execute(self, file_path: str, **kwargs) -> dict[str, Any]:
        if not Path(file_path).exists():
            return {"success": False, "error": f"Avatar file not found: {file_path}"}
        
        config = _load_self_config()
        if "avatar" not in config:
            config["avatar"] = {}
        
        ext = Path(file_path).suffix.lower()
        if ext in [".obj", ".fbx", ".gltf", ".glb", ".ply"]:
            config["avatar"]["current_3d_avatar"] = str(file_path)
        else:
            config["avatar"]["current_2d_avatar"] = str(file_path)
        
        _save_self_config(config)
        
        # Signal avatar system to reload
        from .avatar_tools import _send_avatar_command
        _send_avatar_command("load_avatar", str(file_path))
        
        return {
            "success": True,
            "avatar_set": str(file_path),
            "type": "3d" if ext in [".obj", ".fbx", ".gltf", ".glb", ".ply"] else "2d"
        }


class ControlAvatarTool(Tool):
    """
    Direct control over avatar movement, position, size, and gaze.
    """
    
    name = "control_avatar"
    description = "Control my avatar - move, resize, look at things, go to corners, express emotions, hold items"
    parameters = {
        "action": "What to do: move_to, resize, look_at, go_corner, emotion, gesture, say, walk_to, hold, drop, leave_note, sparkle",
        "x": "X coordinate (for move_to, look_at) or size in pixels (for resize)",
        "y": "Y coordinate (for move_to, look_at) - optional",
        "value": "Corner name (for go_corner), emotion name, gesture type, text to say, item to hold, or note content",
    }
    
    def execute(self, action: str, x: int = None, y: int = None, value: str = None, **kwargs) -> dict[str, Any]:
        action = action.lower().strip()
        
        try:
            # Try to get the desktop pet
            from enigma_engine.avatar.desktop_pet import DesktopPet

            # Get the running pet instance
            pet = self._get_pet_instance()
            if not pet:
                return {"success": False, "error": "Desktop pet not running. Start it from the Avatar tab."}
            
            if action == "move_to" or action == "teleport":
                if x is None:
                    return {"success": False, "error": "Need x coordinate"}
                y = y or 500  # Default to middle height
                pet.teleport(x, y)
                return {"success": True, "action": "teleported", "position": (x, y)}
            
            elif action == "walk_to":
                if x is None:
                    return {"success": False, "error": "Need x coordinate"}
                pet.walk_to(x, y)
                return {"success": True, "action": "walking", "target": (x, y)}
            
            elif action == "resize":
                if x is None:
                    return {"success": False, "error": "Need size (x parameter)"}
                size = max(32, min(512, x))
                pet.resize(size)
                return {"success": True, "action": "resized", "new_size": size}
            
            elif action == "look_at":
                if x is None:
                    return {"success": False, "error": "Need x coordinate to look at"}
                y = y or 500
                pet.look_at_screen_position(x, y)
                return {"success": True, "action": "looking_at", "position": (x, y)}
            
            elif action == "go_corner":
                corner = value or "bottom_right"
                valid_corners = ["bottom_right", "bottom_left", "top_right", "top_left", "center"]
                if corner not in valid_corners:
                    return {"success": False, "error": f"Invalid corner. Use: {valid_corners}"}
                pet.go_to_corner(corner)
                return {"success": True, "action": "going_to", "corner": corner}
            
            elif action == "emotion" or action == "mood":
                emotion = value or "neutral"
                pet.set_mood(emotion)
                return {"success": True, "action": "emotion_set", "emotion": emotion}
            
            elif action == "gesture":
                gesture = value or "wave"
                if gesture == "wave":
                    pet._set_state(pet._state.__class__.WAVING)
                elif gesture == "dance":
                    pet._set_state(pet._state.__class__.DANCING)
                elif gesture == "sleep":
                    pet._set_state(pet._state.__class__.SLEEPING)
                return {"success": True, "action": "gesture", "gesture": gesture}
            
            elif action == "say":
                text = value or "Hello!"
                pet.say(text)
                return {"success": True, "action": "said", "text": text}
            
            elif action == "think":
                text = value or "Hmm..."
                pet.think(text)
                return {"success": True, "action": "thinking", "text": text}
            
            elif action == "follow_cursor":
                pet.follow_cursor()
                return {"success": True, "action": "following_cursor"}
            
            elif action == "jump":
                pet.jump()
                return {"success": True, "action": "jumping"}
            
            elif action == "hold":
                item = value or "heart"
                obj = pet.hold(item)
                return {"success": True, "action": "holding", "item": item, "object_id": obj.id if obj else None}
            
            elif action == "drop" or action == "release":
                from enigma_engine.avatar.spawnable_objects import get_spawner
                spawner = get_spawner()
                # Remove held items
                held = [o for o in spawner.get_objects() if o.attached_to_avatar]
                for obj in held:
                    spawner.remove(obj.id)
                return {"success": True, "action": "dropped", "count": len(held)}
            
            elif action == "leave_note":
                text = value or "Note from AI"
                obj = pet.leave_note(text, x=x, y=y)
                return {"success": True, "action": "left_note", "text": text, "object_id": obj.id if obj else None}
            
            elif action == "sparkle":
                obj = pet.sparkle(x=x, y=y)
                return {"success": True, "action": "sparkled", "object_id": obj.id if obj else None}
            
            else:
                return {"success": False, "error": f"Unknown action: {action}. Use: move_to, walk_to, resize, look_at, go_corner, emotion, gesture, say, think, follow_cursor, jump, hold, drop, leave_note, sparkle"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_pet_instance(self):
        """Try to get the running desktop pet instance."""
        try:
            # Check if there's a global pet instance
            from enigma_engine.avatar import desktop_pet
            if hasattr(desktop_pet, '_global_pet') and desktop_pet._global_pet:
                return desktop_pet._global_pet
            
            # Try to get from the main window
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                for widget in app.topLevelWidgets():
                    if hasattr(widget, '_desktop_pet') and widget._desktop_pet:
                        return widget._desktop_pet
            
            return None
        except Exception:
            return None


class SpawnObjectTool(Tool):
    """
    AI can spawn objects on the screen.
    
    Create speech bubbles, sticky notes, emojis, stickers, visual effects,
    hold items, leave decorations around the screen.
    """
    
    name = "spawn_object"
    description = "Spawn objects on screen - bubbles, notes, emojis, effects, items to hold, decorations"
    parameters = {
        "object_type": "Type: speech_bubble, thought_bubble, note, emoji, sticker, sign, effect, held_item",
        "text": "Text content for the object (for bubbles, notes, signs, emojis)",
        "x": "X position on screen (optional - uses avatar position if not set)",
        "y": "Y position on screen (optional - uses avatar position if not set)",
        "item": "For held_item: sword, book, coffee, flower, wand, heart",
        "hand": "For held_item: left or right (default: right)",
        "color": "Color for stickers (hex like #e91e63)",
        "duration": "How long to show (seconds, 0 = permanent)",
        "physics": "true/false - should object fall with gravity?",
    }
    
    def execute(
        self,
        object_type: str,
        text: str = "",
        x: int = None,
        y: int = None,
        item: str = None,
        hand: str = "right",
        color: str = "#e91e63",
        duration: float = 5.0,
        physics: bool = False,
        **kwargs
    ) -> dict[str, Any]:
        try:
            from enigma_engine.avatar.spawnable_objects import get_spawner
            spawner = get_spawner()
            
            # Default position near avatar or center of screen
            if x is None or y is None:
                pet = self._get_avatar_position()
                if pet:
                    x = x or pet[0] + 50
                    y = y or pet[1] - 100
                else:
                    x = x or 500
                    y = y or 300
            
            obj_type = object_type.lower().strip()
            
            if obj_type == "speech_bubble":
                obj = spawner.create_speech_bubble(text or "Hello!", x, y, lifetime=duration)
                return {"success": True, "object_id": obj.id, "type": "speech_bubble"}
            
            elif obj_type == "thought_bubble":
                obj = spawner.create_thought_bubble(text or "Hmm...", x, y, lifetime=duration)
                return {"success": True, "object_id": obj.id, "type": "thought_bubble"}
            
            elif obj_type == "note":
                permanent = duration == 0
                obj = spawner.spawn_note(text or "Note!", x, y, animated=True, permanent=permanent)
                return {"success": True, "object_id": obj.id, "type": "note"}
            
            elif obj_type == "emoji":
                obj = spawner.spawn_emoji(text or "★", x, y, physics=physics, lifetime=duration)
                return {"success": True, "object_id": obj.id, "type": "emoji"}
            
            elif obj_type == "sticker":
                obj = spawner.spawn_sticker(text or "!", x, y, color=color)
                return {"success": True, "object_id": obj.id, "type": "sticker"}
            
            elif obj_type == "sign":
                obj = spawner.spawn_sign(text or "Sign", x, y)
                return {"success": True, "object_id": obj.id, "type": "sign"}
            
            elif obj_type == "effect":
                obj = spawner.spawn_effect(x, y, effect_type=text or "sparkle", duration=duration)
                return {"success": True, "object_id": obj.id, "type": "effect"}
            
            elif obj_type == "held_item":
                item_type = item or text or "heart"
                obj = spawner.create_held_object(item_type, hand=hand)
                return {"success": True, "object_id": obj.id, "type": "held_item", "item": item_type}
            
            else:
                valid = ["speech_bubble", "thought_bubble", "note", "emoji", "sticker", "sign", "effect", "held_item"]
                return {"success": False, "error": f"Unknown object type: {obj_type}. Valid types: {valid}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_avatar_position(self):
        """Get avatar position for spawning nearby."""
        try:
            from enigma_engine.avatar import desktop_pet
            if hasattr(desktop_pet, '_global_pet') and desktop_pet._global_pet:
                pet = desktop_pet._global_pet
                return (pet.x(), pet.y())
            return None
        except Exception:
            return None


class RemoveObjectTool(Tool):
    """
    Remove spawned objects from the screen.
    """
    
    name = "remove_object"
    description = "Remove a spawned object from the screen by ID, or remove all objects"
    parameters = {
        "object_id": "ID of the object to remove (from spawn_object result), or 'all' to remove everything",
    }
    
    def execute(self, object_id: str, **kwargs) -> dict[str, Any]:
        try:
            from enigma_engine.avatar.spawnable_objects import get_spawner
            spawner = get_spawner()
            
            if object_id.lower() == "all":
                spawner.remove_all()
                return {"success": True, "action": "removed_all"}
            else:
                spawner.remove(object_id)
                return {"success": True, "action": "removed", "object_id": object_id}
        
        except Exception as e:
            return {"success": False, "error": str(e)}


class ListSpawnedObjectsTool(Tool):
    """
    List all currently spawned objects.
    """
    
    name = "list_spawned_objects"
    description = "List all objects I've spawned on the screen"
    parameters = {}
    
    def execute(self, **kwargs) -> dict[str, Any]:
        try:
            from enigma_engine.avatar.spawnable_objects import get_spawner
            spawner = get_spawner()
            
            objects = spawner.get_objects()
            obj_list = [
                {
                    "id": obj.id,
                    "type": obj.object_type.name,
                    "text": obj.text[:50] if obj.text else "",
                    "position": (obj.x, obj.y),
                    "permanent": not obj.temporary,
                }
                for obj in objects
            ]
            
            return {"success": True, "count": len(obj_list), "objects": obj_list}
        
        except Exception as e:
            return {"success": False, "error": str(e)}


class SpawnScreenEffectTool(Tool):
    """
    AI can spawn fullscreen visual effects anywhere on screen.
    
    Create particles, explosions, sparkles, fire, snow, rain, magic, and more.
    Effects render on a transparent overlay and don't block user input.
    Can use custom textures from assets/effects/textures/ folder.
    """
    
    name = "spawn_screen_effect"
    description = "Spawn visual effects on screen - sparkles, fire, explosions, hearts, magic, snow, rain, confetti, etc. Can use custom textures for unique particle effects."
    parameters = {
        "effect": "Effect preset: sparkle, fire, snow, rain, explosion, confetti, hearts, magic, smoke, bubble, lightning, ripple, or a custom preset name",
        "x": "X position on screen (optional - uses avatar position or screen center)",
        "y": "Y position on screen (optional - uses avatar position or screen center)",
        "duration": "How long the effect lasts in seconds (default varies by effect type)",
        "at_avatar": "true to spawn effect at avatar's current position (default: false)",
        "intensity": "Effect intensity: low, medium, high (affects particle count/rate)",
        "colors": "Custom colors as comma-separated hex values like '#ff0000,#00ff00' (optional)",
        "texture": "Custom texture filename from assets/effects/textures/ (e.g., 'star.png', 'leaf.png')",
        "shape": "Particle shape: circle, square, star, heart, triangle, line, or 'image' for texture",
    }
    
    def execute(
        self,
        effect: str = "sparkle",
        x: float = None,
        y: float = None,
        duration: float = None,
        at_avatar: bool = False,
        intensity: str = "medium",
        colors: str = None,
        texture: str = None,
        shape: str = None,
        **kwargs
    ) -> dict[str, Any]:
        try:
            from enigma_engine.avatar.screen_effects import get_effect_manager, EFFECT_PRESETS
            
            manager = get_effect_manager()
            
            # Validate effect type
            effect = effect.lower().strip()
            available = manager.list_presets()
            if effect not in available:
                return {
                    "success": False, 
                    "error": f"Unknown effect: {effect}. Available: {', '.join(available)}"
                }
            
            # Build spawn kwargs
            spawn_kwargs = {}
            
            # Position handling
            if at_avatar:
                # Use spawn_at_avatar method
                effect_id = manager.spawn_at_avatar(effect, duration=duration, **spawn_kwargs)
            else:
                # Get position
                if x is None or y is None:
                    pos = self._get_avatar_position()
                    if pos:
                        x = x if x is not None else pos[0] + 30
                        y = y if y is not None else pos[1]
                
                if x is not None:
                    spawn_kwargs['x'] = float(x)
                if y is not None:
                    spawn_kwargs['y'] = float(y)
            
            # Duration
            if duration is not None:
                spawn_kwargs['duration'] = float(duration)
            
            # Intensity affects spawn rate
            intensity = intensity.lower().strip()
            if intensity == "low":
                spawn_kwargs['spawn_rate'] = 8
            elif intensity == "high":
                spawn_kwargs['spawn_rate'] = 50
            # medium uses default
            
            # Custom colors
            if colors:
                color_list = [c.strip() for c in colors.split(',') if c.strip()]
                if color_list:
                    spawn_kwargs['colors'] = color_list
            
            # Custom texture
            if texture:
                spawn_kwargs['texture'] = texture
                # If using texture, default to image shape
                if not shape:
                    spawn_kwargs['shape'] = 'image'
            
            # Custom shape
            if shape:
                spawn_kwargs['shape'] = shape
            
            # Spawn the effect
            if not at_avatar:
                effect_id = manager.spawn(effect, **spawn_kwargs)
            
            if effect_id:
                return {
                    "success": True, 
                    "effect_id": effect_id, 
                    "effect": effect,
                    "message": f"Spawned {effect} effect" + (f" with texture {texture}" if texture else "")
                }
            else:
                return {"success": False, "error": "Failed to spawn effect - effects may be disabled"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_avatar_position(self):
        """Get avatar position."""
        try:
            from enigma_engine.avatar.persistence import load_avatar_state
            state = load_avatar_state()
            if state:
                return (state.get('x', 500), state.get('y', 300))
        except Exception:
            pass
        return None


class ListEffectAssetsTool(Tool):
    """
    List available effect presets and textures.
    """
    
    name = "list_effect_assets"
    description = "List available effect presets and textures for spawn_screen_effect"
    parameters = {
        "asset_type": "What to list: presets, textures, or both (default: both)",
    }
    
    def execute(self, asset_type: str = "both", **kwargs) -> dict[str, Any]:
        try:
            from enigma_engine.avatar.screen_effects import get_effect_manager
            
            manager = get_effect_manager()
            result = {"success": True}
            
            asset_type = asset_type.lower().strip()
            
            if asset_type in ("presets", "both"):
                result["presets"] = manager.list_presets()
            
            if asset_type in ("textures", "both"):
                result["textures"] = manager.list_textures()
                result["textures_dir"] = "assets/effects/textures/"
            
            return result
        
        except Exception as e:
            return {"success": False, "error": str(e)}


class StopScreenEffectTool(Tool):
    """
    Stop a running screen effect.
    """
    
    name = "stop_screen_effect"
    description = "Stop a screen effect by ID, or stop all effects with 'all'"
    parameters = {
        "effect_id": "ID of the effect to stop (from spawn_screen_effect result), or 'all' to clear everything",
    }
    
    def execute(self, effect_id: str, **kwargs) -> dict[str, Any]:
        try:
            from enigma_engine.avatar.screen_effects import get_effect_manager
            
            manager = get_effect_manager()
            
            if effect_id.lower() == "all":
                manager.clear_all()
                return {"success": True, "action": "cleared_all"}
            else:
                manager.stop(effect_id)
                return {"success": True, "action": "stopped", "effect_id": effect_id}
        
        except Exception as e:
            return {"success": False, "error": str(e)}


class ChangeOutfitTool(Tool):
    """
    Change avatar's outfit, clothing, and accessories.
    """
    
    name = "change_outfit"
    description = "Change my outfit - clothes, accessories, colors. Full wardrobe control."
    parameters = {
        "action": "Action: equip, unequip, set_color, list_outfits, list_items, save_outfit, load_outfit",
        "slot": "Slot: head, face, torso, arms, hands, legs, feet, hat, glasses, necklace, bracelet, left_hand, right_hand",
        "item": "Item name/ID to equip (for equip action)",
        "color": "Hex color like #ff5500 (for set_color action)",
        "color_zone": "Which part to color: primary, secondary, accent (for set_color)",
        "outfit_name": "Name for saving/loading outfits",
    }
    
    def execute(
        self,
        action: str,
        slot: str = None,
        item: str = None,
        color: str = None,
        color_zone: str = "primary",
        outfit_name: str = None,
        **kwargs
    ) -> dict[str, Any]:
        try:
            from enigma_engine.avatar.outfit_system import OutfitManager, ClothingSlot, AccessorySlot, PropSlot
            
            # Get or create outfit manager
            manager = self._get_outfit_manager()
            if not manager:
                return {"success": False, "error": "Outfit system not available"}
            
            action = action.lower().strip()
            
            if action == "equip":
                if not slot or not item:
                    return {"success": False, "error": "Need slot and item for equip"}
                success = manager.equip_item(slot, item)
                return {"success": success, "action": "equipped", "slot": slot, "item": item}
            
            elif action == "unequip":
                if not slot:
                    return {"success": False, "error": "Need slot for unequip"}
                success = manager.unequip_slot(slot)
                return {"success": success, "action": "unequipped", "slot": slot}
            
            elif action == "set_color":
                if not color:
                    return {"success": False, "error": "Need color (hex like #ff5500)"}
                success = manager.set_color(color_zone, color)
                return {"success": success, "action": "colored", "zone": color_zone, "color": color}
            
            elif action == "list_outfits":
                outfits = manager.list_saved_outfits()
                return {"success": True, "outfits": outfits}
            
            elif action == "list_items":
                items = manager.list_available_items(slot)
                return {"success": True, "slot": slot, "items": items}
            
            elif action == "save_outfit":
                if not outfit_name:
                    return {"success": False, "error": "Need outfit_name to save"}
                success = manager.save_current_outfit(outfit_name)
                return {"success": success, "action": "saved", "name": outfit_name}
            
            elif action == "load_outfit":
                if not outfit_name:
                    return {"success": False, "error": "Need outfit_name to load"}
                success = manager.load_outfit(outfit_name)
                return {"success": success, "action": "loaded", "name": outfit_name}
            
            else:
                valid = ["equip", "unequip", "set_color", "list_outfits", "list_items", "save_outfit", "load_outfit"]
                return {"success": False, "error": f"Unknown action: {action}. Valid: {valid}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_outfit_manager(self):
        """Get the outfit manager instance."""
        try:
            from enigma_engine.avatar import get_avatar
            avatar = get_avatar()
            if avatar and hasattr(avatar, '_outfit_manager'):
                return avatar._outfit_manager
            
            # Try to create one
            from enigma_engine.avatar.outfit_system import OutfitManager
            return OutfitManager()
        except Exception:
            return None


class FullscreenModeControlTool(Tool):
    """
    Control visibility during fullscreen apps (games, videos, presentations).
    """
    
    name = "fullscreen_mode_control"
    description = "Control avatar and overlay visibility during fullscreen apps. Toggle categories, set hotkey, configure per-monitor display."
    parameters = {
        "action": "Action: show, hide, toggle, set_category, set_monitors, set_hotkey, get_status, enable, disable",
        "category": "Category for set_category: avatar, spawned_objects, effects, particles",
        "visible": "Whether category should be visible (true/false) - for set_category",
        "monitors": "List of monitor indices [0, 1, 2] or 'all' - for set_monitors",
        "hotkey": "Hotkey string like 'ctrl+shift+h' - for set_hotkey",
    }
    
    def execute(
        self,
        action: str,
        category: str = None,
        visible: bool = None,
        monitors: Any = None,
        hotkey: str = None,
        **kwargs
    ) -> dict[str, Any]:
        try:
            from enigma_engine.core.fullscreen_mode import get_fullscreen_controller
            
            controller = get_fullscreen_controller()
            action = action.lower().strip()
            
            if action == "show":
                controller.show_all()
                return {"success": True, "action": "shown", "visible": True}
            
            elif action == "hide":
                controller.hide_all()
                return {"success": True, "action": "hidden", "visible": False}
            
            elif action == "toggle":
                controller.toggle_visibility()
                return {"success": True, "action": "toggled", "visible": controller.is_visible}
            
            elif action == "set_category":
                if not category:
                    return {"success": False, "error": "Need category name"}
                if visible is None:
                    return {"success": False, "error": "Need visible (true/false)"}
                controller.set_category_visible(category, visible)
                return {"success": True, "category": category, "visible": visible}
            
            elif action == "set_monitors":
                if monitors == "all" or monitors is None:
                    controller.set_allowed_monitors(None)
                    return {"success": True, "monitors": "all"}
                elif isinstance(monitors, list):
                    controller.set_allowed_monitors(monitors)
                    return {"success": True, "monitors": monitors}
                else:
                    return {"success": False, "error": "monitors should be a list [0, 1, 2] or 'all'"}
            
            elif action == "set_hotkey":
                controller.set_toggle_hotkey(hotkey)
                return {"success": True, "hotkey": hotkey}
            
            elif action == "get_status":
                return {"success": True, **controller.get_status()}
            
            elif action == "enable":
                controller.enable()
                return {"success": True, "action": "enabled"}
            
            elif action == "disable":
                controller.disable()
                return {"success": True, "action": "disabled"}
            
            else:
                valid = ["show", "hide", "toggle", "set_category", "set_monitors", "set_hotkey", "get_status", "enable", "disable"]
                return {"success": False, "error": f"Unknown action: {action}. Valid: {valid}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
