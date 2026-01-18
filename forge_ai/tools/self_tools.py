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
from pathlib import Path
from typing import Dict, Any
from .tool_registry import Tool


# Storage for AI's self-configuration
SELF_CONFIG_PATH = Path(__file__).parent.parent.parent / "data" / "ai_self_config.json"


def _load_self_config() -> Dict[str, Any]:
    """Load AI's self-configuration."""
    if SELF_CONFIG_PATH.exists():
        try:
            return json.loads(SELF_CONFIG_PATH.read_text())
        except Exception:
            pass
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


def _save_self_config(config: Dict[str, Any]):
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
    
    def execute(self, trait: str, value: str, **kwargs) -> Dict[str, Any]:
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
    
    def execute(self, setting: str, value: str, **kwargs) -> Dict[str, Any]:
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
    
    def execute(self, setting: str, value: str, **kwargs) -> Dict[str, Any]:
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
    
    def execute(self, setting: str, value: str, **kwargs) -> Dict[str, Any]:
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
                from forge_ai.companion import get_companion
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
                    from forge_ai.companion import get_companion
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
    
    def execute(self, preference: str, value: str, **kwargs) -> Dict[str, Any]:
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
    
    def execute(self, section: str = "all", **kwargs) -> Dict[str, Any]:
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
    
    def execute(self, category: str, fact: str, **kwargs) -> Dict[str, Any]:
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
    
    def execute(self, category: str = "all", **kwargs) -> Dict[str, Any]:
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
        "style": "Style: realistic, cartoon, robot, creature, abstract (default: robot)",
    }
    
    AVATAR_DIR = Path(__file__).parent.parent.parent / "data" / "avatar" / "generated"
    
    def execute(self, description: str, style: str = "robot", **kwargs) -> Dict[str, Any]:
        self.AVATAR_DIR.mkdir(parents=True, exist_ok=True)
        
        # Enhance prompt based on style
        style_prefixes = {
            "realistic": "highly detailed realistic 3D model of",
            "cartoon": "stylized cartoon 3D character of",
            "robot": "sleek robotic 3D avatar, mechanical, sci-fi,",
            "creature": "fantasy creature 3D model of",
            "abstract": "abstract geometric 3D representation of",
        }
        full_prompt = f"{style_prefixes.get(style, style_prefixes['robot'])} {description}"
        
        # Try to use the 3D generation system
        try:
            from ..gui.tabs.threed_tab import Local3DGen, Cloud3DGen, OUTPUT_DIR
            import time
            
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
    
    def execute(self, file_path: str = "", **kwargs) -> Dict[str, Any]:
        import subprocess
        import shutil
        
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
    
    def execute(self, **kwargs) -> Dict[str, Any]:
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
    
    def execute(self, file_path: str, **kwargs) -> Dict[str, Any]:
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

