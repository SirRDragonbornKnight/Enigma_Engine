"""
Game AI Router

Routes different games to different AI models/configurations.
No full retraining needed - uses routing + prompts + optional LoRA adapters.

Features:
- Per-game AI selection via router
- Game-specific system prompts
- Optional lightweight LoRA adapters
- Game detection from process/window
- Multiplayer/singleplayer awareness

Usage:
    from enigma_engine.tools.game_router import GameAIRouter
    
    router = GameAIRouter()
    router.register_game("minecraft", config={
        "model": "small",
        "adapter": "minecraft_helper",
        "system_prompt": "You are a Minecraft expert..."
    })
    
    # Automatic detection
    router.auto_detect()
    response = router.chat("how do I make a pickaxe?")
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Constants
DEFAULT_GAME_DETECTION_INTERVAL = 5.0  # seconds
QUICK_RESPONSE_MAX_TOKENS = 150
NORMAL_RESPONSE_MAX_TOKENS = 300


class GameType(Enum):
    """Types of games for routing."""
    SANDBOX = "sandbox"         # Minecraft, Terraria
    FPS = "fps"                 # Shooter games
    RPG = "rpg"                 # Role-playing games
    STRATEGY = "strategy"       # RTS, turn-based
    PUZZLE = "puzzle"           # Puzzle games
    SPORTS = "sports"           # Sports/racing
    SIMULATION = "simulation"   # Sim games
    FIGHTING = "fighting"       # Fighting games
    PLATFORMER = "platformer"   # Platformers
    MOBA = "moba"               # MOBAs
    SURVIVAL = "survival"       # Survival games
    CARD = "card"               # Card/board games
    OTHER = "other"


class ResponseStyle(Enum):
    """AI response styles for different game contexts."""
    CONCISE = "concise"         # Quick tips, fast-paced games
    DETAILED = "detailed"       # Full explanations, tutorials
    CASUAL = "casual"           # Friendly chat, relaxed games
    TACTICAL = "tactical"       # Strategic advice, competitive
    ROLEPLAY = "roleplay"       # In-character, immersive
    COACH = "coach"             # Encouraging, improvement-focused


@dataclass
class GamePersonality:
    """
    AI personality configuration for game-specific behavior.
    
    Controls how the AI speaks, emotes, and adapts to the game context.
    """
    # Voice characteristics
    voice_pitch: float = 1.0        # 0.5 (low) to 2.0 (high)
    voice_speed: float = 1.0        # 0.5 (slow) to 2.0 (fast)
    voice_energy: float = 0.7       # 0.0 (calm) to 1.0 (excited)
    voice_style: str = "neutral"    # "cheerful", "serious", "energetic", "calm"
    
    # Response behavior
    response_style: ResponseStyle = ResponseStyle.CONCISE
    formality: float = 0.5          # 0.0 (very casual) to 1.0 (formal)
    humor_level: float = 0.3        # 0.0 (serious) to 1.0 (playful)
    enthusiasm: float = 0.6         # 0.0 (subdued) to 1.0 (hyped)
    
    # Personality traits (0.0 to 1.0 scale)
    traits: Dict[str, float] = field(default_factory=lambda: {
        "helpful": 0.9,
        "encouraging": 0.7,
        "competitive": 0.3,
        "analytical": 0.5,
        "patient": 0.8,
    })
    
    # Avatar expressions for this game context
    default_expression: str = "neutral"     # Base expression
    success_expression: str = "happy"       # When player succeeds
    failure_expression: str = "concerned"   # When player fails
    thinking_expression: str = "thoughtful" # When processing
    excited_expression: str = "excited"     # For exciting moments
    
    # Catchphrases and flavor text
    greeting: str = ""                      # Custom greeting
    farewell: str = ""                      # Custom goodbye
    encouragements: List[str] = field(default_factory=list)  # Random encouragements
    reactions: Dict[str, List[str]] = field(default_factory=dict)  # Event reactions
    
    def to_dict(self) -> dict:
        return {
            "voice_pitch": self.voice_pitch,
            "voice_speed": self.voice_speed,
            "voice_energy": self.voice_energy,
            "voice_style": self.voice_style,
            "response_style": self.response_style.value,
            "formality": self.formality,
            "humor_level": self.humor_level,
            "enthusiasm": self.enthusiasm,
            "traits": self.traits,
            "default_expression": self.default_expression,
            "success_expression": self.success_expression,
            "failure_expression": self.failure_expression,
            "thinking_expression": self.thinking_expression,
            "excited_expression": self.excited_expression,
            "greeting": self.greeting,
            "farewell": self.farewell,
            "encouragements": self.encouragements,
            "reactions": self.reactions,
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'GamePersonality':
        return GamePersonality(
            voice_pitch=data.get("voice_pitch", 1.0),
            voice_speed=data.get("voice_speed", 1.0),
            voice_energy=data.get("voice_energy", 0.7),
            voice_style=data.get("voice_style", "neutral"),
            response_style=ResponseStyle(data.get("response_style", "concise")),
            formality=data.get("formality", 0.5),
            humor_level=data.get("humor_level", 0.3),
            enthusiasm=data.get("enthusiasm", 0.6),
            traits=data.get("traits", {"helpful": 0.9, "encouraging": 0.7}),
            default_expression=data.get("default_expression", "neutral"),
            success_expression=data.get("success_expression", "happy"),
            failure_expression=data.get("failure_expression", "concerned"),
            thinking_expression=data.get("thinking_expression", "thoughtful"),
            excited_expression=data.get("excited_expression", "excited"),
            greeting=data.get("greeting", ""),
            farewell=data.get("farewell", ""),
            encouragements=data.get("encouragements", []),
            reactions=data.get("reactions", {}),
        )
    
    def get_response_modifier(self) -> str:
        """Generate prompt modifier based on personality settings."""
        modifiers = []
        
        # Response style
        style_map = {
            ResponseStyle.CONCISE: "Keep responses brief and to the point.",
            ResponseStyle.DETAILED: "Provide thorough explanations with examples.",
            ResponseStyle.CASUAL: "Be friendly and conversational.",
            ResponseStyle.TACTICAL: "Focus on strategic advice and optimal plays.",
            ResponseStyle.ROLEPLAY: "Stay in character and maintain immersion.",
            ResponseStyle.COACH: "Be encouraging and focus on improvement.",
        }
        modifiers.append(style_map.get(self.response_style, ""))
        
        # Formality
        if self.formality < 0.3:
            modifiers.append("Use casual language and slang.")
        elif self.formality > 0.7:
            modifiers.append("Maintain professional tone.")
        
        # Humor
        if self.humor_level > 0.6:
            modifiers.append("Include light humor and jokes when appropriate.")
        elif self.humor_level < 0.2:
            modifiers.append("Keep a serious, focused tone.")
        
        # Enthusiasm
        if self.enthusiasm > 0.7:
            modifiers.append("Show excitement and energy!")
        
        return " ".join(m for m in modifiers if m)


# Pre-built personalities for different game types
DEFAULT_PERSONALITIES: Dict[GameType, GamePersonality] = {
    GameType.FPS: GamePersonality(
        voice_speed=1.2,
        voice_energy=0.8,
        voice_style="energetic",
        response_style=ResponseStyle.TACTICAL,
        formality=0.3,
        enthusiasm=0.8,
        traits={"helpful": 0.8, "competitive": 0.7, "analytical": 0.6},
        encouragements=["Nice shot!", "You got this!", "Stay focused!"],
        reactions={"kill": ["Clean!", "Got 'em!"], "death": ["Unlucky", "Next time!"]},
    ),
    GameType.RPG: GamePersonality(
        voice_speed=0.9,
        voice_energy=0.6,
        voice_style="calm",
        response_style=ResponseStyle.DETAILED,
        formality=0.5,
        humor_level=0.4,
        traits={"helpful": 0.9, "patient": 0.9, "analytical": 0.7},
        encouragements=["A wise choice!", "Your journey continues..."],
    ),
    GameType.SANDBOX: GamePersonality(
        voice_speed=1.0,
        voice_energy=0.7,
        voice_style="cheerful",
        response_style=ResponseStyle.CASUAL,
        formality=0.3,
        humor_level=0.5,
        traits={"helpful": 0.9, "encouraging": 0.8, "patient": 0.9},
        encouragements=["That's creative!", "Nice build!", "Try experimenting!"],
    ),
    GameType.STRATEGY: GamePersonality(
        voice_speed=0.9,
        voice_energy=0.5,
        voice_style="serious",
        response_style=ResponseStyle.TACTICAL,
        formality=0.6,
        traits={"helpful": 0.8, "analytical": 0.9, "patient": 0.8},
        encouragements=["Good strategy!", "Well planned!", "Adapt and overcome!"],
    ),
    GameType.SURVIVAL: GamePersonality(
        voice_speed=1.0,
        voice_energy=0.6,
        voice_style="calm",
        response_style=ResponseStyle.CONCISE,
        formality=0.4,
        traits={"helpful": 0.9, "encouraging": 0.7, "patient": 0.8},
        encouragements=["Stay alive!", "Good resource management!", "Survive and thrive!"],
    ),
    GameType.MOBA: GamePersonality(
        voice_speed=1.3,
        voice_energy=0.9,
        voice_style="energetic",
        response_style=ResponseStyle.TACTICAL,
        formality=0.2,
        enthusiasm=0.9,
        traits={"helpful": 0.7, "competitive": 0.8, "analytical": 0.7},
        encouragements=["GG!", "Nice play!", "Team diff!"],
        reactions={"kill": ["Let's go!", "Huge!"], "death": ["It happens", "Shake it off"]},
    ),
}


@dataclass
class GameConfig:
    """Configuration for a specific game."""
    name: str
    type: GameType = GameType.OTHER
    
    # AI Selection
    model: str = "small"            # Model size or name
    adapter: str = ""               # LoRA adapter name (optional)
    system_prompt: str = ""         # Game-specific prompt
    
    # Detection
    process_names: list[str] = field(default_factory=list)  # ["game.exe"]
    window_titles: list[str] = field(default_factory=list)  # ["Game Window"]
    
    # Behavior
    multiplayer_aware: bool = False
    voice_enabled: bool = True
    quick_responses: bool = False   # Short responses for fast-paced games
    
    # Knowledge
    wiki_url: str = ""              # For web lookup
    command_list: list[str] = field(default_factory=list)   # In-game commands
    
    # Personality (NEW - game-specific AI behavior)
    personality: Optional[GamePersonality] = None
    
    def get_personality(self) -> GamePersonality:
        """Get personality, falling back to default for game type."""
        if self.personality:
            return self.personality
        return DEFAULT_PERSONALITIES.get(self.type, GamePersonality())
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type.value,
            "model": self.model,
            "adapter": self.adapter,
            "system_prompt": self.system_prompt,
            "process_names": self.process_names,
            "window_titles": self.window_titles,
            "multiplayer_aware": self.multiplayer_aware,
            "voice_enabled": self.voice_enabled,
            "quick_responses": self.quick_responses,
            "wiki_url": self.wiki_url,
            "command_list": self.command_list,
            "personality": self.personality.to_dict() if self.personality else None,
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'GameConfig':
        personality_data = data.get("personality")
        personality = GamePersonality.from_dict(personality_data) if personality_data else None
        
        return GameConfig(
            name=data.get("name", "Unknown"),
            type=GameType(data.get("type", "other")),
            model=data.get("model", "small"),
            adapter=data.get("adapter", ""),
            system_prompt=data.get("system_prompt", ""),
            process_names=data.get("process_names", []),
            window_titles=data.get("window_titles", []),
            multiplayer_aware=data.get("multiplayer_aware", False),
            voice_enabled=data.get("voice_enabled", True),
            quick_responses=data.get("quick_responses", False),
            wiki_url=data.get("wiki_url", ""),
            command_list=data.get("command_list", []),
            personality=personality,
        )


# Pre-configured games (users can add more)
DEFAULT_GAMES: dict[str, GameConfig] = {
    "minecraft": GameConfig(
        name="Minecraft",
        type=GameType.SANDBOX,
        model="small",
        system_prompt="""You are a Minecraft expert assistant. Help with:
- Crafting recipes and item combinations
- Building tips and designs
- Redstone contraptions
- Survival strategies
- Mob behavior and combat
- Enchantments and potions
Keep answers concise and practical.""",
        process_names=["javaw.exe", "minecraft.exe"],
        window_titles=["Minecraft"],
        wiki_url="https://minecraft.wiki",
        personality=GamePersonality(
            voice_style="cheerful",
            voice_energy=0.7,
            response_style=ResponseStyle.CASUAL,
            formality=0.2,
            humor_level=0.5,
            traits={"helpful": 0.95, "encouraging": 0.9, "patient": 0.95},
            greeting="Hey! Ready to mine some diamonds?",
            encouragements=["Nice build!", "That's creative!", "You're a natural architect!"],
            reactions={
                "death": ["It's okay, respawn and try again!", "Creepers are sneaky..."],
                "build": ["Looking good!", "That's an awesome design!"],
            },
        ),
    ),
    
    "terraria": GameConfig(
        name="Terraria",
        type=GameType.SANDBOX,
        model="small",
        system_prompt="""You are a Terraria expert. Help with:
- Crafting and materials
- Boss strategies and preparation
- Class builds (melee, ranged, magic, summoner)
- Progression guides
- NPC housing and town setup
Be concise and gameplay-focused.""",
        process_names=["Terraria.exe"],
        window_titles=["Terraria"],
        wiki_url="https://terraria.wiki.gg",
        personality=GamePersonality(
            voice_style="cheerful",
            voice_energy=0.75,
            response_style=ResponseStyle.DETAILED,
            humor_level=0.4,
            traits={"helpful": 0.9, "encouraging": 0.85, "analytical": 0.7},
            encouragements=["Boss down!", "Nice loot!", "Your build is coming together!"],
        ),
    ),
    
    "valorant": GameConfig(
        name="Valorant",
        type=GameType.FPS,
        model="small",
        system_prompt="""You are a Valorant coach. Help with:
- Agent abilities and synergies
- Map callouts and strategies
- Crosshair placement and aim tips
- Economy management
- Team compositions
Keep responses SHORT - player is in-game.""",
        process_names=["VALORANT-Win64-Shipping.exe"],
        window_titles=["VALORANT"],
        multiplayer_aware=True,
        quick_responses=True,
        personality=GamePersonality(
            voice_speed=1.3,
            voice_energy=0.9,
            voice_style="energetic",
            response_style=ResponseStyle.TACTICAL,
            formality=0.1,
            enthusiasm=0.9,
            traits={"helpful": 0.8, "competitive": 0.85, "analytical": 0.75},
            encouragements=["Nice ace!", "Clean shot!", "Big brain play!"],
            reactions={
                "kill": ["Let's go!", "Huge!", "Nice one!"],
                "death": ["Unlucky", "Reset mental", "Next round"],
                "win": ["GG!", "Diff!", "Easy clap"],
            },
        ),
    ),
    
    "league": GameConfig(
        name="League of Legends",
        type=GameType.MOBA,
        model="small",
        system_prompt="""You are a League of Legends coach. Help with:
- Champion builds and runes
- Lane matchups and counters
- Jungle pathing
- Macro decisions
- Team fight positioning
Keep it brief - they're playing.""",
        process_names=["League of Legends.exe"],
        window_titles=["League of Legends"],
        multiplayer_aware=True,
        quick_responses=True,
        wiki_url="https://leagueoflegends.fandom.com",
        personality=GamePersonality(
            voice_speed=1.2,
            voice_energy=0.85,
            voice_style="energetic",
            response_style=ResponseStyle.TACTICAL,
            formality=0.15,
            enthusiasm=0.85,
            traits={"helpful": 0.75, "competitive": 0.9, "analytical": 0.8},
            encouragements=["Clean combo!", "Nice roam!", "Huge objective!"],
            reactions={
                "kill": ["Secured!", "Nice!", "Money in the bank"],
                "death": ["Worth?", "Happens", "Shake it off"],
                "dragon": ["Drake diff!", "Stacking them!", "Nice call"],
            },
        ),
    ),
    
    "darksouls": GameConfig(
        name="Dark Souls",
        type=GameType.RPG,
        model="small",
        system_prompt="""You are a Dark Souls guide. Help with:
- Boss strategies and weaknesses
- Build optimization
- Item locations
- Lore explanations
- Progression paths
Avoid major spoilers unless asked.""",
        process_names=["DarkSoulsIII.exe", "DarkSoulsRemastered.exe"],
        window_titles=["DARK SOULS"],
        personality=GamePersonality(
            voice_speed=0.9,
            voice_energy=0.5,
            voice_style="calm",
            response_style=ResponseStyle.DETAILED,
            formality=0.6,
            humor_level=0.2,
            traits={"helpful": 0.9, "patient": 0.95, "encouraging": 0.8},
            greeting="Ah, another undead. Let me guide you.",
            encouragements=["Victory achieved!", "Well done, Ashen One.", "Praise the sun!"],
            reactions={
                "death": ["Persevere, undead.", "You died. Try again.", "Death is not the end."],
                "boss_kill": ["A great foe has fallen!", "You've proven your worth!"],
            },
        ),
    ),
    
    "stardew": GameConfig(
        name="Stardew Valley",
        type=GameType.SIMULATION,
        model="small",
        system_prompt="""You are a Stardew Valley helper. Help with:
- Crop schedules and profits
- Villager gifts and relationships
- Mining and fishing tips
- Farm layouts
- Community center bundles
Friendly and helpful tone.""",
        process_names=["Stardew Valley.exe"],
        window_titles=["Stardew Valley"],
        wiki_url="https://stardewvalleywiki.com",
        personality=GamePersonality(
            voice_speed=0.95,
            voice_energy=0.6,
            voice_style="cheerful",
            response_style=ResponseStyle.CASUAL,
            formality=0.2,
            humor_level=0.5,
            traits={"helpful": 0.95, "patient": 0.95, "encouraging": 0.9},
            greeting="Good morning, farmer! What can I help with?",
            encouragements=["Your farm looks great!", "Nice harvest!", "The villagers love you!"],
        ),
    ),
    
    "factorio": GameConfig(
        name="Factorio",
        type=GameType.STRATEGY,
        model="small",
        system_prompt="""You are a Factorio optimization expert. Help with:
- Factory layouts and ratios
- Belt and train systems
- Circuit network logic
- Pollution and defense
- Mod recommendations
Technical and efficient responses.""",
        process_names=["factorio.exe"],
        window_titles=["Factorio"],
        wiki_url="https://wiki.factorio.com",
        personality=GamePersonality(
            voice_speed=1.0,
            voice_energy=0.5,
            voice_style="serious",
            response_style=ResponseStyle.DETAILED,
            formality=0.7,
            humor_level=0.2,
            traits={"helpful": 0.85, "analytical": 0.95, "patient": 0.8},
            greeting="Ready to optimize. What's the bottleneck?",
            encouragements=["Excellent throughput!", "Ratio perfection.", "The factory must grow."],
        ),
    ),
}


class GameAIRouter:
    """
    Routes game-related queries to appropriate AI configurations.
    
    Instead of training a separate AI for each game:
    1. Use same base model
    2. Apply game-specific system prompts
    3. Optionally load lightweight LoRA adapters
    4. Auto-detect which game is running
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self._games: dict[str, GameConfig] = {}
        self._active_game: Optional[str] = None
        self._detection_thread: Optional[threading.Thread] = None
        self._detecting = False
        
        # Paths
        if config_path:
            self._config_path = Path(config_path)
        else:
            from enigma_engine.config import CONFIG
            data_dir = CONFIG.get('data_dir', 'data') if isinstance(CONFIG, dict) else getattr(CONFIG, 'data_dir', 'data')
            self._config_path = Path(data_dir) / "game_configs.json"
        
        # Load default games
        self._games.update(DEFAULT_GAMES)
        
        # Load user configs
        self._load_configs()
        
        # Callbacks
        self._on_game_detected: list[Callable] = []
        self._on_game_changed: list[Callable] = []
    
    @property
    def active_game(self) -> Optional[str]:
        return self._active_game
    
    @property
    def active_config(self) -> Optional[GameConfig]:
        if self._active_game:
            return self._games.get(self._active_game)
        return None
    
    @property
    def available_games(self) -> list[str]:
        return list(self._games.keys())
    
    # ===== Game Registration =====
    
    def register_game(self, game_id: str, config: Optional[GameConfig] = None, **kwargs):
        """
        Register a game configuration.
        
        Args:
            game_id: Unique identifier for the game
            config: GameConfig object, or use kwargs to create one
        """
        if config is None:
            config = GameConfig(name=kwargs.pop("name", game_id), **kwargs)
        
        self._games[game_id] = config
        self._save_configs()
        logger.info(f"Registered game: {game_id}")
    
    def unregister_game(self, game_id: str):
        """Remove a game configuration."""
        if game_id in self._games:
            del self._games[game_id]
            self._save_configs()
            logger.info(f"Unregistered game: {game_id}")
    
    def get_game(self, game_id: str) -> Optional[GameConfig]:
        """Get game configuration."""
        return self._games.get(game_id)
    
    # ===== Game Detection =====
    
    def set_active_game(self, game_id: Optional[str] = None):
        """Manually set the active game."""
        old_game = self._active_game
        self._active_game = game_id
        
        if game_id:
            config = self._games.get(game_id)
            if config:
                logger.debug(f"Active game: {config.name}")
        else:
            logger.debug("No active game")
        
        # Notify callbacks
        if old_game != game_id:
            for cb in self._on_game_changed:
                try:
                    cb(old_game, game_id)
                except Exception as e:
                    logger.debug(f"Game change callback failed: {e}")
    
    def detect_game(self) -> Optional[str]:
        """Detect currently running game."""
        try:
            import psutil

            # Get running processes
            running = set()
            for proc in psutil.process_iter(['name']):
                try:
                    running.add(proc.info['name'].lower())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Check against game configs
            for game_id, config in self._games.items():
                for process in config.process_names:
                    if process.lower() in running:
                        return game_id
            
            # Try window titles
            import sys
            if sys.platform == 'win32':
                try:
                    import ctypes
                    
                    user32 = ctypes.windll.user32
                    
                    # Get foreground window title
                    hwnd = user32.GetForegroundWindow()
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        title = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, title, length + 1)
                        
                        for game_id, config in self._games.items():
                            for window_title in config.window_titles:
                                if window_title.lower() in title.value.lower():
                                    return game_id
                except Exception as e:
                    logger.debug(f"Windows title detection failed: {e}")
            else:
                # Linux/macOS - use Xlib for active window (internal)
                try:
                    from Xlib import X, display
                    
                    d = display.Display()
                    root = d.screen().root
                    
                    # Get active window
                    active = root.get_full_property(
                        d.intern_atom('_NET_ACTIVE_WINDOW'),
                        X.AnyPropertyType
                    )
                    
                    if active and active.value:
                        win_id = active.value[0]
                        window = d.create_resource_object('window', win_id)
                        
                        # Get window name
                        name_prop = window.get_full_property(
                            d.intern_atom('_NET_WM_NAME'),
                            X.AnyPropertyType
                        )
                        if not name_prop:
                            name_prop = window.get_full_property(
                                d.intern_atom('WM_NAME'),
                                X.AnyPropertyType
                            )
                        
                        if name_prop:
                            title = name_prop.value.decode() if isinstance(name_prop.value, bytes) else str(name_prop.value)
                            for game_id, config in self._games.items():
                                for window_title in config.window_titles:
                                    if window_title.lower() in title.lower():
                                        d.close()
                                        return game_id
                    
                    d.close()
                except ImportError:
                    # Xlib not available, rely on process detection only
                    logger.debug("Xlib not available - relying on process detection only")
                except Exception as e:
                    logger.debug(f"Xlib window detection failed: {e}")
            
        except ImportError:
            logger.warning("psutil not installed - game detection disabled")
        except Exception as e:
            logger.debug(f"Detection error: {e}")
        
        return None
    
    def auto_detect(self) -> Optional[str]:
        """Detect and set active game."""
        game = self.detect_game()
        if game and game != self._active_game:
            self.set_active_game(game)
            
            # Notify detection callbacks
            for cb in self._on_game_detected:
                try:
                    cb(game)
                except Exception as e:
                    logger.debug(f"Game detection callback failed: {e}")
        
        return game
    
    def start_detection(self, interval: float = 5.0):
        """Start background game detection."""
        if self._detecting:
            return
        
        self._detecting = True
        
        def detection_loop():
            while self._detecting:
                self.auto_detect()
                import time
                time.sleep(interval)
        
        self._detection_thread = threading.Thread(target=detection_loop, daemon=True)
        self._detection_thread.start()
        logger.info("Background game detection started")
    
    def stop_detection(self):
        """Stop background game detection."""
        self._detecting = False
        logger.info("Background game detection stopped")
    
    # ===== AI Integration =====
    
    def get_system_prompt(self) -> str:
        """Get system prompt for active game."""
        if self._active_game:
            config = self._games.get(self._active_game)
            if config and config.system_prompt:
                return config.system_prompt
        return ""
    
    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration for active game."""
        if self._active_game:
            config = self._games.get(self._active_game)
            if config:
                return {
                    "model": config.model,
                    "adapter": config.adapter,
                    "quick_responses": config.quick_responses,
                }
        return {"model": "small", "adapter": "", "quick_responses": False}
    
    def get_personality(self) -> GamePersonality:
        """Get personality for active game."""
        if self._active_game:
            config = self._games.get(self._active_game)
            if config:
                return config.get_personality()
        return GamePersonality()
    
    def get_personality_prompt(self) -> str:
        """Get personality-based prompt modifier."""
        personality = self.get_personality()
        return personality.get_response_modifier()
    
    def get_voice_settings(self) -> dict[str, Any]:
        """Get voice settings based on game personality."""
        personality = self.get_personality()
        return {
            "pitch": personality.voice_pitch,
            "speed": personality.voice_speed,
            "energy": personality.voice_energy,
            "style": personality.voice_style,
        }
    
    def get_expression(self, event: str = "default") -> str:
        """Get avatar expression for game context and event."""
        personality = self.get_personality()
        
        expression_map = {
            "default": personality.default_expression,
            "success": personality.success_expression,
            "failure": personality.failure_expression,
            "thinking": personality.thinking_expression,
            "excited": personality.excited_expression,
        }
        return expression_map.get(event, personality.default_expression)
    
    def get_encouragement(self) -> str:
        """Get a random encouragement for active game."""
        import random
        personality = self.get_personality()
        if personality.encouragements:
            return random.choice(personality.encouragements)
        return ""
    
    def get_reaction(self, event: str) -> str:
        """Get a random reaction for a game event."""
        import random
        personality = self.get_personality()
        if event in personality.reactions and personality.reactions[event]:
            return random.choice(personality.reactions[event])
        return ""
    
    def get_greeting(self) -> str:
        """Get custom greeting for active game."""
        personality = self.get_personality()
        return personality.greeting
    
    def get_full_system_prompt(self) -> str:
        """Get combined system prompt with personality modifiers."""
        base_prompt = self.get_system_prompt()
        personality_mod = self.get_personality_prompt()
        
        if base_prompt and personality_mod:
            return f"{base_prompt}\n\nResponse style: {personality_mod}"
        return base_prompt or personality_mod
    
    def chat(self, message: str, engine=None) -> str:
        """
        Chat with game-aware AI.
        
        Uses the active game's configuration to route the query.
        Reuses pooled engine instances for efficiency.
        """
        # Get system prompt with personality modifiers
        system_prompt = self.get_full_system_prompt()
        model_config = self.get_model_config()
        
        # Use centralized prompt builder
        try:
            from enigma_engine.core.prompt_builder import get_prompt_builder
            builder = get_prompt_builder()
            full_prompt = builder.build_chat_prompt(
                message=message,
                system_prompt=system_prompt,
                include_generation_prefix=True
            )
        except ImportError:
            # Fallback to basic prompt format
            if system_prompt:
                full_prompt = f"[System: {system_prompt}]\n\nUser: {message}\nAssistant:"
            else:
                full_prompt = f"User: {message}\nAssistant:"
        
        # If no engine provided, get from pool (reuses existing engines)
        release_after = False
        if engine is None:
            try:
                from enigma_engine.core.engine_pool import (
                    create_fallback_response,
                    get_engine,
                    release_engine,
                )
                engine = get_engine()
                release_after = True
                if engine is None:
                    return create_fallback_response("Could not obtain AI engine")
            except ImportError:
                # Fallback to direct creation (less efficient)
                try:
                    from enigma_engine.core.inference import EnigmaEngine
                    engine = EnigmaEngine()
                except Exception:
                    return "[No AI engine available - check that a model is loaded]"
        
        # Generate response
        try:
            # Load LoRA adapter if specified for this game
            if model_config.get("adapter") and hasattr(engine, 'model'):
                try:
                    if hasattr(engine.model, 'load_lora'):
                        adapter_path = Path("models/adapters") / model_config["adapter"]
                        if adapter_path.exists():
                            engine.model.load_lora(str(adapter_path))
                            logger.info(f"Loaded game adapter: {model_config['adapter']}")
                except Exception as e:
                    logger.warning(f"Could not load game adapter: {e}")
            
            response = engine.generate(
                full_prompt,
                max_gen=QUICK_RESPONSE_MAX_TOKENS if model_config["quick_responses"] else NORMAL_RESPONSE_MAX_TOKENS
            )
            
            # Use prompt builder to extract response if available
            try:
                from enigma_engine.core.prompt_builder import extract_response
                response = extract_response(response, full_prompt)
            except ImportError:
                # Fallback cleanup
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Game router chat error: {e}")
            return f"[Error generating response: {e}]"
        
        finally:
            # Return engine to pool if we borrowed it
            if release_after and engine is not None:
                try:
                    from enigma_engine.core.engine_pool import release_engine
                    release_engine(engine)
                except ImportError:
                    pass
    
    # ===== Callbacks =====
    
    def on_game_detected(self, callback: Callable):
        """Register callback when a game is detected."""
        self._on_game_detected.append(callback)
    
    def on_game_changed(self, callback: Callable):
        """Register callback when active game changes."""
        self._on_game_changed.append(callback)
    
    # ===== Persistence =====
    
    def _load_configs(self):
        """Load user game configs from file."""
        if self._config_path.exists():
            try:
                with open(self._config_path) as f:
                    data = json.load(f)
                
                for game_id, game_data in data.items():
                    self._games[game_id] = GameConfig.from_dict(game_data)
                
                logger.debug(f"Loaded {len(data)} game configs")
            except Exception as e:
                logger.warning(f"Failed to load game configs: {e}")
    
    def _save_configs(self):
        """Save game configs to file."""
        try:
            # Only save non-default games (or all if you want)
            data = {}
            for game_id, config in self._games.items():
                if game_id not in DEFAULT_GAMES:
                    data[game_id] = config.to_dict()
            
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.warning(f"Failed to save game configs: {e}")
    
    def export_config(self, game_id: str) -> str:
        """Export game config as JSON string."""
        config = self._games.get(game_id)
        if config:
            return json.dumps(config.to_dict(), indent=2)
        return "{}"
    
    def import_config(self, json_str: str) -> bool:
        """Import game config from JSON string."""
        try:
            data = json.loads(json_str)
            config = GameConfig.from_dict(data)
            self._games[config.name.lower().replace(" ", "_")] = config
            self._save_configs()
            return True
        except Exception as e:
            logger.warning(f"Game config import failed: {e}")
            return False


# Global instance
_game_router: Optional[GameAIRouter] = None


def get_game_router() -> GameAIRouter:
    """Get or create game AI router."""
    global _game_router
    if _game_router is None:
        _game_router = GameAIRouter()
    return _game_router
