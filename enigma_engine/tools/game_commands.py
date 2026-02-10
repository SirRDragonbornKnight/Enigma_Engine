"""
Quick Game Commands for Enigma AI Engine

Shortcut commands for game-specific actions.

Features:
- Command prefix system (!command)
- Game-specific command sets
- Customizable bindings
- Macro/combo support
- Voice command integration

Usage:
    from enigma_engine.tools.game_commands import GameCommands, register_command
    
    commands = GameCommands()
    
    # Register a game command
    @register_command("build", game="minecraft")
    def build_house(style="simple"):
        '''Build a house'''
        return execute_building_sequence(style)
    
    # Execute commands
    commands.execute("!build house")
    commands.execute("!attack")
    commands.execute("!heal", game="rpg")
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CommandCategory(Enum):
    """Categories of game commands."""
    MOVEMENT = auto()
    COMBAT = auto()
    BUILDING = auto()
    INVENTORY = auto()
    COMMUNICATION = auto()
    UTILITY = auto()
    MACRO = auto()
    CUSTOM = auto()


@dataclass
class GameCommand:
    """Definition of a game command."""
    # Command name (what user types after !)
    name: str
    
    # Description
    description: str
    
    # Function to execute
    handler: Callable
    
    # Category
    category: CommandCategory = CommandCategory.CUSTOM
    
    # Which games this works in (None = all)
    games: Optional[List[str]] = None
    
    # Aliases
    aliases: List[str] = field(default_factory=list)
    
    # Arguments
    args: List[str] = field(default_factory=list)
    
    # Whether this is a macro (sequence of actions)
    is_macro: bool = False
    
    # Cooldown in seconds
    cooldown: float = 0.0
    
    # Last execution time
    _last_executed: float = 0.0


@dataclass
class CommandResult:
    """Result of executing a command."""
    success: bool
    message: str = ""
    actions_taken: List[str] = field(default_factory=list)
    cooldown_remaining: float = 0.0


# Global command registry
_commands: Dict[str, GameCommand] = {}
_game_commands: Dict[str, Dict[str, GameCommand]] = {}  # game -> {name: command}


def register_command(
    name: str,
    description: str = "",
    category: CommandCategory = CommandCategory.CUSTOM,
    games: Optional[List[str]] = None,
    aliases: Optional[List[str]] = None,
    cooldown: float = 0.0
) -> Callable:
    """
    Decorator to register a game command.
    
    Args:
        name: Command name
        description: Command description
        category: Command category
        games: List of games this applies to (None for all)
        aliases: Alternative names
        cooldown: Cooldown between uses
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        cmd = GameCommand(
            name=name.lower(),
            description=description or func.__doc__ or "",
            handler=func,
            category=category,
            games=games,
            aliases=aliases or [],
            cooldown=cooldown
        )
        
        # Register globally
        _commands[name.lower()] = cmd
        for alias in cmd.aliases:
            _commands[alias.lower()] = cmd
        
        # Register per-game
        if games:
            for game in games:
                if game not in _game_commands:
                    _game_commands[game] = {}
                _game_commands[game][name.lower()] = cmd
        
        logger.debug(f"Registered command: {name}")
        return func
    
    return decorator


class GameCommands:
    """
    Game command manager.
    
    Handles parsing and executing game commands.
    """
    
    def __init__(self, prefix: str = "!"):
        """
        Initialize game commands.
        
        Args:
            prefix: Command prefix (default "!")
        """
        self._prefix = prefix
        self._current_game: Optional[str] = None
        self._command_history: List[Tuple[float, str, CommandResult]] = []
        
        # Register built-in commands
        self._register_builtin_commands()
        
        logger.info("GameCommands initialized")
    
    def set_game(self, game: str):
        """Set the current game context."""
        self._current_game = game.lower()
        logger.info(f"Game context set to: {game}")
    
    def execute(
        self,
        input_text: str,
        game: Optional[str] = None
    ) -> CommandResult:
        """
        Execute a command from text input.
        
        Args:
            input_text: User input (e.g., "!build house")
            game: Optional game override
            
        Returns:
            CommandResult
        """
        game = game or self._current_game
        
        # Check for prefix
        if not input_text.startswith(self._prefix):
            return CommandResult(
                success=False,
                message=f"Not a command (missing {self._prefix} prefix)"
            )
        
        # Parse command
        text = input_text[len(self._prefix):].strip()
        parts = text.split(maxsplit=1)
        
        if not parts:
            return CommandResult(success=False, message="Empty command")
        
        cmd_name = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""
        
        # Find command
        cmd = self._find_command(cmd_name, game)
        
        if not cmd:
            return CommandResult(
                success=False,
                message=f"Unknown command: {cmd_name}"
            )
        
        # Check cooldown
        now = time.time()
        if cmd.cooldown > 0:
            elapsed = now - cmd._last_executed
            if elapsed < cmd.cooldown:
                remaining = cmd.cooldown - elapsed
                return CommandResult(
                    success=False,
                    message=f"Command on cooldown",
                    cooldown_remaining=remaining
                )
        
        # Parse arguments
        args, kwargs = self._parse_args(args_str)
        
        # Execute
        try:
            result = cmd.handler(*args, **kwargs)
            cmd._last_executed = now
            
            cmd_result = CommandResult(
                success=True,
                message=str(result) if result else "Command executed",
                actions_taken=[cmd_name]
            )
            
        except Exception as e:
            logger.error(f"Command error: {e}")
            cmd_result = CommandResult(
                success=False,
                message=f"Error: {e}"
            )
        
        # Record history
        self._command_history.append((now, input_text, cmd_result))
        
        return cmd_result
    
    def _find_command(
        self,
        name: str,
        game: Optional[str]
    ) -> Optional[GameCommand]:
        """Find a command by name, considering game context."""
        # Check game-specific first
        if game and game.lower() in _game_commands:
            game_cmds = _game_commands[game.lower()]
            if name in game_cmds:
                return game_cmds[name]
        
        # Check global commands
        if name in _commands:
            cmd = _commands[name]
            # If command has game restrictions, check them
            if cmd.games and game and game.lower() not in [g.lower() for g in cmd.games]:
                return None
            return cmd
        
        return None
    
    def _parse_args(self, args_str: str) -> Tuple[List[Any], Dict[str, Any]]:
        """Parse argument string into args and kwargs."""
        args = []
        kwargs = {}
        
        if not args_str:
            return args, kwargs
        
        # Simple parsing: split by spaces, key=value for kwargs
        parts = args_str.split()
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                # Try to parse value
                kwargs[key] = self._parse_value(value)
            else:
                args.append(self._parse_value(part))
        
        return args, kwargs
    
    def _parse_value(self, value: str) -> Any:
        """Parse a string value into appropriate type."""
        # Try int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try bool
        if value.lower() in ('true', 'yes', 'on'):
            return True
        if value.lower() in ('false', 'no', 'off'):
            return False
        
        # Return as string
        return value
    
    def list_commands(
        self,
        game: Optional[str] = None,
        category: Optional[CommandCategory] = None
    ) -> List[Dict[str, Any]]:
        """
        List available commands.
        
        Args:
            game: Filter by game
            category: Filter by category
            
        Returns:
            List of command info dicts
        """
        results = []
        
        # Get all commands
        all_cmds = set(_commands.values())
        if game and game.lower() in _game_commands:
            all_cmds.update(_game_commands[game.lower()].values())
        
        for cmd in all_cmds:
            # Filter by game
            if game and cmd.games:
                if game.lower() not in [g.lower() for g in cmd.games]:
                    continue
            
            # Filter by category
            if category and cmd.category != category:
                continue
            
            results.append({
                'name': cmd.name,
                'description': cmd.description,
                'category': cmd.category.name,
                'games': cmd.games,
                'aliases': cmd.aliases,
                'cooldown': cmd.cooldown
            })
        
        return sorted(results, key=lambda x: x['name'])
    
    def create_macro(
        self,
        name: str,
        commands: List[str],
        delay: float = 0.1
    ):
        """
        Create a macro from multiple commands.
        
        Args:
            name: Macro name
            commands: List of commands to execute
            delay: Delay between commands
        """
        def macro_handler():
            results = []
            for cmd in commands:
                result = self.execute(f"{self._prefix}{cmd}")
                results.append(f"{cmd}: {'OK' if result.success else 'FAIL'}")
                time.sleep(delay)
            return "\n".join(results)
        
        register_command(
            name,
            description=f"Macro: {' -> '.join(commands)}",
            category=CommandCategory.MACRO
        )(macro_handler)
    
    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get command execution history."""
        return [
            {
                'time': h[0],
                'command': h[1],
                'success': h[2].success,
                'message': h[2].message
            }
            for h in self._command_history[-limit:]
        ]
    
    def _register_builtin_commands(self):
        """Register built-in commands."""
        
        # --- Universal Commands ---
        
        @register_command("help", description="Show available commands")
        def cmd_help(category: str = None):
            cat = CommandCategory[category.upper()] if category else None
            cmds = self.list_commands(self._current_game, cat)
            lines = [f"!{c['name']}: {c['description']}" for c in cmds[:20]]
            return "\n".join(lines)
        
        @register_command("ping", description="Check if commands are working")
        def cmd_ping():
            return "Pong!"
        
        # --- Minecraft Commands ---
        
        @register_command(
            "build",
            description="Build a structure",
            category=CommandCategory.BUILDING,
            games=["minecraft"]
        )
        def cmd_build(structure: str = "house", material: str = "wood"):
            return f"Building {material} {structure}..."
        
        @register_command(
            "mine",
            description="Start mining",
            category=CommandCategory.UTILITY,
            games=["minecraft"]
        )
        def cmd_mine(target: str = "stone"):
            return f"Mining {target}..."
        
        @register_command(
            "craft",
            description="Craft an item",
            category=CommandCategory.UTILITY,
            games=["minecraft"]
        )
        def cmd_craft(item: str):
            return f"Crafting {item}..."
        
        # --- FPS Commands ---
        
        @register_command(
            "attack",
            description="Attack/shoot",
            category=CommandCategory.COMBAT,
            games=["fps", "valorant", "cs2", "apex", "overwatch"],
            cooldown=0.1
        )
        def cmd_attack():
            return "Attacking!"
        
        @register_command(
            "reload",
            description="Reload weapon",
            category=CommandCategory.COMBAT,
            games=["fps", "valorant", "cs2", "apex"],
            cooldown=2.0
        )
        def cmd_reload():
            return "Reloading..."
        
        @register_command(
            "ability",
            description="Use ability",
            category=CommandCategory.COMBAT,
            games=["valorant", "apex", "overwatch", "lol"],
            aliases=["skill", "spell"]
        )
        def cmd_ability(slot: int = 1):
            return f"Using ability {slot}!"
        
        # --- RPG/MOBA Commands ---
        
        @register_command(
            "heal",
            description="Use healing",
            category=CommandCategory.UTILITY,
            games=["rpg", "mmo", "lol", "dota"],
            cooldown=5.0
        )
        def cmd_heal():
            return "Healing!"
        
        @register_command(
            "recall",
            description="Return to base",
            category=CommandCategory.MOVEMENT,
            games=["lol"],
            aliases=["back", "b"]
        )
        def cmd_recall():
            return "Recalling to base..."
        
        @register_command(
            "ward",
            description="Place a ward",
            category=CommandCategory.UTILITY,
            games=["lol", "dota"],
            cooldown=30.0
        )
        def cmd_ward():
            return "Placing ward!"
        
        @register_command(
            "ult",
            description="Use ultimate ability",
            category=CommandCategory.COMBAT,
            games=["lol", "dota", "overwatch", "valorant"],
            aliases=["ultimate", "r"]
        )
        def cmd_ult():
            return "Ultimate activated!"
        
        # --- Movement ---
        
        @register_command(
            "jump",
            description="Jump",
            category=CommandCategory.MOVEMENT
        )
        def cmd_jump():
            return "Jumping!"
        
        @register_command(
            "sprint",
            description="Start sprinting",
            category=CommandCategory.MOVEMENT,
            aliases=["run"]
        )
        def cmd_sprint():
            return "Sprinting!"
        
        @register_command(
            "crouch",
            description="Crouch/stealth",
            category=CommandCategory.MOVEMENT,
            aliases=["sneak"]
        )
        def cmd_crouch():
            return "Crouching!"
        
        # --- Communication ---
        
        @register_command(
            "ping_loc",
            description="Ping a location",
            category=CommandCategory.COMMUNICATION,
            aliases=["ping_location", "mark"]
        )
        def cmd_ping_loc(loc: str = "here"):
            return f"Pinged: {loc}"
        
        @register_command(
            "callout",
            description="Make a callout",
            category=CommandCategory.COMMUNICATION,
            aliases=["call"]
        )
        def cmd_callout(message: str):
            return f"Callout: {message}"


# Global instance
_commands_instance: Optional[GameCommands] = None


def get_game_commands() -> GameCommands:
    """Get or create global GameCommands instance."""
    global _commands_instance
    if _commands_instance is None:
        _commands_instance = GameCommands()
    return _commands_instance


def quick_command(command: str, game: Optional[str] = None) -> CommandResult:
    """Execute a quick command."""
    return get_game_commands().execute(command, game)
