"""
================================================================================
Steam Integration - Auto-detect games from Steam library
================================================================================

Integrates with Steam to:
1. Find installed games from Steam library
2. Auto-detect running games
3. Get game metadata (name, appid, playtime)
4. Create gaming profiles from Steam games

USAGE:
    from enigma_engine.core.steam_integration import get_steam_integration
    
    steam = get_steam_integration()
    
    # Get installed games
    games = steam.get_installed_games()
    
    # Check for running Steam games
    running = steam.get_running_games()
    
    # Auto-register games with gaming mode
    steam.register_with_gaming_mode()
"""

import logging
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SteamGame:
    """Information about a Steam game."""
    appid: int
    name: str
    install_dir: str = ""
    executable: str = ""
    process_name: str = ""
    playtime_hours: float = 0.0
    last_played: int = 0
    is_installed: bool = True
    categories: List[str] = field(default_factory=list)
    
    @property
    def process_names(self) -> List[str]:
        """Get possible process names for this game."""
        names = []
        if self.process_name:
            names.append(self.process_name)
        if self.executable:
            names.append(Path(self.executable).name)
        # Guess common variations
        if self.install_dir:
            base = Path(self.install_dir).name.lower()
            names.append(f"{base}.exe")
        return list(set(names))


class SteamIntegration:
    """
    Steam integration for auto-detecting games.
    
    Reads Steam library folders and manifests to find installed games,
    and monitors for running Steam games.
    """
    
    def __init__(self):
        self._steam_path: Optional[Path] = None
        self._library_folders: List[Path] = []
        self._games: Dict[int, SteamGame] = {}
        self._initialized = False
        
        self._detect_steam()
    
    def _detect_steam(self):
        """Detect Steam installation path."""
        system = platform.system()
        
        possible_paths = []
        
        if system == "Windows":
            # Common Windows paths
            possible_paths = [
                Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Steam",
                Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Steam",
                Path.home() / "Steam",
            ]
            # Check registry
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Software\Valve\Steam"
                )
                steam_path, _ = winreg.QueryValueEx(key, "SteamPath")
                winreg.CloseKey(key)
                possible_paths.insert(0, Path(steam_path))
            except Exception:
                pass  # Intentionally silent
                
        elif system == "Darwin":
            # macOS
            possible_paths = [
                Path.home() / "Library/Application Support/Steam",
            ]
            
        elif system == "Linux":
            # Linux
            possible_paths = [
                Path.home() / ".steam/steam",
                Path.home() / ".steam/debian-installation",
                Path.home() / ".local/share/Steam",
            ]
        
        # Find valid Steam path
        for path in possible_paths:
            if path.exists() and (path / "steam.exe").exists() or (path / "steamapps").exists():
                self._steam_path = path
                logger.info(f"Found Steam at: {path}")
                break
        
        if self._steam_path:
            self._find_library_folders()
            self._initialized = True
    
    def _find_library_folders(self):
        """Find all Steam library folders."""
        self._library_folders = []
        
        if not self._steam_path:
            return
        
        # Main steamapps folder
        main_steamapps = self._steam_path / "steamapps"
        if main_steamapps.exists():
            self._library_folders.append(main_steamapps)
        
        # Additional library folders from libraryfolders.vdf
        vdf_path = main_steamapps / "libraryfolders.vdf"
        if vdf_path.exists():
            try:
                content = vdf_path.read_text(encoding='utf-8', errors='ignore')
                # Parse paths from VDF
                # Format: "path"    "D:\\SteamLibrary"
                path_pattern = r'"path"\s*"([^"]+)"'
                for match in re.finditer(path_pattern, content):
                    lib_path = Path(match.group(1)) / "steamapps"
                    if lib_path.exists() and lib_path not in self._library_folders:
                        self._library_folders.append(lib_path)
            except Exception as e:
                logger.error(f"Failed to parse libraryfolders.vdf: {e}")
        
        logger.debug(f"Found {len(self._library_folders)} Steam library folders")
    
    def _parse_acf(self, acf_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a Steam ACF manifest file."""
        try:
            content = acf_path.read_text(encoding='utf-8', errors='ignore')
            
            # Simple VDF parser for ACF files
            data = {}
            current_key = None
            
            # Pattern to match "key" "value" pairs
            kv_pattern = r'"(\w+)"\s*"([^"]*)"'
            
            for match in re.finditer(kv_pattern, content):
                key, value = match.groups()
                data[key.lower()] = value
            
            return data
            
        except Exception as e:
            logger.debug(f"Failed to parse ACF {acf_path}: {e}")
            return None
    
    def get_installed_games(self, refresh: bool = False) -> List[SteamGame]:
        """
        Get list of installed Steam games.
        
        Args:
            refresh: Force refresh the game list
            
        Returns:
            List of SteamGame objects
        """
        if not self._initialized:
            return []
        
        if not refresh and self._games:
            return list(self._games.values())
        
        self._games.clear()
        
        for library_folder in self._library_folders:
            # Find all appmanifest files
            for acf_file in library_folder.glob("appmanifest_*.acf"):
                try:
                    data = self._parse_acf(acf_file)
                    if not data:
                        continue
                    
                    appid = int(data.get("appid", 0))
                    if appid == 0:
                        continue
                    
                    name = data.get("name", f"Game {appid}")
                    install_dir = data.get("installdir", "")
                    
                    # Get full install path
                    full_path = ""
                    if install_dir:
                        common = library_folder / "common" / install_dir
                        if common.exists():
                            full_path = str(common)
                    
                    # Try to find executable
                    executable = ""
                    process_name = ""
                    if full_path:
                        game_path = Path(full_path)
                        # Look for common executable patterns
                        for exe in game_path.glob("*.exe"):
                            if exe.name.lower() not in ["unins000.exe", "uninstall.exe", "crash_reporter.exe"]:
                                executable = str(exe)
                                process_name = exe.name.lower()
                                break
                    
                    game = SteamGame(
                        appid=appid,
                        name=name,
                        install_dir=full_path,
                        executable=executable,
                        process_name=process_name,
                        is_installed=True,
                    )
                    
                    self._games[appid] = game
                    
                except Exception as e:
                    logger.debug(f"Failed to process {acf_file}: {e}")
        
        logger.info(f"Found {len(self._games)} installed Steam games")
        return list(self._games.values())
    
    def get_running_games(self) -> List[SteamGame]:
        """Check which Steam games are currently running."""
        if not self._games:
            self.get_installed_games()
        
        running = []
        
        # Get list of running processes
        running_procs = self._get_running_processes()
        
        for game in self._games.values():
            for proc_name in game.process_names:
                if proc_name.lower() in running_procs:
                    running.append(game)
                    break
        
        return running
    
    def _get_running_processes(self) -> Set[str]:
        """Get set of running process names."""
        processes = set()
        
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["tasklist", "/fo", "csv", "/nh"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in result.stdout.strip().split("\n"):
                    if line:
                        parts = line.split(",")
                        if parts:
                            proc = parts[0].strip('"').lower()
                            processes.add(proc)
            else:
                result = subprocess.run(
                    ["ps", "-eo", "comm"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in result.stdout.strip().split("\n"):
                    processes.add(line.strip().lower())
                    
        except Exception as e:
            logger.debug(f"Failed to get running processes: {e}")
        
        return processes
    
    def register_with_gaming_mode(self, priority: str = "HIGH"):
        """
        Register all Steam games with the gaming mode system.
        
        Args:
            priority: Default priority level for Steam games
        """
        try:
            from enigma_engine.core.gaming_mode import (
                GamingPriority,
                GamingProfile,
                get_gaming_mode,
            )
            
            gm = get_gaming_mode()
            games = self.get_installed_games()
            
            registered = 0
            for game in games:
                if not game.process_names:
                    continue
                
                # Check if already has a profile
                key = game.name.lower().replace(" ", "_").replace(":", "").replace("-", "_")
                if key in gm.profiles:
                    continue
                
                # Create profile
                profile = GamingProfile(
                    name=game.name,
                    process_names=game.process_names,
                    priority=GamingPriority[priority],
                    cpu_inference=True,
                    max_vram_mb=512,
                    max_ram_mb=2048,
                    batch_size=1,
                    defer_heavy_tasks=True,
                    voice_enabled=True,
                    avatar_enabled=True,
                    notes=f"Auto-registered from Steam (AppID: {game.appid})",
                )
                
                gm.add_game_profile(profile, auto_save=False)
                registered += 1
            
            # Save all at once
            if registered > 0:
                gm.save_profiles()
                logger.info(f"Registered {registered} Steam games with gaming mode")
            
            return registered
            
        except Exception as e:
            logger.error(f"Failed to register Steam games: {e}")
            return 0
    
    def get_game_by_appid(self, appid: int) -> Optional[SteamGame]:
        """Get a game by its Steam AppID."""
        if not self._games:
            self.get_installed_games()
        return self._games.get(appid)
    
    def get_game_by_name(self, name: str) -> Optional[SteamGame]:
        """Find a game by name (case-insensitive partial match)."""
        if not self._games:
            self.get_installed_games()
        
        name_lower = name.lower()
        for game in self._games.values():
            if name_lower in game.name.lower():
                return game
        return None
    
    @property
    def is_available(self) -> bool:
        """Check if Steam integration is available."""
        return self._initialized
    
    @property
    def steam_path(self) -> Optional[Path]:
        """Get Steam installation path."""
        return self._steam_path
    
    @property
    def game_count(self) -> int:
        """Get count of installed games."""
        return len(self._games)


# Global instance
_steam: Optional[SteamIntegration] = None


def get_steam_integration() -> SteamIntegration:
    """Get or create the global Steam integration instance."""
    global _steam
    if _steam is None:
        _steam = SteamIntegration()
    return _steam


__all__ = [
    'SteamIntegration',
    'SteamGame',
    'get_steam_integration',
]
