"""
Process Monitor - Game Detection System

Monitors system processes to detect when games are running.
Works cross-platform (Windows, Linux, macOS).
"""

import platform
import subprocess
import logging
from typing import Optional, Dict, List, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessMonitor:
    """
    Monitor system processes for game detection.
    
    Detects games through:
    - Known game process names
    - Fullscreen applications
    - GPU usage patterns
    """
    
    # Known game executables (actual games, NOT launchers)
    KNOWN_GAMES = [
        # Common games
        "minecraft.exe", "javaw.exe",  # Minecraft
        "FortniteClient-Win64-Shipping.exe", "fortnite.exe",
        "csgo.exe", "cs2.exe",
        "VALORANT.exe", "valorant-win64-shipping.exe",
        "GTA5.exe", "RDR2.exe",
        "Cyberpunk2077.exe",
        "eldenring.exe",
        "bg3.exe", "baldursgate3.exe",  # Baldur's Gate 3
        "witcher3.exe",
        "starfield.exe",
        "skyrim.exe", "fallout4.exe",
        # VR
        "vrchat.exe",
        "boneworks.exe",
        "alyx.exe",
        "beatsaber.exe",
        # Competitive
        "overwatch.exe",
        "r5apex.exe",  # Apex Legends
        "pubg.exe",
        "cod.exe",
        "modernwarfare.exe",
        # Strategy
        "stellaris.exe",
        "ck3.exe",
        "eu4.exe",
        "hoi4.exe",
        "civ6.exe",
        "totalwar.exe",
        "aoe4.exe",
    ]
    
    # Launchers to IGNORE (they're not actual games)
    LAUNCHER_EXCLUSIONS = [
        "steam.exe", "steamwebhelper.exe", "steamservice.exe",
        "EpicGamesLauncher.exe", "EpicWebHelper.exe",
        "Battle.net.exe", "Agent.exe",  # Blizzard
        "Origin.exe", "OriginWebHelperService.exe",
        "upc.exe", "UbisoftConnect.exe",  # Ubisoft
        "EADesktop.exe", "EABackgroundService.exe",
        "GalaxyClient.exe",  # GOG
        "nw.exe",  # Common game launcher wrapper
    ]
    
    def __init__(self, custom_games: Optional[List[str]] = None):
        """
        Initialize process monitor.
        
        Args:
            custom_games: Additional game process names to detect
        """
        self.custom_games = custom_games or []
        self.all_games = set(g.lower() for g in self.KNOWN_GAMES + self.custom_games)
        self.launcher_exclusions = set(g.lower() for g in self.LAUNCHER_EXCLUSIONS)
        self._system = platform.system()
    
    def get_fullscreen_app(self) -> Optional[str]:
        """
        Get currently fullscreen application name.
        
        Returns:
            Process name if fullscreen app detected, None otherwise
        """
        result = None
        if self._system == "Windows":
            result = self._get_fullscreen_windows()
        elif self._system == "Linux":
            result = self._get_fullscreen_linux()
        elif self._system == "Darwin":
            result = self._get_fullscreen_macos()
        
        # Exclude launcher processes from fullscreen detection
        if result and result.lower() in self.launcher_exclusions:
            return None
        return result
    
    def _get_fullscreen_windows(self) -> Optional[str]:
        """Get fullscreen app on Windows."""
        try:
            import ctypes
            from ctypes import wintypes
            
            user32 = ctypes.windll.user32
            
            # Get foreground window
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None
            
            # Get window rect
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            
            # Get screen size
            screen_width = user32.GetSystemMetrics(0)
            screen_height = user32.GetSystemMetrics(1)
            
            # Check if window covers entire screen
            window_width = rect.right - rect.left
            window_height = rect.bottom - rect.top
            
            if window_width >= screen_width and window_height >= screen_height:
                # Get process ID
                pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                
                # Get process name
                process_name = self._get_process_name_windows(pid.value)
                return process_name
            
        except Exception as e:
            logger.debug(f"Error detecting fullscreen window: {e}")
        
        return None
    
    def _get_fullscreen_linux(self) -> Optional[str]:
        """Get fullscreen app on Linux (X11/Wayland)."""
        try:
            # Try X11 first
            output = subprocess.check_output(
                ["xprop", "-root", "_NET_ACTIVE_WINDOW"],
                text=True,
                stderr=subprocess.DEVNULL
            )
            
            if "window id" in output:
                window_id = output.split()[-1]
                
                # Check if window is fullscreen
                state_output = subprocess.check_output(
                    ["xprop", "-id", window_id, "_NET_WM_STATE"],
                    text=True,
                    stderr=subprocess.DEVNULL
                )
                
                if "_NET_WM_STATE_FULLSCREEN" in state_output:
                    # Get window PID
                    pid_output = subprocess.check_output(
                        ["xprop", "-id", window_id, "_NET_WM_PID"],
                        text=True,
                        stderr=subprocess.DEVNULL
                    )
                    
                    if "=" in pid_output:
                        pid = int(pid_output.split("=")[-1].strip())
                        return self._get_process_name_linux(pid)
        
        except (subprocess.CalledProcessError, FileNotFoundError):
            # X11 not available or xprop not installed
            pass
        
        return None
    
    def _get_fullscreen_macos(self) -> Optional[str]:
        """Get fullscreen app on macOS."""
        try:
            # Use AppleScript to check fullscreen status
            script = '''
            tell application "System Events"
                set frontApp to name of first application process whose frontmost is true
                return frontApp
            end tell
            '''
            
            output = subprocess.check_output(
                ["osascript", "-e", script],
                text=True
            ).strip()
            
            return output if output else None
            
        except Exception as e:
            logger.debug(f"Error detecting fullscreen on macOS: {e}")
        
        return None
    
    def get_gpu_usage_by_process(self) -> Dict[str, float]:
        """
        Get GPU usage per process.
        
        Returns:
            Dictionary mapping process name to GPU usage percentage
        """
        gpu_usage = {}
        
        try:
            if self._system == "Windows":
                # Try nvidia-smi first
                try:
                    output = subprocess.check_output(
                        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                        text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    
                    for line in output.strip().split('\n'):
                        if line:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                pid = int(parts[0].strip())
                                memory = float(parts[1].strip())
                                process_name = self._get_process_name_windows(pid)
                                if process_name:
                                    gpu_usage[process_name] = memory
                
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass
            
            elif self._system == "Linux":
                # Use nvidia-smi on Linux
                try:
                    output = subprocess.check_output(
                        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                        text=True
                    )
                    
                    for line in output.strip().split('\n'):
                        if line:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                pid = int(parts[0].strip())
                                memory = float(parts[1].strip())
                                process_name = self._get_process_name_linux(pid)
                                if process_name:
                                    gpu_usage[process_name] = memory
                
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass
        
        except Exception as e:
            logger.debug(f"Error getting GPU usage: {e}")
        
        return gpu_usage
    
    def is_game_running(self) -> bool:
        """
        Check if any known game is running.
        
        Returns:
            True if game detected, False otherwise
        """
        running = self.get_running_games()
        return len(running) > 0
    
    def get_running_games(self) -> Set[str]:
        """
        Get all currently running known games.
        
        Returns:
            Set of running game process names
        """
        all_processes = self._get_running_processes()
        games = set()
        
        for proc in all_processes:
            proc_lower = proc.lower()
            if proc_lower in self.all_games:
                games.add(proc)
        
        return games
    
    def _get_running_processes(self) -> Set[str]:
        """Get set of all running process names."""
        processes = set()
        
        try:
            if self._system == "Windows":
                output = subprocess.check_output(
                    ["tasklist", "/fo", "csv", "/nh"],
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                for line in output.strip().split('\n'):
                    if line:
                        parts = line.split('","')
                        if parts:
                            proc_name = parts[0].strip('"')
                            processes.add(proc_name)
            
            elif self._system in ("Linux", "Darwin"):
                output = subprocess.check_output(
                    ["ps", "-A", "-o", "comm="],
                    text=True
                )
                
                for line in output.strip().split('\n'):
                    if line:
                        processes.add(line.strip())
        
        except Exception as e:
            logger.debug(f"Error getting process list: {e}")
        
        return processes
    
    def _get_process_name_windows(self, pid: int) -> Optional[str]:
        """Get process name from PID on Windows."""
        try:
            output = subprocess.check_output(
                ["tasklist", "/fi", f"PID eq {pid}", "/fo", "csv", "/nh"],
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            if output.strip():
                parts = output.strip().split('","')
                if parts:
                    return parts[0].strip('"')
        
        except Exception:
            pass
        
        return None
    
    def _get_process_name_linux(self, pid: int) -> Optional[str]:
        """Get process name from PID on Linux/macOS."""
        try:
            comm_file = Path(f"/proc/{pid}/comm")
            if comm_file.exists():
                return comm_file.read_text().strip()
            
            # Fallback to ps
            output = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "comm="],
                text=True
            )
            return output.strip()
        
        except Exception:
            pass
        
        return None
    
    def get_known_game_processes(self) -> List[str]:
        """
        Get list of all known game executables.
        
        Returns:
            List of known game process names
        """
        return list(self.all_games)
    
    def add_custom_game(self, process_name: str):
        """
        Add a custom game to detection list.
        
        Args:
            process_name: Name of game executable
        """
        if process_name not in self.custom_games:
            self.custom_games.append(process_name)
            self.all_games.add(process_name.lower())
    
    def remove_custom_game(self, process_name: str):
        """
        Remove a custom game from detection list.
        
        Args:
            process_name: Name of game executable
        """
        if process_name in self.custom_games:
            self.custom_games.remove(process_name)
            self.all_games.discard(process_name.lower())


# Global instance
_process_monitor: Optional[ProcessMonitor] = None


def get_process_monitor() -> ProcessMonitor:
    """Get or create global ProcessMonitor instance."""
    global _process_monitor
    if _process_monitor is None:
        _process_monitor = ProcessMonitor()
    return _process_monitor


__all__ = ['ProcessMonitor', 'get_process_monitor']
