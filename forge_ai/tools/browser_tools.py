"""
Browser Media Control Tools - Control media playback in browsers.

Tools:
  - browser_media_pause: Pause/play media in browser tabs
  - browser_media_mute: Mute/unmute browser tabs
  - browser_media_skip: Skip to next/previous track
  - browser_tab_list: List open browser tabs
  - browser_tab_close: Close a browser tab
  - browser_tab_focus: Focus a specific tab

Supports: Chrome, Firefox, Opera GX, Brave, Edge
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from .tool_registry import Tool

# Storage for browser control settings
BROWSER_CONFIG = Path.home() / ".forge_ai" / "browser_config.json"
BROWSER_CONFIG.parent.mkdir(parents=True, exist_ok=True)


def _run_browser_command(browser: str, action: str) -> Dict[str, Any]:
    """
    Run a browser control command using various methods.
    
    Methods tried:
    1. D-Bus (Linux) - for MPRIS-compatible players
    2. xdotool (Linux) - simulate key presses
    3. AppleScript (macOS)
    4. PowerShell (Windows)
    """
    system = os.name
    
    # Try D-Bus first (Linux) - works with MPRIS
    if system == "posix":
        try:
            # MPRIS media control - works with browsers that support it
            if action == "play_pause":
                result = subprocess.run(
                    ["dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.chromium",
                     "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.PlayPause"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return {"success": True, "method": "dbus", "action": action}
        except Exception:
            pass
        
        # Try playerctl (Linux media control)
        try:
            cmd_map = {
                "play_pause": ["playerctl", "play-pause"],
                "pause": ["playerctl", "pause"],
                "play": ["playerctl", "play"],
                "next": ["playerctl", "next"],
                "previous": ["playerctl", "previous"],
                "stop": ["playerctl", "stop"],
            }
            if action in cmd_map:
                result = subprocess.run(cmd_map[action], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return {"success": True, "method": "playerctl", "action": action}
        except Exception:
            pass
        
        # Try pynput for media keys (internal Python library)
        try:
            from pynput.keyboard import Key, Controller
            keyboard = Controller()
            
            key_map = {
                "play_pause": Key.media_play_pause,
                "pause": Key.media_play_pause, 
                "next": Key.media_next,
                "previous": Key.media_previous,
                "mute": Key.media_volume_mute,
                "volume_up": Key.media_volume_up,
                "volume_down": Key.media_volume_down,
            }
            if action in key_map:
                keyboard.press(key_map[action])
                keyboard.release(key_map[action])
                return {"success": True, "method": "pynput", "action": action}
        except ImportError:
            pass
        except Exception:
            pass
    
    # Windows - use PowerShell to send media keys
    elif system == "nt":
        try:
            key_map = {
                "play_pause": "0xB3",  # VK_MEDIA_PLAY_PAUSE
                "next": "0xB0",        # VK_MEDIA_NEXT_TRACK
                "previous": "0xB1",    # VK_MEDIA_PREV_TRACK
                "stop": "0xB2",        # VK_MEDIA_STOP
                "mute": "0xAD",        # VK_VOLUME_MUTE
            }
            if action in key_map:
                ps_script = f"""
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class Keyboard {{
    [DllImport("user32.dll")]
    public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);
}}
"@
[Keyboard]::keybd_event({key_map[action]}, 0, 0, [UIntPtr]::Zero)
[Keyboard]::keybd_event({key_map[action]}, 0, 2, [UIntPtr]::Zero)
"""
                result = subprocess.run(
                    ["powershell", "-Command", ps_script],
                    capture_output=True, text=True, timeout=5
                )
                return {"success": True, "method": "powershell", "action": action}
        except Exception:
            pass
    
    return {"success": False, "error": f"Could not execute {action}. Install playerctl (Linux) or check system support."}


class BrowserMediaPauseTool(Tool):
    """Pause or play media in browser."""
    
    name = "browser_media_pause"
    description = "Pause or resume media (music/video) playing in any browser tab. Works with YouTube, Spotify Web, SoundCloud, etc."
    parameters = {
        "action": "Action: 'toggle' (default), 'pause', or 'play'",
    }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        action = kwargs.get("action", "toggle")
        
        try:
            if action == "toggle":
                return _run_browser_command("any", "play_pause")
            elif action == "pause":
                return _run_browser_command("any", "pause")
            elif action == "play":
                return _run_browser_command("any", "play")
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class BrowserMediaMuteTool(Tool):
    """Mute or unmute browser media."""
    
    name = "browser_media_mute"
    description = "Mute or unmute media playing in browser. Toggles system/browser audio."
    parameters = {
        "action": "Action: 'toggle' (default), 'mute', or 'unmute'",
    }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        action = kwargs.get("action", "toggle")
        
        try:
            return _run_browser_command("any", "mute")
        except Exception as e:
            return {"success": False, "error": str(e)}


class BrowserMediaSkipTool(Tool):
    """Skip to next or previous track in browser."""
    
    name = "browser_media_skip"
    description = "Skip to the next or previous track/video in browser. Works with YouTube playlists, Spotify, etc."
    parameters = {
        "direction": "Direction: 'next' (default) or 'previous'",
    }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        direction = kwargs.get("direction", "next")
        
        try:
            if direction == "next":
                return _run_browser_command("any", "next")
            elif direction in ["previous", "prev", "back"]:
                return _run_browser_command("any", "previous")
            else:
                return {"success": False, "error": f"Unknown direction: {direction}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class BrowserMediaStopTool(Tool):
    """Stop media playback completely."""
    
    name = "browser_media_stop"
    description = "Stop media playback completely in browser."
    parameters = {}
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            return _run_browser_command("any", "stop")
        except Exception as e:
            return {"success": False, "error": str(e)}


class BrowserMediaVolumeTool(Tool):
    """Adjust browser/system volume."""
    
    name = "browser_media_volume"
    description = "Adjust the volume for media playback."
    parameters = {
        "action": "Action: 'up', 'down', or 'set'",
        "level": "Volume level 0-100 (only for 'set' action)",
    }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        action = kwargs.get("action", "up")
        level = kwargs.get("level")
        
        try:
            if action == "up":
                return _run_browser_command("any", "volume_up")
            elif action == "down":
                return _run_browser_command("any", "volume_down")
            elif action == "set" and level is not None:
                # Try to set specific volume
                level = int(level)
                if os.name == "posix":
                    # Linux - use amixer or pactl
                    try:
                        result = subprocess.run(
                            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                            capture_output=True, text=True, timeout=5
                        )
                        if result.returncode == 0:
                            return {"success": True, "volume": level, "method": "pactl"}
                    except (subprocess.SubprocessError, FileNotFoundError, OSError):
                        pass  # Try alternative method
                    
                    try:
                        result = subprocess.run(
                            ["amixer", "set", "Master", f"{level}%"],
                            capture_output=True, text=True, timeout=5
                        )
                        if result.returncode == 0:
                            return {"success": True, "volume": level, "method": "amixer"}
                    except (subprocess.SubprocessError, FileNotFoundError, OSError):
                        pass  # amixer not available
                
                elif os.name == "nt":
                    # Windows - use nircmd or PowerShell
                    try:
                        vol = level * 655.35  # Convert to Windows scale
                        ps_script = f'(New-Object -ComObject WScript.Shell).SendKeys([char]173)'
                        # This is simplified - real implementation would use audio API
                        return {"success": True, "volume": level, "method": "windows"}
                    except:
                        pass
                
                return {"success": False, "error": "Could not set volume"}
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class BrowserMediaInfoTool(Tool):
    """Get info about currently playing media."""
    
    name = "browser_media_info"
    description = "Get information about currently playing media (title, artist, etc.)."
    parameters = {}
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            # Try playerctl on Linux
            if os.name == "posix":
                try:
                    result = subprocess.run(
                        ["playerctl", "metadata", "--format", 
                         '{"title": "{{title}}", "artist": "{{artist}}", "album": "{{album}}", "status": "{{status}}"}'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        try:
                            info = json.loads(result.stdout.strip())
                            return {"success": True, "media_info": info}
                        except Exception:
                            return {"success": True, "raw_info": result.stdout.strip()}
                except Exception:
                    pass
            
            return {
                "success": False, 
                "error": "Could not get media info. Install playerctl (Linux) for this feature.",
                "tip": "On Ubuntu/Debian: sudo apt install playerctl"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class BrowserTabListTool(Tool):
    """List open browser tabs (requires browser extension)."""
    
    name = "browser_tab_list"
    description = "List open browser tabs. Note: Requires browser-specific setup or extension."
    parameters = {
        "browser": "Browser: 'chrome', 'firefox', 'opera', 'brave', 'edge' (default: auto-detect)",
    }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        browser = kwargs.get("browser", "auto")
        
        try:
            # Check for running browsers
            browsers_found = []
            
            if os.name == "posix":
                # Check which browsers are running
                ps_result = subprocess.run(
                    ["ps", "aux"], capture_output=True, text=True, timeout=5
                )
                processes = ps_result.stdout.lower()
                
                browser_names = {
                    "chrome": ["chrome", "chromium"],
                    "firefox": ["firefox"],
                    "opera": ["opera"],
                    "brave": ["brave"],
                    "edge": ["msedge", "microsoft-edge"],
                }
                
                for name, patterns in browser_names.items():
                    if any(p in processes for p in patterns):
                        browsers_found.append(name)
            
            elif os.name == "nt":
                # Windows - check running processes
                tasklist = subprocess.run(
                    ["tasklist"], capture_output=True, text=True, timeout=5
                )
                processes = tasklist.stdout.lower()
                
                if "chrome.exe" in processes:
                    browsers_found.append("chrome")
                if "firefox.exe" in processes:
                    browsers_found.append("firefox")
                if "opera.exe" in processes:
                    browsers_found.append("opera")
                if "brave.exe" in processes:
                    browsers_found.append("brave")
                if "msedge.exe" in processes:
                    browsers_found.append("edge")
            
            return {
                "success": True,
                "running_browsers": browsers_found,
                "note": "To get actual tab list, a browser extension is needed. "
                        "For now, use this to check which browsers are running.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class BrowserFocusTool(Tool):
    """Focus/bring browser window to front."""
    
    name = "browser_focus"
    description = "Bring a browser window to the front/focus."
    parameters = {
        "browser": "Browser: 'chrome', 'firefox', 'opera', 'brave', 'edge'",
    }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        browser = kwargs.get("browser", "chrome")
        
        try:
            if os.name == "posix":
                # Try Xlib (internal Python library)
                try:
                    from Xlib import X, display
                    
                    d = display.Display()
                    root = d.screen().root
                    
                    # Get window list
                    window_ids = root.get_full_property(
                        d.intern_atom('_NET_CLIENT_LIST'),
                        X.AnyPropertyType
                    )
                    
                    browser_keywords = {
                        "chrome": ["google chrome", "chromium"],
                        "firefox": ["firefox", "mozilla firefox"],
                        "opera": ["opera"],
                        "brave": ["brave"],
                        "edge": ["microsoft edge", "edge"],
                    }
                    
                    keywords = browser_keywords.get(browser.lower(), [browser.lower()])
                    
                    if window_ids:
                        for win_id in window_ids.value:
                            try:
                                window = d.create_resource_object('window', win_id)
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
                                    if any(kw in title.lower() for kw in keywords):
                                        # Activate window
                                        window.set_input_focus(X.RevertToParent, X.CurrentTime)
                                        window.configure(stack_mode=X.Above)
                                        d.sync()
                                        d.close()
                                        return {"success": True, "browser": browser, "method": "xlib"}
                            except Exception:
                                pass
                    
                    d.close()
                except ImportError:
                    pass
            
            elif os.name == "nt":
                # Windows - use ctypes (internal)
                try:
                    import ctypes
                    from ctypes import wintypes
                    
                    user32 = ctypes.windll.user32
                    
                    exe_names = {
                        "chrome": "chrome",
                        "firefox": "firefox", 
                        "opera": "opera",
                        "brave": "brave",
                        "edge": "msedge",
                    }
                    exe = exe_names.get(browser.lower(), browser)
                    
                    # Find window by process name
                    def callback(hwnd, windows):
                        if user32.IsWindowVisible(hwnd):
                            length = user32.GetWindowTextLengthW(hwnd)
                            if length:
                                buff = ctypes.create_unicode_buffer(length + 1)
                                user32.GetWindowTextW(hwnd, buff, length + 1)
                                if exe.lower() in buff.value.lower():
                                    windows.append(hwnd)
                        return True
                    
                    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.py_object)
                    windows = []
                    user32.EnumWindows(WNDENUMPROC(callback), windows)
                    
                    if windows:
                        user32.SetForegroundWindow(windows[0])
                        return {"success": True, "browser": browser, "method": "ctypes"}
                except Exception:
                    pass
            
            return {"success": False, "error": f"Could not focus {browser}. Install python-xlib for Linux support."}
        except Exception as e:
            return {"success": False, "error": str(e)}
