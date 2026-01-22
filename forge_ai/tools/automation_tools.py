"""
Automation Tools - Scheduling, clipboard, hotkeys, folder watching.

Tools:
  - schedule_task: Schedule commands to run at specific times
  - clipboard_read: Read from clipboard
  - clipboard_write: Write to clipboard
  - clipboard_history: Get clipboard history
  - record_macro: Record keyboard/mouse actions
  - play_macro: Playback recorded macros
  - watch_folder: Monitor folder for changes
"""

import os
import json
import time
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from .tool_registry import Tool

# Storage paths
SCHEDULE_FILE = Path.home() / ".forge_ai" / "schedules.json"
MACRO_DIR = Path.home() / ".forge_ai" / "macros"
CLIPBOARD_HISTORY_FILE = Path.home() / ".forge_ai" / "clipboard_history.json"

SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
MACRO_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SCHEDULING TOOLS
# ============================================================================

class ScheduleManager:
    """Manages scheduled tasks."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.schedules: List[Dict] = []
        self.running = False
        self._thread = None
        self._load_schedules()
    
    def _load_schedules(self):
        """Load schedules from file."""
        if SCHEDULE_FILE.exists():
            try:
                with open(SCHEDULE_FILE, 'r') as f:
                    self.schedules = json.load(f)
            except:
                self.schedules = []
    
    def _save_schedules(self):
        """Save schedules to file."""
        with open(SCHEDULE_FILE, 'w') as f:
            json.dump(self.schedules, f, indent=2, default=str)
    
    def add_schedule(self, name: str, command: str, schedule_type: str, 
                     time_spec: str, enabled: bool = True) -> Dict:
        """
        Add a scheduled task.
        
        Args:
            name: Task name
            command: Command to run
            schedule_type: 'once', 'daily', 'hourly', 'interval', 'cron'
            time_spec: Time specification (e.g., '14:30', '*/5' for every 5 min)
            enabled: Whether task is active
        """
        schedule = {
            "id": len(self.schedules) + 1,
            "name": name,
            "command": command,
            "type": schedule_type,
            "time_spec": time_spec,
            "enabled": enabled,
            "created": datetime.now().isoformat(),
            "last_run": None,
            "next_run": self._calculate_next_run(schedule_type, time_spec),
        }
        self.schedules.append(schedule)
        self._save_schedules()
        return schedule
    
    def _calculate_next_run(self, schedule_type: str, time_spec: str) -> str:
        """Calculate next run time."""
        now = datetime.now()
        
        if schedule_type == 'once':
            # time_spec is ISO format or "HH:MM"
            if 'T' in time_spec or '-' in time_spec:
                return time_spec
            else:
                h, m = map(int, time_spec.split(':'))
                next_run = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                return next_run.isoformat()
        
        elif schedule_type == 'daily':
            h, m = map(int, time_spec.split(':'))
            next_run = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run.isoformat()
        
        elif schedule_type == 'hourly':
            m = int(time_spec) if time_spec else 0
            next_run = now.replace(minute=m, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(hours=1)
            return next_run.isoformat()
        
        elif schedule_type == 'interval':
            # time_spec is minutes
            minutes = int(time_spec)
            return (now + timedelta(minutes=minutes)).isoformat()
        
        return now.isoformat()
    
    def remove_schedule(self, schedule_id: int) -> bool:
        """Remove a schedule by ID."""
        self.schedules = [s for s in self.schedules if s['id'] != schedule_id]
        self._save_schedules()
        return True
    
    def list_schedules(self) -> List[Dict]:
        """List all schedules."""
        return self.schedules
    
    def start_scheduler(self):
        """Start the scheduler thread."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        self.running = False
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            now = datetime.now()
            for schedule in self.schedules:
                if not schedule.get('enabled'):
                    continue
                
                next_run = datetime.fromisoformat(schedule['next_run'])
                if now >= next_run:
                    # Run the task
                    try:
                        subprocess.run(schedule['command'], shell=True, timeout=300)
                        schedule['last_run'] = now.isoformat()
                        
                        # Update next run
                        if schedule['type'] != 'once':
                            schedule['next_run'] = self._calculate_next_run(
                                schedule['type'], schedule['time_spec']
                            )
                        else:
                            schedule['enabled'] = False
                        
                        self._save_schedules()
                    except Exception as e:
                        print(f"Schedule error: {e}")
            
            time.sleep(30)  # Check every 30 seconds


class ScheduleTaskTool(Tool):
    """Schedule a command to run at a specific time."""
    
    name = "schedule_task"
    description = "Schedule a command to run at a specific time. Supports one-time, daily, hourly, and interval schedules."
    parameters = {
        "name": "Name for this scheduled task",
        "command": "Shell command to execute",
        "schedule_type": "Type: 'once', 'daily', 'hourly', 'interval'",
        "time_spec": "Time: 'HH:MM' for daily/once, minutes for interval/hourly",
    }
    
    def execute(self, name: str, command: str, schedule_type: str = "once", 
                time_spec: str = None, **kwargs) -> Dict[str, Any]:
        try:
            manager = ScheduleManager()
            schedule = manager.add_schedule(name, command, schedule_type, time_spec or "0")
            manager.start_scheduler()
            
            return {
                "success": True,
                "message": f"Scheduled task '{name}'",
                "schedule": schedule,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class ListSchedulesTool(Tool):
    """List all scheduled tasks."""
    
    name = "list_schedules"
    description = "List all scheduled tasks with their status and next run time."
    parameters = {}
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            manager = ScheduleManager()
            schedules = manager.list_schedules()
            return {
                "success": True,
                "count": len(schedules),
                "schedules": schedules,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class RemoveScheduleTool(Tool):
    """Remove a scheduled task."""
    
    name = "remove_schedule"
    description = "Remove a scheduled task by its ID."
    parameters = {
        "schedule_id": "The ID of the schedule to remove",
    }
    
    def execute(self, schedule_id: int, **kwargs) -> Dict[str, Any]:
        try:
            manager = ScheduleManager()
            manager.remove_schedule(int(schedule_id))
            return {"success": True, "message": f"Removed schedule {schedule_id}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# CLIPBOARD TOOLS
# ============================================================================

class ClipboardHistory:
    """Manages clipboard history."""
    
    _instance = None
    MAX_HISTORY = 50
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.history: List[Dict] = []
        self._load_history()
    
    def _load_history(self):
        if CLIPBOARD_HISTORY_FILE.exists():
            try:
                with open(CLIPBOARD_HISTORY_FILE, 'r') as f:
                    self.history = json.load(f)
            except:
                self.history = []
    
    def _save_history(self):
        with open(CLIPBOARD_HISTORY_FILE, 'w') as f:
            json.dump(self.history[-self.MAX_HISTORY:], f, indent=2)
    
    def add(self, content: str, content_type: str = "text"):
        entry = {
            "content": content[:1000],  # Limit size
            "type": content_type,
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(entry)
        self._save_history()
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        return self.history[-limit:]


def _get_clipboard() -> str:
    """Get clipboard content (cross-platform, internal only)."""
    # Try pyperclip first (pure Python with fallbacks)
    try:
        import pyperclip
        return pyperclip.paste()
    except ImportError:
        pass
    
    # Try PyQt5 QClipboard (internal)
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QMimeData
        
        app = QApplication.instance()
        if app:
            clipboard = app.clipboard()
            return clipboard.text() or ""
    except ImportError:
        pass
    
    # Windows fallback using ctypes (internal)
    if os.name == 'nt':
        try:
            import ctypes
            CF_TEXT = 1
            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32
            user32.OpenClipboard(0)
            handle = user32.GetClipboardData(CF_TEXT)
            data = ctypes.c_char_p(handle).value
            user32.CloseClipboard()
            return data.decode('utf-8') if data else ""
        except:
            pass
    
    # Linux fallback using xclip/xsel
    if os.name == 'posix':
        try:
            import subprocess
            # Try xclip first
            result = subprocess.run(['xclip', '-selection', 'clipboard', '-o'], 
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        try:
            # Try xsel as fallback
            result = subprocess.run(['xsel', '--clipboard', '--output'], 
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        try:
            # macOS pbpaste
            result = subprocess.run(['pbpaste'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    
    return ""


def _set_clipboard(text: str) -> bool:
    """Set clipboard content (cross-platform, internal only)."""
    # Try pyperclip first (pure Python with fallbacks)
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except ImportError:
        pass
    
    # Try PyQt5 QClipboard (internal)
    try:
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app:
            clipboard = app.clipboard()
            clipboard.setText(text)
            return True
    except ImportError:
        pass
    
    # Linux fallback using xclip/xsel
    if os.name == 'posix':
        try:
            import subprocess
            # Try xclip first
            process = subprocess.Popen(['xclip', '-selection', 'clipboard'],
                                       stdin=subprocess.PIPE, text=True)
            process.communicate(input=text, timeout=5)
            if process.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        try:
            # Try xsel as fallback
            process = subprocess.Popen(['xsel', '--clipboard', '--input'],
                                       stdin=subprocess.PIPE, text=True)
            process.communicate(input=text, timeout=5)
            if process.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        try:
            # macOS pbcopy
            process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE, text=True)
            process.communicate(input=text, timeout=5)
            if process.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    
    return False


class ClipboardReadTool(Tool):
    """Read from clipboard."""
    
    name = "clipboard_read"
    description = "Read the current content of the system clipboard."
    parameters = {}
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            content = _get_clipboard()
            return {
                "success": True,
                "content": content,
                "length": len(content),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class ClipboardWriteTool(Tool):
    """Write to clipboard."""
    
    name = "clipboard_write"
    description = "Write text to the system clipboard."
    parameters = {
        "text": "The text to copy to clipboard",
    }
    
    def execute(self, text: str, **kwargs) -> Dict[str, Any]:
        try:
            success = _set_clipboard(text)
            if success:
                # Add to history
                ClipboardHistory().add(text)
                return {
                    "success": True,
                    "message": "Copied to clipboard",
                    "length": len(text),
                }
            return {"success": False, "error": "Failed to set clipboard"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ClipboardHistoryTool(Tool):
    """Get clipboard history."""
    
    name = "clipboard_history"
    description = "Get recent clipboard history."
    parameters = {
        "limit": "Maximum number of items to return (default: 10)",
    }
    
    def execute(self, limit: int = 10, **kwargs) -> Dict[str, Any]:
        try:
            history = ClipboardHistory().get_history(int(limit))
            return {
                "success": True,
                "count": len(history),
                "history": history,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# MACRO TOOLS
# ============================================================================

class MacroManager:
    """Manages keyboard/mouse macros."""
    
    def __init__(self):
        self.macros: Dict[str, List[Dict]] = {}
        self._load_macros()
    
    def _load_macros(self):
        """Load all macros from disk."""
        for file in MACRO_DIR.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    self.macros[file.stem] = json.load(f)
            except:
                pass
    
    def save_macro(self, name: str, actions: List[Dict]):
        """Save a macro."""
        self.macros[name] = actions
        with open(MACRO_DIR / f"{name}.json", 'w') as f:
            json.dump(actions, f, indent=2)
    
    def get_macro(self, name: str) -> Optional[List[Dict]]:
        """Get a macro by name."""
        return self.macros.get(name)
    
    def list_macros(self) -> List[str]:
        """List all macro names."""
        return list(self.macros.keys())
    
    def delete_macro(self, name: str):
        """Delete a macro."""
        if name in self.macros:
            del self.macros[name]
            macro_file = MACRO_DIR / f"{name}.json"
            if macro_file.exists():
                macro_file.unlink()


class RecordMacroTool(Tool):
    """Record a macro (simplified - records commands)."""
    
    name = "record_macro"
    description = "Create a macro that runs a sequence of shell commands."
    parameters = {
        "name": "Name for the macro",
        "commands": "List of shell commands to run (comma-separated or as list)",
        "delays": "Optional delays between commands in seconds (comma-separated)",
    }
    
    def execute(self, name: str, commands: str, delays: str = None, **kwargs) -> Dict[str, Any]:
        try:
            # Parse commands
            if isinstance(commands, str):
                cmd_list = [c.strip() for c in commands.split(',')]
            else:
                cmd_list = commands
            
            # Parse delays
            delay_list = []
            if delays:
                delay_list = [float(d.strip()) for d in delays.split(',')]
            
            # Build actions
            actions = []
            for i, cmd in enumerate(cmd_list):
                action = {"type": "command", "command": cmd}
                if i < len(delay_list):
                    action["delay_after"] = delay_list[i]
                actions.append(action)
            
            # Save
            manager = MacroManager()
            manager.save_macro(name, actions)
            
            return {
                "success": True,
                "message": f"Created macro '{name}' with {len(actions)} actions",
                "macro": actions,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class PlayMacroTool(Tool):
    """Play a recorded macro."""
    
    name = "play_macro"
    description = "Execute a previously recorded macro."
    parameters = {
        "name": "Name of the macro to play",
        "repeat": "Number of times to repeat (default: 1)",
    }
    
    def execute(self, name: str, repeat: int = 1, **kwargs) -> Dict[str, Any]:
        try:
            manager = MacroManager()
            actions = manager.get_macro(name)
            
            if not actions:
                return {"success": False, "error": f"Macro '{name}' not found"}
            
            results = []
            for _ in range(int(repeat)):
                for action in actions:
                    if action['type'] == 'command':
                        result = subprocess.run(
                            action['command'], shell=True,
                            capture_output=True, text=True, timeout=60
                        )
                        results.append({
                            "command": action['command'],
                            "returncode": result.returncode,
                            "stdout": result.stdout[:500],
                        })
                        
                        if 'delay_after' in action:
                            time.sleep(action['delay_after'])
            
            return {
                "success": True,
                "message": f"Executed macro '{name}' {repeat} time(s)",
                "results": results,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class ListMacrosTool(Tool):
    """List all available macros."""
    
    name = "list_macros"
    description = "List all saved macros."
    parameters = {}
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            manager = MacroManager()
            macros = manager.list_macros()
            return {
                "success": True,
                "count": len(macros),
                "macros": macros,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# FOLDER WATCH TOOLS
# ============================================================================

class FolderWatcher:
    """Watches folders for changes."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.watches: Dict[str, Dict] = {}
        self.running = False
        self._thread = None
        self._file_states: Dict[str, Dict[str, float]] = {}
    
    def add_watch(self, path: str, action: str, patterns: List[str] = None) -> Dict:
        """Add a folder watch."""
        path = str(Path(path).expanduser().resolve())
        watch = {
            "path": path,
            "action": action,
            "patterns": patterns or ["*"],
            "created": datetime.now().isoformat(),
        }
        self.watches[path] = watch
        self._file_states[path] = self._scan_folder(path)
        return watch
    
    def _scan_folder(self, path: str) -> Dict[str, float]:
        """Scan folder and return file modification times."""
        states = {}
        folder = Path(path)
        if folder.exists():
            for file in folder.rglob("*"):
                if file.is_file():
                    try:
                        states[str(file)] = file.stat().st_mtime
                    except:
                        pass
        return states
    
    def remove_watch(self, path: str):
        """Remove a folder watch."""
        path = str(Path(path).expanduser().resolve())
        if path in self.watches:
            del self.watches[path]
            if path in self._file_states:
                del self._file_states[path]
    
    def list_watches(self) -> List[Dict]:
        """List all active watches."""
        return list(self.watches.values())
    
    def start_watching(self):
        """Start the watcher thread."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
    
    def stop_watching(self):
        """Stop watching."""
        self.running = False
    
    def _watch_loop(self):
        """Main watch loop."""
        while self.running:
            for path, watch in list(self.watches.items()):
                current_state = self._scan_folder(path)
                old_state = self._file_states.get(path, {})
                
                # Find new or modified files
                for file, mtime in current_state.items():
                    if file not in old_state or old_state[file] != mtime:
                        # Check patterns
                        file_path = Path(file)
                        patterns = watch.get('patterns', ['*'])
                        
                        match = any(file_path.match(p) for p in patterns)
                        if match:
                            # Run action
                            action = watch['action'].replace('{file}', file)
                            action = action.replace('{filename}', file_path.name)
                            action = action.replace('{folder}', path)
                            
                            try:
                                subprocess.run(action, shell=True, timeout=60)
                            except Exception as e:
                                print(f"Watch action error: {e}")
                
                self._file_states[path] = current_state
            
            time.sleep(5)  # Check every 5 seconds


class WatchFolderTool(Tool):
    """Watch a folder for changes."""
    
    name = "watch_folder"
    description = "Monitor a folder for new or modified files and run a command when changes occur."
    parameters = {
        "path": "Path to the folder to watch",
        "action": "Command to run when files change. Use {file} for full path, {filename} for name only",
        "patterns": "File patterns to match (default: all files). Comma-separated, e.g. '*.jpg,*.png'",
    }
    
    def execute(self, path: str, action: str, patterns: str = None, **kwargs) -> Dict[str, Any]:
        try:
            pattern_list = None
            if patterns:
                pattern_list = [p.strip() for p in patterns.split(',')]
            
            watcher = FolderWatcher()
            watch = watcher.add_watch(path, action, pattern_list)
            watcher.start_watching()
            
            return {
                "success": True,
                "message": f"Now watching {path}",
                "watch": watch,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class StopWatchTool(Tool):
    """Stop watching a folder."""
    
    name = "stop_watch"
    description = "Stop watching a folder for changes."
    parameters = {
        "path": "Path to the folder to stop watching",
    }
    
    def execute(self, path: str, **kwargs) -> Dict[str, Any]:
        try:
            watcher = FolderWatcher()
            watcher.remove_watch(path)
            return {"success": True, "message": f"Stopped watching {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ListWatchesTool(Tool):
    """List all folder watches."""
    
    name = "list_watches"
    description = "List all active folder watches."
    parameters = {}
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            watcher = FolderWatcher()
            watches = watcher.list_watches()
            return {
                "success": True,
                "count": len(watches),
                "watches": watches,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
