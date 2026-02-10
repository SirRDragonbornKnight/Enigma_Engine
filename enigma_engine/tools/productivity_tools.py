"""
Productivity Tools - System monitoring, processes, SSH, Docker, Git.
All tools execute system commands that AI cannot learn.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .tool_registry import Tool, RichParameter


class SystemMonitorTool(Tool):
    """Monitor system resources."""
    name = "system_monitor"
    description = "Get CPU, memory, disk usage, and temperature."
    parameters = {"detailed": "Show detailed breakdown (default: False)"}
    category = "productivity"
    rich_parameters = [
        RichParameter(name="detailed", type="boolean", description="Show detailed breakdown", required=False, default=False),
    ]
    examples = ["system_monitor()", "system_monitor(detailed=True)"]
    
    def execute(self, detailed: bool = False, **kwargs) -> dict[str, Any]:
        try:
            result = {"success": True, "timestamp": datetime.now().isoformat()}
            
            try:
                import psutil
                result["cpu"] = {"percent": psutil.cpu_percent(interval=1), "count": psutil.cpu_count()}
                mem = psutil.virtual_memory()
                result["memory"] = {"total_gb": round(mem.total/1024**3, 2), "used_percent": mem.percent}
                disks = []
                for p in psutil.disk_partitions():
                    try:
                        u = psutil.disk_usage(p.mountpoint)
                        disks.append({"mount": p.mountpoint, "used_percent": u.percent, "free_gb": round(u.free/1024**3, 2)})
                    except (PermissionError, OSError):
                        pass  # Skip inaccessible disk partitions
                result["disks"] = disks
                try:
                    temps = psutil.sensors_temperatures()
                    if temps: result["temps"] = {k: [{"label": e.label, "temp": e.current} for e in v] for k, v in temps.items()}
                except (AttributeError, OSError):
                    pass  # Temperature sensors not available on all platforms
                return result
            except ImportError: pass
            
            # Fallback for Linux
            try:
                with open('/proc/loadavg') as f: result["load"] = f.read().split()[:3]
            except (FileNotFoundError, PermissionError):
                pass  # Linux-specific file not available
            try:
                with open('/sys/class/thermal/thermal_zone0/temp') as f: result["cpu_temp_c"] = int(f.read())/1000
            except (FileNotFoundError, PermissionError, ValueError):
                pass  # Thermal sensor not available
            result["note"] = "Install psutil for full details"
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}


class ProcessListTool(Tool):
    """List running processes."""
    name = "process_list"
    description = "List running processes sorted by CPU/memory usage."
    parameters = {"sort_by": "Sort by: cpu, memory, name (default: cpu)", "limit": "Max processes (default: 20)"}
    category = "productivity"
    rich_parameters = [
        RichParameter(name="sort_by", type="string", description="Sort by", required=False, default="cpu", enum=["cpu", "memory", "name"]),
        RichParameter(name="limit", type="integer", description="Max processes to show", required=False, default=20, min_value=1, max_value=100),
    ]
    examples = ["process_list()", "process_list(sort_by='memory', limit=10)"]
    
    def execute(self, sort_by: str = "cpu", limit: int = 20, **kwargs) -> dict[str, Any]:
        try:
            try:
                import psutil
                procs = []
                for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try: 
                        procs.append(p.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass  # Process terminated or access denied during iteration
                key = {'cpu': 'cpu_percent', 'memory': 'memory_percent', 'name': 'name'}.get(sort_by, 'cpu_percent')
                procs.sort(key=lambda x: x.get(key, 0) or 0, reverse=key != 'name')
                return {"success": True, "processes": procs[:limit]}
            except ImportError:
                result = subprocess.run(['ps', 'aux', '--sort=-pcpu'], capture_output=True, text=True, timeout=10)
                return {"success": True, "output": result.stdout[:5000]}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ProcessKillTool(Tool):
    """Kill a process."""
    name = "process_kill"
    description = "Kill a process by PID or name."
    parameters = {"pid": "Process ID (optional)", "name": "Process name (optional)", "signal": "Signal: TERM, KILL (default: TERM)"}
    category = "productivity"
    rich_parameters = [
        RichParameter(name="pid", type="integer", description="Process ID", required=False),
        RichParameter(name="name", type="string", description="Process name", required=False),
        RichParameter(name="signal", type="string", description="Signal to send", required=False, default="TERM", enum=["TERM", "KILL", "HUP"]),
    ]
    examples = ["process_kill(pid=12345)", "process_kill(name='firefox', signal='TERM')"]
    
    def execute(self, pid: int = None, name: str = None, signal: str = "TERM", **kwargs) -> dict[str, Any]:
        try:
            import signal as sig
            signals = {"TERM": sig.SIGTERM, "KILL": sig.SIGKILL, "HUP": sig.SIGHUP}
            sig_num = signals.get(signal.upper(), sig.SIGTERM)
            
            if pid:
                os.kill(int(pid), sig_num)
                return {"success": True, "killed_pid": pid}
            elif name:
                result = subprocess.run(['pkill', f'-{signal}', name], capture_output=True, text=True, timeout=30)
                return {"success": result.returncode == 0, "killed_name": name}
            return {"success": False, "error": "Provide pid or name"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class SSHExecuteTool(Tool):
    """Execute SSH commands."""
    name = "ssh_execute"
    description = "Execute command on remote machine via SSH."
    parameters = {"host": "user@host", "command": "Command to run", "timeout": "Timeout seconds (default: 30)"}
    category = "productivity"
    rich_parameters = [
        RichParameter(name="host", type="string", description="SSH target (user@host)", required=True),
        RichParameter(name="command", type="string", description="Command to execute", required=True),
        RichParameter(name="timeout", type="integer", description="Timeout in seconds", required=False, default=30, min_value=5, max_value=600),
    ]
    examples = ["ssh_execute(host='user@server.com', command='uptime')"]
    
    def execute(self, host: str, command: str, timeout: int = 30, **kwargs) -> dict[str, Any]:
        try:
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, command],
                                   capture_output=True, text=True, timeout=timeout)
            return {"success": result.returncode == 0, "host": host, "stdout": result.stdout, "stderr": result.stderr}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "SSH timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class DockerListTool(Tool):
    """List Docker containers."""
    name = "docker_list"
    description = "List Docker containers."
    parameters = {"all": "Show all containers (default: True)"}
    category = "productivity"
    rich_parameters = [
        RichParameter(name="all", type="boolean", description="Show all containers (including stopped)", required=False, default=True),
    ]
    examples = ["docker_list()", "docker_list(all=False)"]
    
    def execute(self, all: bool = True, **kwargs) -> dict[str, Any]:
        try:
            cmd = ['docker', 'ps', '--format', '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}']
            if all: cmd.insert(2, '-a')
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0: return {"success": False, "error": result.stderr or "Docker unavailable"}
            containers = [{"id": p[0], "name": p[1], "image": p[2], "status": p[3]} 
                         for line in result.stdout.strip().split('\n') if line for p in [line.split('\t')] if len(p) >= 4]
            return {"success": True, "count": len(containers), "containers": containers}
        except Exception as e:
            return {"success": False, "error": str(e)}


class DockerControlTool(Tool):
    """Control Docker containers."""
    name = "docker_control"
    description = "Start, stop, restart, or get logs from Docker container."
    parameters = {"container": "Container name/ID", "action": "start, stop, restart, remove, logs"}
    category = "productivity"
    rich_parameters = [
        RichParameter(name="container", type="string", description="Container name or ID", required=True),
        RichParameter(name="action", type="string", description="Action to perform", required=True, enum=["start", "stop", "restart", "remove", "logs"]),
    ]
    examples = ["docker_control(container='my-app', action='restart')", "docker_control(container='nginx', action='logs')"]
    
    def execute(self, container: str, action: str, **kwargs) -> dict[str, Any]:
        try:
            if action not in ['start', 'stop', 'restart', 'remove', 'logs']:
                return {"success": False, "error": "Invalid action"}
            cmd = ['docker', 'logs', '--tail', '100', container] if action == 'logs' else ['docker', action, container]
            if action == 'remove': cmd = ['docker', 'rm', '-f', container]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return {"success": result.returncode == 0, "action": action, "output": result.stdout if action == 'logs' else None, 
                    "error": result.stderr if result.returncode != 0 else None}
        except Exception as e:
            return {"success": False, "error": str(e)}


class GitStatusTool(Tool):
    """Check git status."""
    name = "git_status"
    description = "Check git repository status."
    parameters = {"path": "Repository path (default: .)"}
    category = "productivity"
    rich_parameters = [
        RichParameter(name="path", type="string", description="Repository path", required=False, default="."),
    ]
    examples = ["git_status()", "git_status(path='/home/user/project')"]
    
    def execute(self, path: str = ".", **kwargs) -> dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            if not (path / '.git').exists(): return {"success": False, "error": "Not a git repo"}
            
            branch = subprocess.run(['git', 'branch', '--show-current'], cwd=str(path), capture_output=True, text=True, timeout=10).stdout.strip()
            status = subprocess.run(['git', 'status', '--porcelain'], cwd=str(path), capture_output=True, text=True, timeout=10).stdout
            staged, modified, untracked = [], [], []
            for line in status.strip().split('\n'):
                if not line: continue
                if line[0] in 'MADRC': staged.append(line[3:])
                if line[1] == 'M': modified.append(line[3:])
                if line[:2] == '??': untracked.append(line[3:])
            last = subprocess.run(['git', 'log', '-1', '--pretty=%h %s (%cr)'], cwd=str(path), capture_output=True, text=True, timeout=10).stdout.strip()
            return {"success": True, "branch": branch, "staged": staged, "modified": modified, "untracked": untracked, 
                    "clean": not (staged or modified), "last_commit": last}
        except Exception as e:
            return {"success": False, "error": str(e)}


class GitCommitTool(Tool):
    """Create git commit."""
    name = "git_commit"
    description = "Stage and commit changes."
    parameters = {"message": "Commit message", "path": "Repository path (default: .)", "add_all": "Add all changes (default: True)"}
    category = "productivity"
    rich_parameters = [
        RichParameter(name="message", type="string", description="Commit message", required=True),
        RichParameter(name="path", type="string", description="Repository path", required=False, default="."),
        RichParameter(name="add_all", type="boolean", description="Add all changes", required=False, default=True),
    ]
    examples = ["git_commit(message='Fix bug in login')", "git_commit(message='Add feature', add_all=False)"]
    
    def execute(self, message: str, path: str = ".", add_all: bool = True, **kwargs) -> dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            if add_all:
                subprocess.run(['git', 'add', '-A'], cwd=str(path), capture_output=True, timeout=30)
            result = subprocess.run(['git', 'commit', '-m', message], cwd=str(path), capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                if 'nothing to commit' in result.stdout + result.stderr:
                    return {"success": True, "committed": False, "message": "Nothing to commit"}
                return {"success": False, "error": result.stderr}
            hash = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=str(path), capture_output=True, text=True, timeout=10).stdout.strip()
            return {"success": True, "committed": True, "hash": hash, "message": message}
        except Exception as e:
            return {"success": False, "error": str(e)}


class GitPushTool(Tool):
    """Push to remote."""
    name = "git_push"
    description = "Push commits to remote repository."
    parameters = {"path": "Repository path (default: .)", "remote": "Remote name (default: origin)"}
    category = "productivity"
    rich_parameters = [
        RichParameter(name="path", type="string", description="Repository path", required=False, default="."),
        RichParameter(name="remote", type="string", description="Remote name", required=False, default="origin"),
    ]
    examples = ["git_push()", "git_push(remote='upstream')"]
    
    def execute(self, path: str = ".", remote: str = "origin", **kwargs) -> dict[str, Any]:
        try:
            result = subprocess.run(['git', 'push', remote], cwd=str(Path(path).resolve()), capture_output=True, text=True, timeout=120)
            return {"success": result.returncode == 0, "output": result.stderr + result.stdout}
        except Exception as e:
            return {"success": False, "error": str(e)}


class GitPullTool(Tool):
    """Pull from remote."""
    name = "git_pull"
    description = "Pull changes from remote repository."
    parameters = {"path": "Repository path (default: .)", "remote": "Remote name (default: origin)"}
    category = "productivity"
    rich_parameters = [
        RichParameter(name="path", type="string", description="Repository path", required=False, default="."),
        RichParameter(name="remote", type="string", description="Remote name", required=False, default="origin"),
    ]
    examples = ["git_pull()", "git_pull(remote='upstream')"]
    
    def execute(self, path: str = ".", remote: str = "origin", **kwargs) -> dict[str, Any]:
        try:
            result = subprocess.run(['git', 'pull', remote], cwd=str(Path(path).resolve()), capture_output=True, text=True, timeout=120)
            return {"success": result.returncode == 0, "output": result.stdout + result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Export
PRODUCTIVITY_TOOLS = [
    SystemMonitorTool(), ProcessListTool(), ProcessKillTool(),
    SSHExecuteTool(), DockerListTool(), DockerControlTool(),
    GitStatusTool(), GitCommitTool(), GitPushTool(), GitPullTool(),
]
def get_productivity_tools(): return PRODUCTIVITY_TOOLS
