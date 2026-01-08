"""
Productivity Tools - System monitoring, process management, SSH, Docker, Git.

Tools:
  - system_monitor: Check CPU, RAM, disk, temps
  - process_list: List running processes
  - process_kill: Kill a process
  - ssh_execute: Execute commands on remote machines
  - docker_list: List Docker containers
  - docker_control: Start/stop containers
  - git_status: Check git repository status
  - git_commit: Create a git commit
  - git_diff: Show git diff
"""

import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from .tool_registry import Tool


# ============================================================================
# SYSTEM MONITORING TOOLS
# ============================================================================

class SystemMonitorTool(Tool):
    """Monitor system resources."""
    
    name = "system_monitor"
    description = "Get current system status: CPU usage, memory, disk space, and temperature (if available)."
    parameters = {
        "detailed": "Show detailed breakdown (default: False)",
    }
    
    def execute(self, detailed: bool = False, **kwargs) -> Dict[str, Any]:
        try:
            result = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Try psutil for detailed info
            try:
                import psutil
                
                # CPU
                result["cpu"] = {
                    "percent": psutil.cpu_percent(interval=1),
                    "count": psutil.cpu_count(),
                    "count_logical": psutil.cpu_count(logical=True),
                }
                if detailed:
                    result["cpu"]["per_cpu"] = psutil.cpu_percent(interval=1, percpu=True)
                    freq = psutil.cpu_freq()
                    if freq:
                        result["cpu"]["frequency_mhz"] = freq.current
                
                # Memory
                mem = psutil.virtual_memory()
                result["memory"] = {
                    "total_gb": round(mem.total / (1024**3), 2),
                    "available_gb": round(mem.available / (1024**3), 2),
                    "used_gb": round(mem.used / (1024**3), 2),
                    "percent": mem.percent,
                }
                
                # Swap
                swap = psutil.swap_memory()
                result["swap"] = {
                    "total_gb": round(swap.total / (1024**3), 2),
                    "used_gb": round(swap.used / (1024**3), 2),
                    "percent": swap.percent,
                }
                
                # Disk
                disks = []
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        disks.append({
                            "mountpoint": partition.mountpoint,
                            "total_gb": round(usage.total / (1024**3), 2),
                            "used_gb": round(usage.used / (1024**3), 2),
                            "free_gb": round(usage.free / (1024**3), 2),
                            "percent": usage.percent,
                        })
                    except:
                        pass
                result["disks"] = disks
                
                # Temperature (Linux specific)
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        result["temperatures"] = {}
                        for name, entries in temps.items():
                            result["temperatures"][name] = [
                                {"label": e.label or "core", "current": e.current}
                                for e in entries
                            ]
                except:
                    pass
                
                # Network
                if detailed:
                    net = psutil.net_io_counters()
                    result["network"] = {
                        "bytes_sent_mb": round(net.bytes_sent / (1024**2), 2),
                        "bytes_recv_mb": round(net.bytes_recv / (1024**2), 2),
                    }
                
                # Load average (Unix only)
                try:
                    load = os.getloadavg()
                    result["load_average"] = {
                        "1min": round(load[0], 2),
                        "5min": round(load[1], 2),
                        "15min": round(load[2], 2),
                    }
                except:
                    pass
                
                return result
                
            except ImportError:
                pass
            
            # Fallback: use subprocess
            # CPU
            try:
                with open('/proc/loadavg', 'r') as f:
                    load = f.read().split()[:3]
                    result["load_average"] = {
                        "1min": float(load[0]),
                        "5min": float(load[1]),
                        "15min": float(load[2]),
                    }
            except:
                pass
            
            # Memory
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split(':')
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip().split()[0]
                            meminfo[key] = int(value)
                    
                    total = meminfo.get('MemTotal', 0) / 1024 / 1024
                    free = meminfo.get('MemAvailable', meminfo.get('MemFree', 0)) / 1024 / 1024
                    result["memory"] = {
                        "total_gb": round(total, 2),
                        "available_gb": round(free, 2),
                        "percent": round((1 - free/total) * 100, 1) if total else 0,
                    }
            except:
                pass
            
            # Disk
            try:
                df = subprocess.run(['df', '-h', '/'], capture_output=True, text=True, timeout=5)
                if df.returncode == 0:
                    lines = df.stdout.strip().split('\n')
                    if len(lines) > 1:
                        parts = lines[1].split()
                        result["disk_root"] = {
                            "total": parts[1],
                            "used": parts[2],
                            "free": parts[3],
                            "percent": parts[4],
                        }
            except:
                pass
            
            # Raspberry Pi temperature
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = int(f.read().strip()) / 1000
                    result["cpu_temperature_c"] = round(temp, 1)
            except:
                pass
            
            result["note"] = "For more detailed info, install: pip install psutil"
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# PROCESS MANAGEMENT TOOLS
# ============================================================================

class ProcessListTool(Tool):
    """List running processes."""
    
    name = "process_list"
    description = "List running processes with CPU and memory usage."
    parameters = {
        "filter": "Filter by process name (optional)",
        "limit": "Maximum processes to show (default: 20)",
        "sort_by": "Sort by: 'cpu', 'memory', 'name' (default: cpu)",
    }
    
    def execute(self, filter: str = None, limit: int = 20, 
                sort_by: str = "cpu", **kwargs) -> Dict[str, Any]:
        try:
            processes = []
            
            # Try psutil
            try:
                import psutil
                
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                    try:
                        info = proc.info
                        if filter and filter.lower() not in info['name'].lower():
                            continue
                        processes.append({
                            "pid": info['pid'],
                            "name": info['name'],
                            "cpu_percent": round(info['cpu_percent'] or 0, 1),
                            "memory_percent": round(info['memory_percent'] or 0, 1),
                            "status": info['status'],
                        })
                    except:
                        pass
                
                # Sort
                if sort_by == 'cpu':
                    processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
                elif sort_by == 'memory':
                    processes.sort(key=lambda x: x['memory_percent'], reverse=True)
                else:
                    processes.sort(key=lambda x: x['name'].lower())
                
                return {
                    "success": True,
                    "count": len(processes[:int(limit)]),
                    "processes": processes[:int(limit)],
                }
                
            except ImportError:
                pass
            
            # Fallback: use ps
            try:
                cmd = ['ps', 'aux', '--sort=-%cpu']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines[:int(limit)]:
                        parts = line.split(None, 10)
                        if len(parts) >= 11:
                            name = parts[10].split()[0].split('/')[-1]
                            if filter and filter.lower() not in name.lower():
                                continue
                            processes.append({
                                "pid": int(parts[1]),
                                "name": name,
                                "cpu_percent": float(parts[2]),
                                "memory_percent": float(parts[3]),
                            })
                    
                    return {
                        "success": True,
                        "count": len(processes),
                        "processes": processes,
                    }
            except:
                pass
            
            return {"success": False, "error": "Could not list processes"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class ProcessKillTool(Tool):
    """Kill a process."""
    
    name = "process_kill"
    description = "Kill a process by PID or name. Use with caution!"
    parameters = {
        "pid": "Process ID to kill",
        "name": "Process name to kill (alternative to PID)",
        "signal": "Signal: 'TERM' (graceful) or 'KILL' (force). Default: TERM",
    }
    
    def execute(self, pid: int = None, name: str = None, 
                signal: str = "TERM", **kwargs) -> Dict[str, Any]:
        try:
            if not pid and not name:
                return {"success": False, "error": "Must provide either pid or name"}
            
            # Block dangerous process names
            dangerous = ['init', 'systemd', 'kernel', 'bash', 'ssh', 'sshd']
            if name and name.lower() in dangerous:
                return {"success": False, "error": f"Cannot kill system process: {name}"}
            
            killed = []
            
            # Try psutil
            try:
                import psutil
                import signal as sig
                
                signal_map = {
                    'TERM': sig.SIGTERM,
                    'KILL': sig.SIGKILL,
                }
                sig_num = signal_map.get(signal.upper(), sig.SIGTERM)
                
                if pid:
                    proc = psutil.Process(int(pid))
                    proc_name = proc.name()
                    if proc_name.lower() in dangerous:
                        return {"success": False, "error": f"Cannot kill system process: {proc_name}"}
                    proc.send_signal(sig_num)
                    killed.append({"pid": pid, "name": proc_name})
                else:
                    for proc in psutil.process_iter(['pid', 'name']):
                        try:
                            if proc.info['name'].lower() == name.lower():
                                proc.send_signal(sig_num)
                                killed.append({"pid": proc.info['pid'], "name": proc.info['name']})
                        except:
                            pass
                
                return {
                    "success": True,
                    "killed": killed,
                    "signal": signal,
                }
                
            except ImportError:
                pass
            
            # Fallback: use kill command
            sig_map = {'TERM': '-15', 'KILL': '-9'}
            sig_flag = sig_map.get(signal.upper(), '-15')
            
            if pid:
                result = subprocess.run(['kill', sig_flag, str(pid)], capture_output=True, text=True)
                if result.returncode == 0:
                    return {"success": True, "killed": [{"pid": pid}], "signal": signal}
                else:
                    return {"success": False, "error": result.stderr}
            else:
                result = subprocess.run(['pkill', sig_flag, name], capture_output=True, text=True)
                return {"success": result.returncode == 0, "signal": signal}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# SSH TOOLS
# ============================================================================

class SSHExecuteTool(Tool):
    """Execute commands on remote machines via SSH."""
    
    name = "ssh_execute"
    description = "Execute a command on a remote machine via SSH."
    parameters = {
        "host": "Remote host (e.g., 'user@192.168.1.100' or just hostname if in SSH config)",
        "command": "Command to execute remotely",
        "timeout": "Timeout in seconds (default: 30)",
        "key_file": "Path to SSH key file (optional)",
    }
    
    def execute(self, host: str, command: str, timeout: int = 30, 
                key_file: str = None, **kwargs) -> Dict[str, Any]:
        try:
            # Build SSH command
            cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new']
            
            if key_file:
                key_path = Path(key_file).expanduser()
                if key_path.exists():
                    cmd.extend(['-i', str(key_path)])
            
            cmd.extend([host, command])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout))
            
            return {
                "success": result.returncode == 0,
                "host": host,
                "command": command,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"SSH command timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class SSHCopyTool(Tool):
    """Copy files to/from remote machines via SCP."""
    
    name = "ssh_copy"
    description = "Copy files to or from a remote machine using SCP."
    parameters = {
        "source": "Source path (use 'user@host:path' for remote)",
        "destination": "Destination path (use 'user@host:path' for remote)",
        "recursive": "Copy directories recursively (default: False)",
    }
    
    def execute(self, source: str, destination: str, 
                recursive: bool = False, **kwargs) -> Dict[str, Any]:
        try:
            cmd = ['scp', '-o', 'BatchMode=yes']
            
            if recursive:
                cmd.append('-r')
            
            cmd.extend([source, destination])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            return {
                "success": result.returncode == 0,
                "source": source,
                "destination": destination,
                "stderr": result.stderr if result.returncode != 0 else None,
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "SCP timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# DOCKER TOOLS
# ============================================================================

class DockerListTool(Tool):
    """List Docker containers."""
    
    name = "docker_list"
    description = "List Docker containers (running and stopped)."
    parameters = {
        "all": "Show all containers, not just running (default: True)",
    }
    
    def execute(self, all: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            cmd = ['docker', 'ps', '--format', '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}']
            if all:
                cmd.insert(2, '-a')
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return {"success": False, "error": result.stderr or "Docker not available"}
            
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        containers.append({
                            "id": parts[0],
                            "name": parts[1],
                            "image": parts[2],
                            "status": parts[3],
                            "ports": parts[4] if len(parts) > 4 else "",
                        })
            
            return {
                "success": True,
                "count": len(containers),
                "containers": containers,
            }
            
        except FileNotFoundError:
            return {"success": False, "error": "Docker is not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class DockerControlTool(Tool):
    """Control Docker containers."""
    
    name = "docker_control"
    description = "Start, stop, restart, or remove a Docker container."
    parameters = {
        "container": "Container name or ID",
        "action": "Action: 'start', 'stop', 'restart', 'remove', 'logs'",
        "tail": "For logs: number of lines to show (default: 100)",
    }
    
    def execute(self, container: str, action: str, tail: int = 100, **kwargs) -> Dict[str, Any]:
        try:
            actions = ['start', 'stop', 'restart', 'remove', 'logs']
            if action not in actions:
                return {"success": False, "error": f"Invalid action. Use: {actions}"}
            
            if action == 'logs':
                cmd = ['docker', 'logs', '--tail', str(tail), container]
            elif action == 'remove':
                cmd = ['docker', 'rm', '-f', container]
            else:
                cmd = ['docker', action, container]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            return {
                "success": result.returncode == 0,
                "container": container,
                "action": action,
                "output": result.stdout if action == 'logs' else None,
                "error": result.stderr if result.returncode != 0 else None,
            }
            
        except FileNotFoundError:
            return {"success": False, "error": "Docker is not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class DockerImagesTool(Tool):
    """List Docker images."""
    
    name = "docker_images"
    description = "List Docker images on the system."
    parameters = {}
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            cmd = ['docker', 'images', '--format', '{{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return {"success": False, "error": result.stderr or "Docker not available"}
            
            images = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        images.append({
                            "repository": parts[0],
                            "tag": parts[1],
                            "id": parts[2],
                            "size": parts[3],
                        })
            
            return {
                "success": True,
                "count": len(images),
                "images": images,
            }
            
        except FileNotFoundError:
            return {"success": False, "error": "Docker is not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# GIT TOOLS
# ============================================================================

class GitStatusTool(Tool):
    """Check git repository status."""
    
    name = "git_status"
    description = "Check the status of a git repository."
    parameters = {
        "path": "Path to the repository (default: current directory)",
    }
    
    def execute(self, path: str = ".", **kwargs) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            
            # Check if it's a git repo
            git_dir = path / '.git'
            if not git_dir.exists():
                return {"success": False, "error": "Not a git repository"}
            
            # Get branch
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=str(path), capture_output=True, text=True, timeout=10
            )
            branch = result.stdout.strip() if result.returncode == 0 else "unknown"
            
            # Get status
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=str(path), capture_output=True, text=True, timeout=10
            )
            
            staged = []
            modified = []
            untracked = []
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                status = line[:2]
                file = line[3:]
                
                if status[0] in 'MADRC':
                    staged.append(file)
                if status[1] == 'M':
                    modified.append(file)
                if status == '??':
                    untracked.append(file)
            
            # Get last commit
            result = subprocess.run(
                ['git', 'log', '-1', '--pretty=format:%h %s (%cr)'],
                cwd=str(path), capture_output=True, text=True, timeout=10
            )
            last_commit = result.stdout.strip() if result.returncode == 0 else ""
            
            # Get remote status
            result = subprocess.run(
                ['git', 'status', '-sb'],
                cwd=str(path), capture_output=True, text=True, timeout=10
            )
            ahead_behind = ""
            if '[' in result.stdout:
                ahead_behind = result.stdout.split('[')[1].split(']')[0]
            
            return {
                "success": True,
                "path": str(path),
                "branch": branch,
                "staged": staged,
                "modified": modified,
                "untracked": untracked,
                "clean": len(staged) == 0 and len(modified) == 0,
                "last_commit": last_commit,
                "ahead_behind": ahead_behind,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class GitCommitTool(Tool):
    """Create a git commit."""
    
    name = "git_commit"
    description = "Stage and commit changes in a git repository."
    parameters = {
        "message": "Commit message",
        "path": "Path to the repository (default: current directory)",
        "add_all": "Add all changes before committing (default: True)",
    }
    
    def execute(self, message: str, path: str = ".", add_all: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            
            # Add changes
            if add_all:
                result = subprocess.run(
                    ['git', 'add', '-A'],
                    cwd=str(path), capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0:
                    return {"success": False, "error": f"Git add failed: {result.stderr}"}
            
            # Commit
            result = subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=str(path), capture_output=True, text=True, timeout=30
            )
            
            if result.returncode != 0:
                if 'nothing to commit' in result.stdout + result.stderr:
                    return {"success": True, "message": "Nothing to commit", "committed": False}
                return {"success": False, "error": result.stderr}
            
            # Get commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=str(path), capture_output=True, text=True, timeout=10
            )
            commit_hash = result.stdout.strip()[:7] if result.returncode == 0 else ""
            
            return {
                "success": True,
                "committed": True,
                "message": message,
                "hash": commit_hash,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class GitDiffTool(Tool):
    """Show git diff."""
    
    name = "git_diff"
    description = "Show changes (diff) in a git repository."
    parameters = {
        "path": "Path to the repository (default: current directory)",
        "file": "Specific file to diff (optional)",
        "staged": "Show staged changes only (default: False)",
    }
    
    def execute(self, path: str = ".", file: str = None, 
                staged: bool = False, **kwargs) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            
            cmd = ['git', 'diff']
            if staged:
                cmd.append('--staged')
            if file:
                cmd.append(file)
            
            result = subprocess.run(cmd, cwd=str(path), capture_output=True, text=True, timeout=30)
            
            diff = result.stdout
            
            # Parse stats
            additions = diff.count('\n+') - diff.count('\n+++')
            deletions = diff.count('\n-') - diff.count('\n---')
            
            return {
                "success": True,
                "diff": diff[:10000],  # Limit size
                "truncated": len(diff) > 10000,
                "additions": additions,
                "deletions": deletions,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class GitPushTool(Tool):
    """Push commits to remote."""
    
    name = "git_push"
    description = "Push commits to the remote repository."
    parameters = {
        "path": "Path to the repository (default: current directory)",
        "remote": "Remote name (default: origin)",
        "branch": "Branch to push (default: current branch)",
    }
    
    def execute(self, path: str = ".", remote: str = "origin", 
                branch: str = None, **kwargs) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            
            cmd = ['git', 'push', remote]
            if branch:
                cmd.append(branch)
            
            result = subprocess.run(cmd, cwd=str(path), capture_output=True, text=True, timeout=120)
            
            return {
                "success": result.returncode == 0,
                "remote": remote,
                "output": result.stderr + result.stdout,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class GitPullTool(Tool):
    """Pull changes from remote."""
    
    name = "git_pull"
    description = "Pull changes from the remote repository."
    parameters = {
        "path": "Path to the repository (default: current directory)",
        "remote": "Remote name (default: origin)",
        "branch": "Branch to pull (default: current branch)",
    }
    
    def execute(self, path: str = ".", remote: str = "origin", 
                branch: str = None, **kwargs) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser().resolve()
            
            cmd = ['git', 'pull', remote]
            if branch:
                cmd.append(branch)
            
            result = subprocess.run(cmd, cwd=str(path), capture_output=True, text=True, timeout=120)
            
            return {
                "success": result.returncode == 0,
                "remote": remote,
                "output": result.stdout + result.stderr,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
