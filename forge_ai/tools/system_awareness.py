"""
System Awareness - Know everything about the environment.

Provides the AI with awareness of:
- System state (CPU, memory, disk, network)
- Running processes
- Time/date
- File system
- Hardware capabilities
"""

import os
import sys
import time
import socket
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """Current system information."""
    os_name: str
    os_version: str
    hostname: str
    architecture: str
    python_version: str
    cpu_count: int
    total_memory_gb: float
    available_memory_gb: float
    disk_total_gb: float
    disk_free_gb: float
    uptime_hours: float
    load_average: Tuple[float, float, float]
    network_interfaces: List[str]
    gpu_available: bool
    gpu_name: Optional[str] = None


@dataclass 
class ProcessInfo:
    """Information about a running process."""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_mb: float
    command: str
    user: str
    started: Optional[datetime] = None


@dataclass
class TimeInfo:
    """Current time information."""
    now: datetime
    timezone: str
    utc_offset: str
    day_of_week: str
    is_weekend: bool
    is_business_hours: bool
    unix_timestamp: float
    uptime_seconds: float


class SystemAwareness:
    """
    Provides comprehensive system awareness for the AI.
    
    The AI can ask about anything regarding the system state,
    running processes, time, files, and hardware.
    """
    
    def __init__(self):
        self._boot_time: Optional[float] = None
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl = 5.0  # seconds
        
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return value
        return None
        
    def _set_cached(self, key: str, value: Any) -> Any:
        """Cache a value."""
        self._cache[key] = (time.time(), value)
        return value
    
    # ========== System State ==========
    
    def get_system_info(self) -> SystemInfo:
        """Get comprehensive system information."""
        cached = self._get_cached('system_info')
        if cached:
            return cached
            
        # Basic info
        os_name = platform.system()
        os_version = platform.release()
        hostname = socket.gethostname()
        arch = platform.machine()
        python_ver = platform.python_version()
        cpu_count = os.cpu_count() or 1
        
        # Memory
        total_mem = 0.0
        avail_mem = 0.0
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        total_mem = int(line.split()[1]) / (1024 * 1024)
                    elif line.startswith('MemAvailable:'):
                        avail_mem = int(line.split()[1]) / (1024 * 1024)
        except Exception:
            pass
            
        # Disk
        disk_total = 0.0
        disk_free = 0.0
        try:
            statvfs = os.statvfs('/')
            disk_total = (statvfs.f_blocks * statvfs.f_frsize) / (1024**3)
            disk_free = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)
        except Exception:
            pass
            
        # Uptime
        uptime_hours = 0.0
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_hours = float(f.read().split()[0]) / 3600
        except Exception:
            pass
            
        # Load average
        load_avg = (0.0, 0.0, 0.0)
        try:
            load_avg = os.getloadavg()
        except Exception:
            pass
            
        # Network interfaces
        interfaces = []
        try:
            result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True, timeout=5)
            for line in result.stdout.split('\n'):
                if ':' in line and '@' not in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        iface = parts[1].strip()
                        if iface and not iface.startswith(' '):
                            interfaces.append(iface)
        except Exception:
            interfaces = ['unknown']
            
        # GPU
        gpu_available = False
        gpu_name = None
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_available = True
                gpu_name = result.stdout.strip()
        except Exception:
            pass
            
        info = SystemInfo(
            os_name=os_name,
            os_version=os_version,
            hostname=hostname,
            architecture=arch,
            python_version=python_ver,
            cpu_count=cpu_count,
            total_memory_gb=round(total_mem, 2),
            available_memory_gb=round(avail_mem, 2),
            disk_total_gb=round(disk_total, 2),
            disk_free_gb=round(disk_free, 2),
            uptime_hours=round(uptime_hours, 2),
            load_average=load_avg,
            network_interfaces=interfaces,
            gpu_available=gpu_available,
            gpu_name=gpu_name
        )
        
        return self._set_cached('system_info', info)
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()
                parts = line.split()
                idle = int(parts[4])
                total = sum(int(p) for p in parts[1:])
                
            time.sleep(0.1)
            
            with open('/proc/stat', 'r') as f:
                line = f.readline()
                parts = line.split()
                idle2 = int(parts[4])
                total2 = sum(int(p) for p in parts[1:])
                
            idle_delta = idle2 - idle
            total_delta = total2 - total
            
            if total_delta > 0:
                return round(100.0 * (1 - idle_delta / total_delta), 1)
        except Exception:
            pass
        return 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage details."""
        result = {
            'total_gb': 0.0,
            'used_gb': 0.0,
            'available_gb': 0.0,
            'percent_used': 0.0
        }
        try:
            with open('/proc/meminfo', 'r') as f:
                mem = {}
                for line in f:
                    parts = line.split()
                    mem[parts[0].rstrip(':')] = int(parts[1])
                    
            total = mem.get('MemTotal', 0) / (1024 * 1024)
            available = mem.get('MemAvailable', 0) / (1024 * 1024)
            used = total - available
            
            result['total_gb'] = round(total, 2)
            result['used_gb'] = round(used, 2)
            result['available_gb'] = round(available, 2)
            result['percent_used'] = round(100 * used / total, 1) if total > 0 else 0.0
        except Exception:
            pass
        return result
    
    def get_disk_usage(self, path: str = '/') -> Dict[str, float]:
        """Get disk usage for a path."""
        result = {
            'total_gb': 0.0,
            'used_gb': 0.0,
            'free_gb': 0.0,
            'percent_used': 0.0
        }
        try:
            statvfs = os.statvfs(path)
            total = (statvfs.f_blocks * statvfs.f_frsize) / (1024**3)
            free = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)
            used = total - free
            
            result['total_gb'] = round(total, 2)
            result['used_gb'] = round(used, 2)
            result['free_gb'] = round(free, 2)
            result['percent_used'] = round(100 * used / total, 1) if total > 0 else 0.0
        except Exception:
            pass
        return result
    
    def is_online(self) -> bool:
        """Check if the system has internet connectivity."""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        info = {
            'online': self.is_online(),
            'hostname': socket.gethostname(),
            'local_ip': None,
            'interfaces': []
        }
        
        try:
            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            info['local_ip'] = s.getsockname()[0]
            s.close()
        except Exception:
            pass
            
        try:
            result = subprocess.run(['ip', 'addr'], capture_output=True, text=True, timeout=5)
            info['interfaces_raw'] = result.stdout
        except Exception:
            pass
            
        return info
    
    # ========== Process Awareness ==========
    
    def get_processes(self, sort_by: str = 'memory', limit: int = 20) -> List[ProcessInfo]:
        """Get list of running processes."""
        processes = []
        
        try:
            # Use ps command for process info
            result = subprocess.run(
                ['ps', 'aux', '--sort=-%mem'],
                capture_output=True, text=True, timeout=10
            )
            
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines[:limit * 2]:  # Get extra, filter later
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    try:
                        proc = ProcessInfo(
                            pid=int(parts[1]),
                            name=parts[10].split()[0] if parts[10] else '',
                            status='running',
                            cpu_percent=float(parts[2]),
                            memory_mb=float(parts[5]) / 1024,  # RSS in KB
                            command=parts[10],
                            user=parts[0]
                        )
                        processes.append(proc)
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as e:
            logger.warning(f"Failed to get processes: {e}")
            
        # Sort
        if sort_by == 'cpu':
            processes.sort(key=lambda p: p.cpu_percent, reverse=True)
        elif sort_by == 'memory':
            processes.sort(key=lambda p: p.memory_mb, reverse=True)
        elif sort_by == 'name':
            processes.sort(key=lambda p: p.name.lower())
            
        return processes[:limit]
    
    def find_process(self, name: str) -> List[ProcessInfo]:
        """Find processes by name."""
        all_procs = self.get_processes(limit=100)
        return [p for p in all_procs if name.lower() in p.name.lower()]
    
    def is_process_running(self, name: str) -> bool:
        """Check if a process is running."""
        return len(self.find_process(name)) > 0
    
    def get_python_processes(self) -> List[ProcessInfo]:
        """Get all Python processes."""
        return self.find_process('python')
    
    # ========== Time Awareness ==========
    
    def get_time_info(self) -> TimeInfo:
        """Get comprehensive time information."""
        now = datetime.now()
        
        # Timezone
        tz_name = time.tzname[time.daylight] if time.daylight else time.tzname[0]
        utc_offset = time.strftime('%z')
        
        # Day info
        day_of_week = now.strftime('%A')
        is_weekend = now.weekday() >= 5
        
        # Business hours (9 AM - 5 PM weekdays)
        is_business = not is_weekend and 9 <= now.hour < 17
        
        # Uptime
        uptime = 0.0
        try:
            with open('/proc/uptime', 'r') as f:
                uptime = float(f.read().split()[0])
        except Exception:
            pass
            
        return TimeInfo(
            now=now,
            timezone=tz_name,
            utc_offset=utc_offset,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            is_business_hours=is_business,
            unix_timestamp=now.timestamp(),
            uptime_seconds=uptime
        )
    
    def get_current_time(self) -> str:
        """Get current time as readable string."""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def get_current_date(self) -> str:
        """Get current date."""
        return datetime.now().strftime('%Y-%m-%d')
    
    def time_until(self, target: datetime) -> timedelta:
        """Get time remaining until a target datetime."""
        return target - datetime.now()
    
    def time_since(self, target: datetime) -> timedelta:
        """Get time elapsed since a target datetime."""
        return datetime.now() - target
    
    # ========== File System Awareness ==========
    
    def list_directory(self, path: str, include_hidden: bool = False) -> List[Dict[str, Any]]:
        """List contents of a directory."""
        items = []
        try:
            p = Path(path)
            for item in p.iterdir():
                if not include_hidden and item.name.startswith('.'):
                    continue
                    
                stat = item.stat()
                items.append({
                    'name': item.name,
                    'path': str(item),
                    'is_dir': item.is_dir(),
                    'size_bytes': stat.st_size if item.is_file() else 0,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'permissions': oct(stat.st_mode)[-3:]
                })
        except Exception as e:
            logger.warning(f"Failed to list {path}: {e}")
            
        return sorted(items, key=lambda x: (not x['is_dir'], x['name'].lower()))
    
    def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        return Path(path).exists()
    
    def get_file_info(self, path: str) -> Optional[Dict[str, Any]]:
        """Get information about a file."""
        try:
            p = Path(path)
            if not p.exists():
                return None
                
            stat = p.stat()
            return {
                'path': str(p.absolute()),
                'name': p.name,
                'extension': p.suffix,
                'is_dir': p.is_dir(),
                'is_file': p.is_file(),
                'size_bytes': stat.st_size,
                'size_readable': self._format_size(stat.st_size),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
                'permissions': oct(stat.st_mode)[-3:],
                'owner_uid': stat.st_uid
            }
        except Exception as e:
            logger.warning(f"Failed to get file info for {path}: {e}")
            return None
    
    def _format_size(self, size: int) -> str:
        """Format size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    def find_files(self, directory: str, pattern: str, recursive: bool = True) -> List[str]:
        """Find files matching a pattern."""
        results = []
        try:
            p = Path(directory)
            if recursive:
                matches = p.rglob(pattern)
            else:
                matches = p.glob(pattern)
            results = [str(m) for m in matches][:100]  # Limit results
        except Exception as e:
            logger.warning(f"Failed to find files: {e}")
        return results
    
    # ========== Hardware Awareness ==========
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        info = {
            'cpu': {},
            'memory': {},
            'storage': [],
            'gpu': None,
            'audio': [],
            'usb': []
        }
        
        # CPU info
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        info['cpu']['model'] = line.split(':')[1].strip()
                        break
            info['cpu']['cores'] = os.cpu_count()
        except Exception:
            pass
            
        # Memory
        info['memory'] = self.get_memory_usage()
        
        # Storage devices
        try:
            result = subprocess.run(['lsblk', '-J'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                info['storage'] = data.get('blockdevices', [])
        except Exception:
            pass
            
        # GPU
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,temperature.gpu', 
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                info['gpu'] = {
                    'name': parts[0].strip(),
                    'memory_total': parts[1].strip() if len(parts) > 1 else None,
                    'memory_free': parts[2].strip() if len(parts) > 2 else None,
                    'temperature': parts[3].strip() if len(parts) > 3 else None
                }
        except Exception:
            pass
            
        # USB devices
        try:
            result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info['usb'] = result.stdout.strip().split('\n')
        except Exception:
            pass
            
        return info
    
    # ========== Summary / Natural Language ==========
    
    def describe(self) -> str:
        """Get a natural language description of the system state."""
        sys_info = self.get_system_info()
        time_info = self.get_time_info()
        mem = self.get_memory_usage()
        
        parts = [
            f"It's {time_info.day_of_week}, {time_info.now.strftime('%B %d, %Y at %I:%M %p')}.",
            f"Running on {sys_info.hostname} ({sys_info.os_name} {sys_info.os_version}, {sys_info.architecture}).",
            f"CPU: {sys_info.cpu_count} cores, load: {sys_info.load_average[0]:.1f}.",
            f"Memory: {mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB used ({mem['percent_used']:.0f}%).",
            f"Disk: {sys_info.disk_free_gb:.1f}/{sys_info.disk_total_gb:.1f} GB free.",
        ]
        
        if sys_info.gpu_available:
            parts.append(f"GPU: {sys_info.gpu_name}.")
            
        if self.is_online():
            parts.append("Network: Online.")
        else:
            parts.append("Network: Offline.")
            
        return " ".join(parts)
    
    def answer(self, question: str) -> str:
        """
        Answer a natural language question about the system.
        
        Examples:
        - "What time is it?"
        - "How much memory is available?"
        - "Is python running?"
        - "What's my IP address?"
        """
        q = question.lower()
        
        # Time questions
        if any(w in q for w in ['time', 'clock', 'hour']):
            return f"The current time is {self.get_current_time()}"
            
        if any(w in q for w in ['date', 'today', 'day']):
            ti = self.get_time_info()
            return f"Today is {ti.day_of_week}, {ti.now.strftime('%B %d, %Y')}"
            
        # Memory questions
        if 'memory' in q or 'ram' in q:
            mem = self.get_memory_usage()
            return f"Memory: {mem['used_gb']:.1f} GB used of {mem['total_gb']:.1f} GB ({mem['percent_used']:.0f}% used, {mem['available_gb']:.1f} GB available)"
            
        # CPU questions
        if 'cpu' in q or 'processor' in q or 'load' in q:
            si = self.get_system_info()
            usage = self.get_cpu_usage()
            return f"CPU: {si.cpu_count} cores, {usage}% usage, load average: {si.load_average[0]:.2f}, {si.load_average[1]:.2f}, {si.load_average[2]:.2f}"
            
        # Disk questions
        if 'disk' in q or 'storage' in q or 'space' in q:
            disk = self.get_disk_usage()
            return f"Disk: {disk['free_gb']:.1f} GB free of {disk['total_gb']:.1f} GB ({disk['percent_used']:.0f}% used)"
            
        # Network questions
        if 'network' in q or 'internet' in q or 'online' in q or 'ip' in q:
            net = self.get_network_info()
            status = "online" if net['online'] else "offline"
            ip = net.get('local_ip', 'unknown')
            return f"Network: {status}, local IP: {ip}, hostname: {net['hostname']}"
            
        # Process questions
        if 'process' in q or 'running' in q:
            # Check if asking about specific process
            for word in q.split():
                if word not in ['is', 'running', 'process', 'processes', 'the', 'a', 'are']:
                    procs = self.find_process(word)
                    if procs:
                        return f"Yes, {len(procs)} {word} process(es) running: {', '.join(p.name for p in procs[:5])}"
                    else:
                        return f"No {word} processes found running"
            # General process info
            procs = self.get_processes(limit=5)
            return f"Top processes by memory: {', '.join(f'{p.name}({p.memory_mb:.0f}MB)' for p in procs)}"
            
        # GPU questions
        if 'gpu' in q or 'graphics' in q or 'nvidia' in q:
            si = self.get_system_info()
            if si.gpu_available:
                return f"GPU: {si.gpu_name}"
            return "No GPU detected"
            
        # System overview
        if 'system' in q or 'status' in q or 'overview' in q:
            return self.describe()
            
        # Uptime
        if 'uptime' in q or 'how long' in q:
            si = self.get_system_info()
            return f"System uptime: {si.uptime_hours:.1f} hours"
            
        # Hostname
        if 'hostname' in q or 'name' in q:
            return f"Hostname: {socket.gethostname()}"
            
        # OS
        if 'os' in q or 'operating' in q or 'linux' in q or 'windows' in q:
            si = self.get_system_info()
            return f"Operating System: {si.os_name} {si.os_version} ({si.architecture})"
            
        # Python
        if 'python' in q and 'version' in q:
            return f"Python version: {platform.python_version()}"
            
        # Default
        return self.describe()


# Singleton instance
_awareness: Optional[SystemAwareness] = None

def get_awareness() -> SystemAwareness:
    """Get the system awareness singleton."""
    global _awareness
    if _awareness is None:
        _awareness = SystemAwareness()
    return _awareness


# Convenience functions
def what_time() -> str:
    """Get current time."""
    return get_awareness().get_current_time()

def what_date() -> str:
    """Get current date."""
    return get_awareness().get_current_date()

def system_status() -> str:
    """Get system status description."""
    return get_awareness().describe()

def ask_system(question: str) -> str:
    """Ask a question about the system."""
    return get_awareness().answer(question)

def is_process_running(name: str) -> bool:
    """Check if a process is running."""
    return get_awareness().is_process_running(name)

def memory_available() -> float:
    """Get available memory in GB."""
    return get_awareness().get_memory_usage()['available_gb']

def disk_free() -> float:
    """Get free disk space in GB."""
    return get_awareness().get_disk_usage()['free_gb']

def is_online() -> bool:
    """Check internet connectivity."""
    return get_awareness().is_online()
