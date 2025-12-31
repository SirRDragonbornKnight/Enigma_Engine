"""
Hardware Detection and Adaptation

Automatically detects available hardware and configures Enigma accordingly.
Works on: PC, Mac, Linux, Raspberry Pi, Android (via Termux/Pydroid), any device.
"""

import os
import sys
import platform
import subprocess
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class HardwareProfile:
    """
    Detects and stores hardware capabilities.
    """

    def __init__(self):
        self.profile = self._detect_all()

    def _detect_all(self) -> Dict[str, Any]:
        """Detect all hardware characteristics."""
        return {
            "platform": self._detect_platform(),
            "cpu": self._detect_cpu(),
            "memory": self._detect_memory(),
            "gpu": self._detect_gpu(),
            "storage": self._detect_storage(),
            "display": self._detect_display(),
            "recommended_model_size": self._recommend_model_size(),
        }

    def _detect_platform(self) -> Dict[str, Any]:
        """Detect OS and platform type."""
        system = platform.system().lower()
        machine = platform.machine().lower()

        # Detect specific platforms
        is_raspberry_pi = False
        is_android = False
        is_ios = False
        is_wsl = False

        # Check for Raspberry Pi
        if system == "linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    if "raspberry" in cpuinfo.lower() or "bcm" in cpuinfo.lower():
                        is_raspberry_pi = True
            except BaseException:
                pass

            # Check for Android (Termux)
            if "ANDROID_ROOT" in os.environ or "TERMUX_VERSION" in os.environ:
                is_android = True

            # Check for WSL
            try:
                with open("/proc/version", "r") as f:
                    if "microsoft" in f.read().lower():
                        is_wsl = True
            except BaseException:
                pass

        # iOS detection (via Pythonista or similar)
        if system == "darwin" and "iP" in platform.platform():
            is_ios = True

        return {
            "system": system,
            "machine": machine,
            "python_version": platform.python_version(),
            "is_raspberry_pi": is_raspberry_pi,
            "is_android": is_android,
            "is_ios": is_ios,
            "is_wsl": is_wsl,
            "is_mobile": is_android or is_ios,
            "is_64bit": sys.maxsize > 2**32,
        }

    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information."""
        cpu_count = os.cpu_count() or 1

        # Try to get CPU model
        cpu_model = "Unknown"
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line.lower():
                            cpu_model = line.split(":")[1].strip()
                            break
            elif platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True
                )
                cpu_model = result.stdout.strip()
            elif platform.system() == "Windows":
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                     r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        except BaseException:
            pass

        return {
            "cores": cpu_count,
            "model": cpu_model,
            "architecture": platform.machine(),
        }

    def _detect_memory(self) -> Dict[str, Any]:
        """Detect RAM information."""
        total_ram = 0
        available_ram = 0

        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if "MemTotal" in line:
                            total_ram = int(line.split()[1]) * 1024  # Convert KB to bytes
                        elif "MemAvailable" in line:
                            available_ram = int(line.split()[1]) * 1024
            elif platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True
                )
                total_ram = int(result.stdout.strip())
            elif platform.system() == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulonglong = ctypes.c_ulonglong

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', c_ulonglong),
                        ('ullAvailPhys', c_ulonglong),
                        ('ullTotalPageFile', c_ulonglong),
                        ('ullAvailPageFile', c_ulonglong),
                        ('ullTotalVirtual', c_ulonglong),
                        ('ullAvailVirtual', c_ulonglong),
                        ('ullAvailExtendedVirtual', c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                total_ram = stat.ullTotalPhys
                available_ram = stat.ullAvailPhys
        except BaseException:
            pass

        # Convert to GB for readability
        total_gb = total_ram / (1024**3)
        available_gb = available_ram / (1024**3)

        return {
            "total_bytes": total_ram,
            "available_bytes": available_ram,
            "total_gb": round(total_gb, 1),
            "available_gb": round(available_gb, 1),
        }

    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU and CUDA availability."""
        gpu_info = {
            "available": False,
            "cuda_available": False,
            "mps_available": False,  # Apple Silicon
            "name": None,
            "vram_gb": 0,
            "cuda_version": None,
        }

        # Check for CUDA (NVIDIA)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["cuda_available"] = True
                gpu_info["name"] = torch.cuda.get_device_name(0)
                gpu_info["vram_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
                )
                gpu_info["cuda_version"] = torch.version.cuda

            # Check for Apple Silicon MPS
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info["available"] = True
                gpu_info["mps_available"] = True
                gpu_info["name"] = "Apple Silicon (MPS)"
        except ImportError:
            pass

        # Fallback: check nvidia-smi
        if not gpu_info["available"]:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    gpu_info["available"] = True
                    gpu_info["name"] = parts[0].strip()
                    vram_str = parts[1].strip().replace("MiB", "").strip()
                    gpu_info["vram_gb"] = round(int(vram_str) / 1024, 1)
            except BaseException:
                pass

        return gpu_info

    def _detect_storage(self) -> Dict[str, Any]:
        """Detect storage information."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(Path.home())
            return {
                "total_gb": round(total / (1024**3), 1),
                "used_gb": round(used / (1024**3), 1),
                "free_gb": round(free / (1024**3), 1),
            }
        except BaseException:
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0}

    def _detect_display(self) -> Dict[str, Any]:
        """Detect display capabilities."""
        display = {
            "available": False,
            "width": 0,
            "height": 0,
            "headless": True,
        }

        # Check for display
        if platform.system() == "Linux":
            display["available"] = "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ
        elif platform.system() == "Darwin":
            display["available"] = True  # macOS always has display
        elif platform.system() == "Windows":
            display["available"] = True

        display["headless"] = not display["available"]

        # Try to get resolution
        try:
            if display["available"]:
                # Try PyQt5 first
                try:
                    from PyQt5.QtWidgets import QApplication
                    app = QApplication.instance() or QApplication([])
                    screen = app.primaryScreen()
                    size = screen.size()
                    display["width"] = size.width()
                    display["height"] = size.height()
                except BaseException:
                    pass

                # Fallback: xrandr on Linux
                if display["width"] == 0 and platform.system() == "Linux":
                    try:
                        result = subprocess.run(
                            ["xrandr", "--current"],
                            capture_output=True, text=True
                        )
                        for line in result.stdout.split("\n"):
                            if " connected" in line and "x" in line:
                                # Parse resolution like "1920x1080"
                                import re
                                match = re.search(r'(\d+)x(\d+)', line)
                                if match:
                                    display["width"] = int(match.group(1))
                                    display["height"] = int(match.group(2))
                                    break
                    except BaseException:
                        pass
        except BaseException:
            pass

        return display

    def _recommend_model_size(self) -> str:
        """Recommend optimal model size based on hardware."""
        memory = self.profile.get(
            "memory", {}) if hasattr(
            self, 'profile') else self._detect_memory()
        gpu = self.profile.get("gpu", {}) if hasattr(self, 'profile') else self._detect_gpu()
        platform_info = self.profile.get(
            "platform", {}) if hasattr(
            self, 'profile') else self._detect_platform()

        ram_gb = memory.get("total_gb", 0)
        vram_gb = gpu.get("vram_gb", 0)
        is_mobile = platform_info.get("is_mobile", False)

        # Mobile devices: always tiny or small
        if is_mobile:
            return "tiny" if ram_gb < 4 else "small"

        # Use VRAM if GPU available, else RAM
        if gpu.get("cuda_available") or gpu.get("mps_available"):
            effective_memory = vram_gb
        else:
            effective_memory = ram_gb * 0.5  # Can use ~50% of RAM for model

        # Recommend based on memory
        if effective_memory >= 16:
            return "xl"
        elif effective_memory >= 8:
            return "large"
        elif effective_memory >= 4:
            return "medium"
        elif effective_memory >= 2:
            return "small"
        else:
            return "tiny"

    def get_device(self) -> str:
        """Get the best PyTorch device for this hardware."""
        gpu = self.profile.get("gpu", {})

        if gpu.get("cuda_available"):
            return "cuda"
        elif gpu.get("mps_available"):
            return "mps"
        else:
            return "cpu"

    def summary(self) -> str:
        """Get a human-readable summary."""
        p = self.profile
        plat = p["platform"]

        device_type = "Unknown"
        if plat["is_android"]:
            device_type = "Android Phone/Tablet"
        elif plat["is_ios"]:
            device_type = "iPhone/iPad"
        elif plat["is_raspberry_pi"]:
            device_type = "Raspberry Pi"
        elif plat["system"] == "darwin":
            device_type = "Mac"
        elif plat["system"] == "windows":
            device_type = "Windows PC"
        elif plat["system"] == "linux":
            device_type = "Linux PC"

        lines = [
            f"Device: {device_type}",
            f"CPU: {p['cpu']['model']} ({p['cpu']['cores']} cores)",
            f"RAM: {p['memory']['total_gb']} GB",
        ]

        if p["gpu"]["available"]:
            lines.append(f"GPU: {p['gpu']['name']} ({p['gpu']['vram_gb']} GB VRAM)")
        else:
            lines.append("GPU: None (CPU only)")

        lines.append(f"Recommended Model: {p['recommended_model_size']}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Get full profile as dictionary."""
        return self.profile


# Global instance
_hardware_profile: Optional[HardwareProfile] = None


def get_hardware() -> HardwareProfile:
    """Get or create the hardware profile singleton."""
    global _hardware_profile
    if _hardware_profile is None:
        _hardware_profile = HardwareProfile()
    return _hardware_profile


def get_best_device() -> str:
    """Quick helper to get best device string."""
    return get_hardware().get_device()


def get_recommended_model_size() -> str:
    """Quick helper to get recommended model size."""
    return get_hardware().profile["recommended_model_size"]


if __name__ == "__main__":
    hw = HardwareProfile()
    print("[SYSTEM] " + "=" * 50)
    print("[SYSTEM] HARDWARE PROFILE")
    print("[SYSTEM] " + "=" * 50)
    print("[SYSTEM]", hw.summary())
    print()
    print("[SYSTEM] Best PyTorch device:", hw.get_device())
