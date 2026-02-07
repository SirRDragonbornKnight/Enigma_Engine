#!/usr/bin/env python3
"""
Enigma AI Engine Installer
-----------------
Cross-platform installer that detects your system and installs appropriate dependencies.

Usage:
    python install.py              # Interactive mode
    python install.py --minimal    # Core only (smallest footprint)
    python install.py --standard   # Core + GUI + Voice
    python install.py --full       # Everything
    python install.py --check      # Check what's installed
"""

import os
import sys
import platform
import subprocess
import argparse
from pathlib import Path

# Color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def color(text, c):
    """Add color if terminal supports it."""
    if sys.stdout.isatty() and os.name != 'nt':
        return f"{c}{text}{Colors.END}"
    return text

def print_header(text):
    print("\n" + "=" * 70)
    print(color(f"  {text}", Colors.BOLD))
    print("=" * 70)

def print_ok(text):
    print(color(f"[OK] {text}", Colors.GREEN))

def print_warn(text):
    print(color(f"[WARN] {text}", Colors.YELLOW))

def print_error(text):
    print(color(f"[ERROR] {text}", Colors.RED))

def print_info(text):
    print(color(f"[INFO] {text}", Colors.CYAN))

def run_cmd(cmd, check=True, capture=True):
    """Run a shell command."""
    try:
        result = subprocess.run(
            cmd, shell=True, check=check,
            capture_output=capture, text=True
        )
        return result.returncode == 0, result.stdout if capture else ""
    except subprocess.CalledProcessError as e:
        return False, str(e)

def detect_system():
    """Detect system information."""
    info = {
        'os': platform.system(),
        'arch': platform.machine(),
        'python': platform.python_version(),
        'is_pi': False,
        'has_gpu': False,
        'gpu_type': None,
        'ram_mb': 0,
    }
    
    # Check if Raspberry Pi
    if info['os'] == 'Linux':
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi' in model:
                    info['is_pi'] = True
                    info['pi_model'] = model.strip().replace('\x00', '')
        except:
            pass
    
    # Check RAM
    try:
        import psutil
        info['ram_mb'] = psutil.virtual_memory().total // (1024 * 1024)
    except:
        pass
    
    # Check GPU - works on Linux, Windows, and macOS
    if info['os'] == 'Linux':
        success, output = run_cmd('nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null')
        if success and output.strip():
            info['has_gpu'] = True
            info['gpu_type'] = 'nvidia'
            info['gpu_name'] = output.strip().split('\n')[0]
    elif info['os'] == 'Windows':
        # Windows - nvidia-smi is in PATH if CUDA drivers are installed
        success, output = run_cmd('nvidia-smi --query-gpu=name --format=csv,noheader 2>nul')
        if success and output.strip():
            info['has_gpu'] = True
            info['gpu_type'] = 'nvidia'
            info['gpu_name'] = output.strip().split('\n')[0]
    elif info['os'] == 'Darwin':
        # macOS - check for Apple Silicon
        if info['arch'] == 'arm64':
            info['has_gpu'] = True
            info['gpu_type'] = 'mps'
            info['gpu_name'] = 'Apple Silicon'
    
    return info

def get_installed_packages():
    """Get list of installed pip packages."""
    success, output = run_cmd(f'{sys.executable} -m pip list --format=freeze')
    packages = {}
    if success:
        for line in output.strip().split('\n'):
            if '==' in line:
                name, ver = line.split('==', 1)
                packages[name.lower()] = ver
    return packages

def install_package(package, extra_args=""):
    """Install a pip package."""
    cmd = f'{sys.executable} -m pip install {extra_args} "{package}"'
    success, _ = run_cmd(cmd, check=False, capture=False)
    return success

# Package groups
PACKAGES = {
    'core': [
        'python-dotenv>=0.19.0',
        'numpy>=1.21.0',
        'flask>=2.0.0',
        'flask-cors>=3.0.10',
        'psutil>=5.9.0',
    ],
    'model': [
        'safetensors>=0.4.0',
        'tokenizers>=0.13.0',
        'transformers>=4.20.0',
        'huggingface-hub>=0.20.0',
    ],
    'gui': [
        'PyQt5>=5.15.0',
        'pillow>=9.0.0',
    ],
    'voice': [
        'pyttsx3>=2.90',
        'SpeechRecognition>=3.8.0',
        'sounddevice>=0.4.0',
    ],
    'voice_advanced': [
        'vosk>=0.3.0',
        'openai-whisper>=20230314',
    ],
    'vision': [
        'opencv-python>=4.5.0',
        'mss>=6.1.0',
    ],
    'web': [
        'fastapi>=0.104.0',
        'uvicorn[standard]>=0.24.0',
        'websockets>=12.0',
        'python-multipart>=0.0.6',
        'flask-socketio>=5.0.0',
    ],
    'network': [
        'qrcode[pil]>=7.4.0',
        'zeroconf>=0.131.0',
    ],
    'database': [
        'sqlalchemy>=1.4.0',
    ],
    '3d': [
        'trimesh>=3.20.0',
    ],
    'gpu_nvidia': [
        'pynvml>=11.0.0',
        'GPUtil>=1.4.0',
    ],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=3.0.0',
        'flake8>=5.0.0',
    ],
}

# Installation profiles
PROFILES = {
    'minimal': ['core'],
    'standard': ['core', 'model', 'gui', 'voice', 'database'],
    'full': ['core', 'model', 'gui', 'voice', 'voice_advanced', 'vision', 'web', 'network', 'database', '3d'],
    'dev': ['core', 'model', 'gui', 'voice', 'database', 'dev'],
}

def check_installation(installed):
    """Check what's installed vs needed."""
    print_header("Installation Check")
    
    all_packages = set()
    for group, pkgs in PACKAGES.items():
        for p in pkgs:
            # Extract package name
            name = p.split('>=')[0].split('[')[0].lower()
            all_packages.add((name, group, p))
    
    installed_count = 0
    missing = []
    
    for name, group, spec in sorted(all_packages):
        name_lower = name.lower().replace('-', '_')
        name_dash = name.lower().replace('_', '-')
        
        if name_lower in installed or name_dash in installed:
            installed_count += 1
        else:
            missing.append((name, group, spec))
    
    print(f"\nInstalled: {installed_count}/{len(all_packages)} packages")
    
    if missing:
        print(f"\nMissing packages by group:")
        groups = {}
        for name, group, spec in missing:
            if group not in groups:
                groups[group] = []
            groups[group].append(spec)
        
        for group, pkgs in sorted(groups.items()):
            print(f"\n  [{group}]")
            for p in pkgs:
                print(f"    - {p}")
    else:
        print_ok("All packages installed!")
    
    return missing

def install_pytorch(system_info):
    """Install PyTorch with appropriate backend."""
    print_header("Installing PyTorch")
    
    if system_info['is_pi']:
        print_info("Raspberry Pi detected - installing CPU-only PyTorch")
        cmd = f'{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
    elif system_info['gpu_type'] == 'nvidia':
        gpu_name = system_info.get('gpu_name', 'unknown').lower()
        print_info(f"NVIDIA GPU detected ({system_info.get('gpu_name', 'unknown')}) - installing CUDA PyTorch")
        
        # RTX 50 series (Blackwell) needs CUDA 12.8+ nightly
        if any(x in gpu_name for x in ['5090', '5080', '5070', '5060', '50 series']):
            print_info("RTX 50 series detected - using CUDA 12.8 nightly for Blackwell support")
            cmd = f'{sys.executable} -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128'
        # RTX 40/30/20 series work well with CUDA 12.4
        else:
            cmd = f'{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124'
    elif system_info['gpu_type'] == 'mps':
        print_info("Apple Silicon detected - installing MPS PyTorch")
        cmd = f'{sys.executable} -m pip install torch torchvision torchaudio'
    else:
        print_info("No GPU detected - installing CPU PyTorch")
        cmd = f'{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
    
    print(f"Running: {cmd}")
    success, _ = run_cmd(cmd, check=False, capture=False)
    
    if success:
        print_ok("PyTorch installed successfully")
    else:
        print_error("PyTorch installation failed - try installing manually")
    
    return success

def install_system_deps(system_info):
    """Install system-level dependencies."""
    print_header("System Dependencies")
    
    if system_info['os'] == 'Linux':
        if system_info['is_pi']:
            print_info("Raspberry Pi - checking apt packages...")
            packages = [
                'python3-pyqt5',
                'portaudio19-dev',
                'python3-pyaudio',
                'libespeak-ng1',
                'espeak-ng',
            ]
            
            for pkg in packages:
                success, _ = run_cmd(f'dpkg -l {pkg} 2>/dev/null | grep -q "^ii"')
                if success:
                    print_ok(f"{pkg} already installed")
                else:
                    print_warn(f"{pkg} not installed - run: sudo apt install {pkg}")
        else:
            print_info("Linux detected - you may need:")
            print("  sudo apt install python3-pyqt5 portaudio19-dev libespeak-ng1")
    
    elif system_info['os'] == 'Darwin':
        print_info("macOS detected - you may need:")
        print("  brew install portaudio espeak-ng")
    
    elif system_info['os'] == 'Windows':
        print_info("Windows detected - most deps install via pip")
        print("  For voice: Install espeak-ng from https://github.com/espeak-ng/espeak-ng/releases")

def install_profile(profile_name, system_info, installed):
    """Install packages for a profile."""
    if profile_name not in PROFILES:
        print_error(f"Unknown profile: {profile_name}")
        return False
    
    groups = PROFILES[profile_name]
    print_header(f"Installing Profile: {profile_name}")
    print(f"Groups: {', '.join(groups)}")
    
    # Collect all packages to install
    to_install = []
    for group in groups:
        if group in PACKAGES:
            for pkg in PACKAGES[group]:
                name = pkg.split('>=')[0].split('[')[0].lower()
                name_alt = name.replace('-', '_')
                if name not in installed and name_alt not in installed:
                    to_install.append(pkg)
    
    if not to_install:
        print_ok("All packages already installed!")
        return True
    
    print(f"\nPackages to install: {len(to_install)}")
    for pkg in to_install:
        print(f"  - {pkg}")
    
    print("\nInstalling...")
    failed = []
    for pkg in to_install:
        print(f"  Installing {pkg}...", end=" ", flush=True)
        if install_package(pkg):
            print(color("OK", Colors.GREEN))
        else:
            print(color("FAILED", Colors.RED))
            failed.append(pkg)
    
    if failed:
        print_warn(f"\n{len(failed)} packages failed to install:")
        for pkg in failed:
            print(f"  - {pkg}")
        return False
    
    print_ok("\nAll packages installed successfully!")
    return True

def interactive_install(system_info, installed):
    """Interactive installation wizard."""
    print_header("Enigma AI Engine Interactive Installer")
    
    print(f"""
System Information:
  OS: {system_info['os']} ({system_info['arch']})
  Python: {system_info['python']}
  RAM: {system_info['ram_mb']} MB
  GPU: {system_info.get('gpu_name', 'None detected')}
  Raspberry Pi: {'Yes - ' + system_info.get('pi_model', '') if system_info['is_pi'] else 'No'}

Installation Profiles:
  1. minimal  - Core only (~50MB) - CLI inference, no GUI
  2. standard - Core + GUI + Voice (~200MB) - Desktop use
  3. full     - Everything (~500MB+) - All features
  4. dev      - Standard + Testing tools
  5. check    - Just check what's installed
  6. pytorch  - Install/update PyTorch only
  7. custom   - Choose specific groups
  0. exit     - Exit installer
""")
    
    while True:
        try:
            choice = input("\nSelect option [1-7, 0 to exit]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            return
        
        if choice == '0':
            print("Exiting.")
            return
        elif choice == '1':
            install_pytorch(system_info)
            install_profile('minimal', system_info, installed)
        elif choice == '2':
            install_pytorch(system_info)
            install_profile('standard', system_info, installed)
        elif choice == '3':
            install_pytorch(system_info)
            install_profile('full', system_info, installed)
        elif choice == '4':
            install_pytorch(system_info)
            install_profile('dev', system_info, installed)
        elif choice == '5':
            check_installation(installed)
        elif choice == '6':
            install_pytorch(system_info)
        elif choice == '7':
            print("\nAvailable groups:")
            for i, group in enumerate(PACKAGES.keys(), 1):
                pkgs = PACKAGES[group]
                print(f"  {i}. {group} ({len(pkgs)} packages)")
            
            try:
                selections = input("\nEnter group numbers (comma-separated): ").strip()
                group_nums = [int(x.strip()) for x in selections.split(',')]
                group_names = list(PACKAGES.keys())
                
                for num in group_nums:
                    if 1 <= num <= len(group_names):
                        group = group_names[num - 1]
                        print(f"\nInstalling {group}...")
                        for pkg in PACKAGES[group]:
                            print(f"  {pkg}...", end=" ", flush=True)
                            if install_package(pkg):
                                print(color("OK", Colors.GREEN))
                            else:
                                print(color("FAILED", Colors.RED))
            except ValueError:
                print_error("Invalid input")
        else:
            print("Invalid choice")
        
        # Refresh installed packages
        installed.clear()
        installed.update(get_installed_packages())

def main():
    parser = argparse.ArgumentParser(description='Enigma AI Engine Installer')
    parser.add_argument('--minimal', action='store_true', help='Install minimal (core only)')
    parser.add_argument('--standard', action='store_true', help='Install standard (core + GUI + voice)')
    parser.add_argument('--full', action='store_true', help='Install everything')
    parser.add_argument('--dev', action='store_true', help='Install dev profile')
    parser.add_argument('--check', action='store_true', help='Check installation status')
    parser.add_argument('--pytorch', action='store_true', help='Install PyTorch only')
    parser.add_argument('--system-deps', action='store_true', help='Show system dependency info')
    args = parser.parse_args()
    
    print(color("""
╔═══════════════════════════════════════════════════════════════════╗
║                     Enigma AI Engine Installer v1.0                        ║
║              Cross-platform AI Framework Setup                    ║
╚═══════════════════════════════════════════════════════════════════╝
""", Colors.CYAN))
    
    # Detect system
    print("Detecting system...")
    system_info = detect_system()
    installed = get_installed_packages()
    
    print(f"  Platform: {system_info['os']} {system_info['arch']}")
    print(f"  Python: {system_info['python']}")
    if system_info['is_pi']:
        print(f"  Device: {system_info.get('pi_model', 'Raspberry Pi')}")
    if system_info['has_gpu']:
        print(f"  GPU: {system_info.get('gpu_name', system_info['gpu_type'])}")
    print(f"  Installed packages: {len(installed)}")
    
    # Handle command line args
    if args.check:
        check_installation(installed)
    elif args.system_deps:
        install_system_deps(system_info)
    elif args.pytorch:
        install_pytorch(system_info)
    elif args.minimal:
        install_pytorch(system_info)
        install_profile('minimal', system_info, installed)
    elif args.standard:
        install_pytorch(system_info)
        install_profile('standard', system_info, installed)
    elif args.full:
        install_pytorch(system_info)
        install_profile('full', system_info, installed)
    elif args.dev:
        install_pytorch(system_info)
        install_profile('dev', system_info, installed)
    else:
        # Interactive mode
        interactive_install(system_info, installed)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
