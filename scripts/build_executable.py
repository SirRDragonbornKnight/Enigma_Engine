#!/usr/bin/env python3
"""
Build ForgeAI as a Standalone Executable

Creates a single executable (or folder) that can run without Python installed.

Requirements:
    pip install pyinstaller

Usage:
    python scripts/build_executable.py              # Build one-folder distribution
    python scripts/build_executable.py --onefile    # Build single .exe (slower startup)
    python scripts/build_executable.py --portable   # Build portable version with models

Output:
    dist/ForgeAI/ForgeAI.exe   (one-folder mode)
    dist/ForgeAI.exe           (one-file mode)
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        return True
    except ImportError:
        print("PyInstaller not found. Installing...")
        os.system(f"{sys.executable} -m pip install pyinstaller")
        return True


def build_executable(onefile=False, portable=False):
    """Build the ForgeAI executable."""
    project_root = Path(__file__).parent.parent.resolve()
    
    # Paths
    main_script = project_root / "run.py"
    icon_path = project_root / "data" / "icons" / "forge.ico"
    if not icon_path.exists():
        icon_path = project_root / "forge_ai" / "gui" / "icons" / "forge.ico"
    
    # Data files to include
    data_files = [
        # Icons
        (str(project_root / "data" / "icons"), "data/icons"),
        (str(project_root / "forge_ai" / "gui" / "icons"), "forge_ai/gui/icons"),
        # Training data
        (str(project_root / "data" / "training.txt"), "data"),
        (str(project_root / "data" / "instructions.txt"), "data"),
        # Vocab model
        (str(project_root / "forge_ai" / "vocab_model"), "forge_ai/vocab_model"),
        # Config
        (str(project_root / "forge_ai" / "config"), "forge_ai/config"),
        # Builtin fallbacks
        (str(project_root / "forge_ai" / "builtin"), "forge_ai/builtin"),
    ]
    
    # Build data args
    add_data_args = []
    for src, dst in data_files:
        if Path(src).exists():
            add_data_args.append(f'--add-data={src};{dst}')
    
    # Hidden imports (modules that PyInstaller might miss)
    hidden_imports = [
        "torch",
        "numpy",
        "PyQt5",
        "PyQt5.QtWidgets",
        "PyQt5.QtGui",
        "PyQt5.QtCore",
        "PIL",
        "flask",
        "transformers",
        "safetensors",
        "pyttsx3",
        "speech_recognition",
        "forge_ai",
        "forge_ai.core",
        "forge_ai.gui",
        "forge_ai.modules",
        "forge_ai.tools",
        "forge_ai.voice",
        "forge_ai.companion",
    ]
    
    hidden_import_args = [f'--hidden-import={mod}' for mod in hidden_imports]
    
    # Build command
    cmd_parts = [
        sys.executable, "-m", "PyInstaller",
        "--name=ForgeAI",
        "--windowed",  # No console window
        f"--icon={icon_path}" if icon_path.exists() else "",
        "--noconfirm",  # Overwrite without asking
        "--clean",  # Clean cache
    ]
    
    if onefile:
        cmd_parts.append("--onefile")
    else:
        cmd_parts.append("--onedir")
    
    cmd_parts.extend(add_data_args)
    cmd_parts.extend(hidden_import_args)
    cmd_parts.append(str(main_script))
    
    # Filter empty strings
    cmd_parts = [p for p in cmd_parts if p]
    
    # Run PyInstaller
    print("Building ForgeAI executable...")
    print(f"Command: {' '.join(cmd_parts[:5])}...")
    
    os.chdir(project_root)
    result = os.system(" ".join(f'"{p}"' if " " in p else p for p in cmd_parts))
    
    if result != 0:
        print("\nBuild failed!")
        return False
    
    # Copy additional files if portable
    if portable:
        dist_dir = project_root / "dist" / "ForgeAI"
        if dist_dir.exists():
            print("\nCreating portable package...")
            
            # Copy models folder if exists
            models_src = project_root / "models"
            if models_src.exists():
                print("  Copying models...")
                shutil.copytree(models_src, dist_dir / "models", dirs_exist_ok=True)
            
            # Copy data folder
            data_src = project_root / "data"
            if data_src.exists():
                print("  Copying data...")
                shutil.copytree(data_src, dist_dir / "data", dirs_exist_ok=True)
    
    print("\n" + "=" * 50)
    print("  Build Complete!")
    print("=" * 50)
    
    if onefile:
        exe_path = project_root / "dist" / "ForgeAI.exe"
    else:
        exe_path = project_root / "dist" / "ForgeAI" / "ForgeAI.exe"
    
    print(f"\nExecutable: {exe_path}")
    print(f"Size: {exe_path.stat().st_size / 1024 / 1024:.1f} MB" if exe_path.exists() else "")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Build ForgeAI executable")
    parser.add_argument("--onefile", action="store_true", 
                        help="Build single executable (slower startup)")
    parser.add_argument("--portable", action="store_true",
                        help="Include models and data for portable distribution")
    args = parser.parse_args()
    
    if not check_pyinstaller():
        print("Failed to install PyInstaller")
        return 1
    
    success = build_executable(onefile=args.onefile, portable=args.portable)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
