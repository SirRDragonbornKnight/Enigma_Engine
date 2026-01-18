#!/usr/bin/env python3
"""
Create Windows Desktop Shortcut for ForgeAI

Creates a shortcut on the desktop with:
- ForgeAI icon
- Direct launch to GUI
- Proper working directory

Usage:
    python scripts/create_shortcut.py
"""

import os
import sys
from pathlib import Path


def create_windows_shortcut():
    """Create a Windows desktop shortcut for ForgeAI."""
    try:
        import winshell
        from win32com.client import Dispatch
    except ImportError:
        print("Installing required packages...")
        os.system("pip install pywin32 winshell")
        import winshell
        from win32com.client import Dispatch
    
    # Paths
    project_root = Path(__file__).parent.parent.resolve()
    desktop = Path(winshell.desktop())
    shortcut_path = desktop / "ForgeAI.lnk"
    
    # Icon path
    icon_path = project_root / "data" / "icons" / "forge.ico"
    if not icon_path.exists():
        icon_path = project_root / "forge_ai" / "gui" / "icons" / "forge.ico"
    
    # Python executable
    python_exe = sys.executable
    
    # Create shortcut
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(str(shortcut_path))
    
    # Set properties
    shortcut.Targetpath = python_exe
    shortcut.Arguments = f'"{project_root / "run.py"}" --gui'
    shortcut.WorkingDirectory = str(project_root)
    shortcut.Description = "ForgeAI - Your AI Companion"
    
    if icon_path.exists():
        shortcut.IconLocation = str(icon_path)
    
    shortcut.save()
    
    print(f"Shortcut created: {shortcut_path}")
    print(f"Icon: {icon_path}")
    return shortcut_path


def create_start_menu_entry():
    """Create a Start Menu entry for ForgeAI."""
    try:
        import winshell
        from win32com.client import Dispatch
    except ImportError:
        return None
    
    project_root = Path(__file__).parent.parent.resolve()
    
    # Start Menu Programs folder
    start_menu = Path(winshell.programs())
    forge_folder = start_menu / "ForgeAI"
    forge_folder.mkdir(exist_ok=True)
    
    shortcut_path = forge_folder / "ForgeAI.lnk"
    
    # Icon path
    icon_path = project_root / "data" / "icons" / "forge.ico"
    if not icon_path.exists():
        icon_path = project_root / "forge_ai" / "gui" / "icons" / "forge.ico"
    
    python_exe = sys.executable
    
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(str(shortcut_path))
    shortcut.Targetpath = python_exe
    shortcut.Arguments = f'"{project_root / "run.py"}" --gui'
    shortcut.WorkingDirectory = str(project_root)
    shortcut.Description = "ForgeAI - Your AI Companion"
    
    if icon_path.exists():
        shortcut.IconLocation = str(icon_path)
    
    shortcut.save()
    
    print(f"Start Menu entry created: {shortcut_path}")
    return shortcut_path


def main():
    if sys.platform != 'win32':
        print("This script is for Windows only.")
        print("On Linux/macOS, create a .desktop file or alias manually.")
        return
    
    print("=" * 50)
    print("  ForgeAI Shortcut Creator")
    print("=" * 50)
    print()
    
    # Create desktop shortcut
    print("Creating desktop shortcut...")
    desktop_shortcut = create_windows_shortcut()
    
    # Create Start Menu entry
    print("\nCreating Start Menu entry...")
    try:
        start_menu_shortcut = create_start_menu_entry()
    except Exception as e:
        print(f"  Could not create Start Menu entry: {e}")
    
    print("\n" + "=" * 50)
    print("  Done! You can now launch ForgeAI from:")
    print("  - Desktop shortcut")
    print("  - Start Menu > ForgeAI")
    print("=" * 50)


if __name__ == "__main__":
    main()
