#!/usr/bin/env python3
"""
Reset ForgeAI for Public Sharing

This script removes user-specific data and settings to prepare
the project for public release/sharing.

What gets cleaned:
- Trained models (models/ folder contents)
- Conversation history (data/conversations/)
- User training data (data/user_training.txt)
- GUI settings (data/gui_settings.json)
- Module configuration
- Avatar orientation settings
- Voice profiles (data/voice_profiles/)
- Output files (outputs/)
- Memory database (memory/)
- Logs (logs/)
- URL blocklist cache

What is preserved:
- All source code
- Example data files
- Documentation
- Default configuration
"""

import argparse
import shutil
import json
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def get_items_to_clean():
    """Get list of items to clean with descriptions."""
    root = get_project_root()
    
    return {
        'directories': [
            (root / 'models', 'Trained models'),
            (root / 'outputs', 'Generated outputs'),
            (root / 'memory', 'Memory database'),
            (root / 'logs', 'Log files'),
            (root / 'data' / 'conversations', 'Conversation history'),
            (root / 'data' / 'voice_profiles', 'Voice profiles'),
            (root / 'data' / 'avatar', 'Avatar settings'),
        ],
        'files': [
            (root / 'data' / 'gui_settings.json', 'GUI settings'),
            (root / 'data' / 'user_training.txt', 'User training data'),
            (root / 'data' / 'url_blocklist_cache.json', 'URL blocklist cache'),
            (root / 'data' / 'tool_routing.json', 'Tool routing config'),
            (root / 'information' / 'gui_settings.json', 'Information GUI settings'),
            (root / 'forge_ai' / 'modules' / 'module_config.json', 'Module configuration'),
        ],
    }


def preview_cleanup():
    """Show what will be cleaned without actually deleting."""
    items = get_items_to_clean()
    
    print("\n=== Items to be cleaned ===\n")
    
    print("DIRECTORIES (will be emptied, not deleted):")
    for path, desc in items['directories']:
        exists = path.exists()
        status = "EXISTS" if exists else "not found"
        if exists and path.is_dir():
            count = len(list(path.glob('**/*')))
            status = f"EXISTS ({count} items)"
        print(f"  [{status}] {path.relative_to(get_project_root())} - {desc}")
    
    print("\nFILES (will be deleted):")
    for path, desc in items['files']:
        exists = path.exists()
        status = "EXISTS" if exists else "not found"
        print(f"  [{status}] {path.relative_to(get_project_root())} - {desc}")
    
    print()


def clean_directory(path: Path, keep_gitkeep: bool = True):
    """Clean a directory but keep the folder itself."""
    if not path.exists():
        return 0
    
    count = 0
    for item in path.iterdir():
        if keep_gitkeep and item.name == '.gitkeep':
            continue
        
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
        count += 1
    
    return count


def perform_cleanup(verbose: bool = True):
    """Perform the actual cleanup."""
    items = get_items_to_clean()
    
    print("\n=== Cleaning ForgeAI for sharing ===\n")
    
    # Clean directories
    for path, desc in items['directories']:
        if path.exists() and path.is_dir():
            count = clean_directory(path)
            if verbose:
                print(f"Cleaned {desc}: {count} items removed")
            
            # Add .gitkeep to keep directory structure
            gitkeep = path / '.gitkeep'
            if not gitkeep.exists():
                gitkeep.touch()
        elif verbose:
            print(f"Skipped {desc}: directory not found")
    
    # Delete files
    for path, desc in items['files']:
        if path.exists():
            path.unlink()
            if verbose:
                print(f"Deleted {desc}")
        elif verbose:
            print(f"Skipped {desc}: file not found")
    
    # Reset certain files to defaults
    root = get_project_root()
    
    # Reset gui_settings.json to minimal defaults
    default_gui_settings = {
        "theme": "dark",
        "window_width": 1400,
        "window_height": 900
    }
    gui_settings_path = root / 'data' / 'gui_settings.json'
    with open(gui_settings_path, 'w') as f:
        json.dump(default_gui_settings, f, indent=2)
    if verbose:
        print("Reset GUI settings to defaults")
    
    print("\n=== Cleanup complete! ===")
    print("\nThe project is now ready for public sharing.")
    print("User-specific data has been removed.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Reset ForgeAI for public sharing by removing user-specific data.'
    )
    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Preview what will be cleaned without actually deleting'
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("  ForgeAI - Reset for Public Sharing")
    print("=" * 50)
    
    if args.preview:
        preview_cleanup()
        return
    
    preview_cleanup()
    
    if not args.yes:
        print("WARNING: This will permanently delete user data!")
        response = input("Continue? [y/N]: ").strip().lower()
        if response not in ('y', 'yes'):
            print("Cancelled.")
            return
    
    perform_cleanup(verbose=not args.quiet)


if __name__ == '__main__':
    main()
