#!/usr/bin/env python3
"""
Reset Enigma AI Engine for Public Sharing

Removes user-specific data and settings to prepare for public release.

What gets cleaned:
- Trained models (models/)
- Conversation history (data/conversations/)
- User training data, GUI settings, module config
- Voice profiles, avatar settings
- Outputs, memory database, logs, cache

What is preserved:
- All source code
- Example data files
- Documentation

Usage:
    python scripts/reset_for_sharing.py --preview      # See what will be cleaned
    python scripts/reset_for_sharing.py --yes          # Clean without confirmation
    python scripts/reset_for_sharing.py --skip models  # Keep models folder
    python scripts/reset_for_sharing.py --skip models --skip outputs  # Keep multiple
"""

from __future__ import annotations

import argparse
import shutil
import json
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Valid skip options (for typo detection)
VALID_SKIP_OPTIONS = frozenset({
    'models', 'outputs', 'memory', 'logs', 'conversations',
    'voice_profiles', 'avatar', '__pycache__', 'gui_settings',
    'user_training', 'url_blocklist', 'tool_routing', 
    'information_gui', 'module_config'
})

DEFAULT_GUI_SETTINGS: dict[str, str | int] = {
    "theme": "dark",
    "window_width": 1400,
    "window_height": 900
}


@dataclass
class CleanupItem:
    """Represents an item to be cleaned."""
    path: Path
    description: str
    name: str  # For skip matching
    is_directory: bool = True
    
    def exists(self) -> bool:
        return self.path.is_dir() if self.is_directory else self.path.is_file()
    
    def count(self) -> int:
        if not self.exists():
            return 0
        return sum(1 for _ in self.path.rglob('*')) if self.is_directory else 1


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    success: bool = True
    items_removed: int = 0
    errors: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Get the project root directory (cached)."""
    return Path(__file__).parent.parent.resolve()


def get_cleanup_items(root: Path) -> tuple[list[CleanupItem], list[CleanupItem]]:
    """Get directories and files to clean."""
    directories = [
        CleanupItem(root / 'models', 'Trained models', 'models'),
        CleanupItem(root / 'outputs', 'Generated outputs', 'outputs'),
        CleanupItem(root / 'memory', 'Memory database', 'memory'),
        CleanupItem(root / 'logs', 'Log files', 'logs'),
        CleanupItem(root / 'data' / 'conversations', 'Conversation history', 'conversations'),
        CleanupItem(root / 'data' / 'voice_profiles', 'Voice profiles', 'voice_profiles'),
        CleanupItem(root / 'data' / 'avatar', 'Avatar settings', 'avatar'),
        CleanupItem(root / '__pycache__', 'Python cache', '__pycache__'),
    ]
    
    files = [
        CleanupItem(root / 'data' / 'gui_settings.json', 'GUI settings', 'gui_settings', is_directory=False),
        CleanupItem(root / 'data' / 'user_training.txt', 'User training data', 'user_training', is_directory=False),
        CleanupItem(root / 'data' / 'url_blocklist_cache.json', 'URL blocklist cache', 'url_blocklist', is_directory=False),
        CleanupItem(root / 'data' / 'tool_routing.json', 'Tool routing config', 'tool_routing', is_directory=False),
        CleanupItem(root / 'information' / 'gui_settings.json', 'Information GUI settings', 'information_gui', is_directory=False),
        CleanupItem(root / 'enigma_engine' / 'modules' / 'module_config.json', 'Module configuration', 'module_config', is_directory=False),
    ]
    
    return directories, files


def validate_skip_items(skip_items: list[str]) -> list[str]:
    """
    Validate skip items and suggest corrections for typos.
    
    Returns list of error messages (empty if all valid).
    """
    errors = []
    for item in skip_items:
        item_lower = item.lower()
        if item_lower not in VALID_SKIP_OPTIONS:
            # Find similar options (simple fuzzy match)
            suggestions = [opt for opt in VALID_SKIP_OPTIONS 
                          if opt.startswith(item_lower[:3]) or item_lower[:3] in opt]
            
            msg = f"Unknown skip item: '{item}'"
            if suggestions:
                msg += f" - did you mean: {', '.join(sorted(suggestions))}?"
            else:
                msg += f"\n  Valid options: {', '.join(sorted(VALID_SKIP_OPTIONS))}"
            errors.append(msg)
    
    return errors


def preview_cleanup(root: Path, skip_items: set[str]) -> None:
    """Show what will be cleaned."""
    directories, files = get_cleanup_items(root)
    
    print("\n=== Items to be cleaned ===\n")
    
    print("DIRECTORIES (will be emptied):")
    for item in directories:
        if item.name in skip_items:
            print(f"  [SKIP] {item.path.relative_to(root)} - {item.description}")
        elif item.exists():
            print(f"  [EXISTS ({item.count()} items)] {item.path.relative_to(root)} - {item.description}")
        else:
            print(f"  [not found] {item.path.relative_to(root)} - {item.description}")
    
    print("\nFILES (will be deleted):")
    for item in files:
        if item.name in skip_items:
            print(f"  [SKIP] {item.path.relative_to(root)} - {item.description}")
        else:
            status = "EXISTS" if item.exists() else "not found"
            print(f"  [{status}] {item.path.relative_to(root)} - {item.description}")
    
    print()


def clean_directory(path: Path, result: CleanupResult) -> int:
    """Clean a directory but keep the folder and .gitkeep."""
    if not path.is_dir():
        return 0
    
    count = 0
    for item in path.iterdir():
        if item.name == '.gitkeep':
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            count += 1
        except (PermissionError, OSError) as e:
            result.errors.append(f"Could not remove {item.name}: {e}")
            result.success = False
    
    return count


def perform_cleanup(root: Path, skip_items: set[str], verbose: bool = True) -> CleanupResult:
    """Perform the actual cleanup."""
    result = CleanupResult()
    directories, files = get_cleanup_items(root)
    
    if verbose:
        print("\n=== Cleaning Enigma AI Engine for sharing ===\n")
    
    # Clean directories
    for item in directories:
        if item.name in skip_items:
            if verbose:
                print(f"Skipped {item.description}")
            continue
            
        if item.exists():
            count = clean_directory(item.path, result)
            result.items_removed += count
            if verbose:
                print(f"Cleaned {item.description}: {count} items removed")
            
            # Add .gitkeep to preserve directory
            gitkeep = item.path / '.gitkeep'
            if not gitkeep.exists():
                try:
                    gitkeep.touch()
                except OSError:
                    pass
        elif verbose:
            print(f"Skipped {item.description}: not found")
    
    # Delete files
    for item in files:
        if item.name in skip_items:
            if verbose:
                print(f"Skipped {item.description}")
            continue
            
        if item.exists():
            try:
                item.path.unlink()
                result.items_removed += 1
                if verbose:
                    print(f"Deleted {item.description}")
            except (PermissionError, OSError) as e:
                result.errors.append(f"Could not delete {item.description}: {e}")
                result.success = False
        elif verbose:
            print(f"Skipped {item.description}: not found")
    
    # Reset gui_settings.json to defaults (unless skipped)
    if 'gui_settings' not in skip_items:
        gui_path = root / 'data' / 'gui_settings.json'
        try:
            gui_path.parent.mkdir(parents=True, exist_ok=True)
            gui_path.write_text(json.dumps(DEFAULT_GUI_SETTINGS, indent=2))
            if verbose:
                print("Reset GUI settings to defaults")
        except OSError as e:
            result.errors.append(f"Could not reset GUI settings: {e}")
    
    if verbose:
        print("\n=== Cleanup complete! ===")
        if result.errors:
            print(f"\nWarnings ({len(result.errors)}):")
            for error in result.errors:
                print(f"  - {error}")
        print(f"\nTotal items removed: {result.items_removed}")
        print("The project is now ready for public sharing.\n")
    
    return result


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Reset Enigma AI Engine for public sharing by removing user-specific data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Valid --skip options:
  {', '.join(sorted(VALID_SKIP_OPTIONS))}

Examples:
  %(prog)s --preview              Show what will be cleaned
  %(prog)s --yes                  Clean without confirmation  
  %(prog)s --skip models          Keep trained models
  %(prog)s -s models -s outputs   Keep multiple items
        """
    )
    parser.add_argument('--preview', '-p', action='store_true',
                        help='Preview what will be cleaned')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    parser.add_argument('--skip', '-s', action='append', default=[], metavar='ITEM',
                        help='Skip cleaning specific items (can repeat)')
    
    args = parser.parse_args()
    
    # Validate skip items for typos
    if args.skip:
        errors = validate_skip_items(args.skip)
        if errors:
            print("Error:", file=sys.stderr)
            for err in errors:
                print(f"  {err}", file=sys.stderr)
            return 1
    
    skip_items = {s.lower() for s in args.skip}
    root = get_project_root()
    
    print("=" * 50)
    print("  Enigma AI Engine - Reset for Public Sharing")
    print("=" * 50)
    
    if args.preview:
        preview_cleanup(root, skip_items)
        return 0
    
    preview_cleanup(root, skip_items)
    
    if not args.yes:
        print("WARNING: This will permanently delete user data!")
        try:
            response = input("Continue? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 1
        if response not in ('y', 'yes'):
            print("Cancelled.")
            return 0
    
    result = perform_cleanup(root, skip_items, verbose=not args.quiet)
    return 0 if result.success else 1


if __name__ == '__main__':
    sys.exit(main())
