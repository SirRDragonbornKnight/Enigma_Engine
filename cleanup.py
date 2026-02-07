#!/usr/bin/env python3
"""
Enigma AI Engine Cleanup Script
----------------------
Removes temporary files, caches, and optionally large downloadable models
to reduce disk space usage.

Usage:
    python cleanup.py              # Standard cleanup
    python cleanup.py --deep       # Include downloadable models
    python cleanup.py --dry-run    # Show what would be deleted
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    END = '\033[0m'

def color(text, c):
    if sys.stdout.isatty() and os.name != 'nt':
        return f"{c}{text}{Colors.END}"
    return text

def get_size(path):
    """Get size of file or directory in bytes."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except (OSError, FileNotFoundError):
                pass
    return total

def format_size(size_bytes):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def cleanup_patterns():
    """Return patterns to clean."""
    return {
        'pycache': {
            'patterns': ['**/__pycache__'],
            'description': 'Python bytecode cache',
        },
        'pyc': {
            'patterns': ['**/*.pyc', '**/*.pyo'],
            'description': 'Compiled Python files',
        },
        'pytest': {
            'patterns': ['**/.pytest_cache', '.coverage', 'htmlcov'],
            'description': 'Pytest cache and coverage',
        },
        'mypy': {
            'patterns': ['**/.mypy_cache'],
            'description': 'MyPy type checking cache',
        },
        'ruff': {
            'patterns': ['**/.ruff_cache'],
            'description': 'Ruff linter cache',
        },
        'logs': {
            'patterns': ['logs/*.log', '**/*.log'],
            'description': 'Log files',
        },
        'temp': {
            'patterns': ['tmp/', 'temp/', '**/*.tmp', '**/*.bak'],
            'description': 'Temporary files',
        },
        'checkpoints': {
            'patterns': ['models/*/checkpoints/epoch_*.pth', 'models/*/checkpoints/latest.pth'],
            'description': 'Training checkpoints (keeps best.pth)',
        },
    }

def deep_cleanup_patterns():
    """Additional patterns for deep cleanup."""
    return {
        'hf_cache': {
            'patterns': ['models/hf_cache'],
            'description': 'HuggingFace model cache (can redownload)',
        },
        'downloaded_models': {
            'patterns': ['models/dialogpt_fresh', 'models/dialogpt_small'],
            'description': 'Downloaded HuggingFace models (can redownload)',
        },
    }

def find_files(root, pattern):
    """Find files matching a glob pattern."""
    root = Path(root)
    return list(root.glob(pattern))

def cleanup(root_dir, deep=False, dry_run=False):
    """Perform cleanup."""
    root = Path(root_dir)
    patterns = cleanup_patterns()
    
    if deep:
        patterns.update(deep_cleanup_patterns())
    
    total_saved = 0
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}Cleaning up {root}\n")
    
    for name, info in patterns.items():
        category_size = 0
        files_found = []
        
        for pattern in info['patterns']:
            matches = find_files(root, pattern)
            for match in matches:
                if match.exists():
                    size = get_size(match)
                    category_size += size
                    files_found.append((match, size))
        
        if files_found:
            print(f"{info['description']}:")
            for path, size in files_found:
                rel_path = path.relative_to(root)
                print(f"  {rel_path} ({format_size(size)})")
                
                if not dry_run:
                    try:
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                    except Exception as e:
                        print(color(f"    Error: {e}", Colors.RED))
            
            total_saved += category_size
            status = "would save" if dry_run else "saved"
            print(color(f"  -> {status} {format_size(category_size)}", Colors.GREEN))
            print()
    
    # Summary
    print("=" * 50)
    status = "Would save" if dry_run else "Saved"
    print(color(f"{status}: {format_size(total_saved)}", Colors.CYAN))
    
    if not dry_run:
        # Get current size
        current_size = get_size(root)
        print(f"Current project size: {format_size(current_size)}")
    
    return total_saved

def main():
    parser = argparse.ArgumentParser(description='Enigma AI Engine Cleanup Script')
    parser.add_argument('--deep', action='store_true', 
                        help='Include downloadable models (can be re-downloaded)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without deleting')
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent.absolute()
    
    # Confirm deep cleanup
    if args.deep and not args.dry_run:
        print(color("\nWARNING: Deep cleanup will remove downloaded HuggingFace models.", Colors.YELLOW))
        print("These can be re-downloaded but will require internet and time.")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
    
    cleanup(script_dir, deep=args.deep, dry_run=args.dry_run)

if __name__ == '__main__':
    main()
