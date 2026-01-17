#!/usr/bin/env python3
"""
Training Data Validator - Check for common issues before training.

Usage:
    python scripts/validate_training_data.py                    # Validate all
    python scripts/validate_training_data.py data/training.txt  # Validate one file
"""

import sys
import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ValidationResult:
    """Validation results for a file."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_file(filepath: Path) -> ValidationResult:
    """Validate a training data file."""
    result = ValidationResult()
    
    if not filepath.exists():
        result.errors.append(f"File not found: {filepath}")
        return result
    
    try:
        content = filepath.read_text(encoding='utf-8')
        lines = content.split('\n')
    except Exception as e:
        result.errors.append(f"Could not read file: {e}")
        return result
    
    # Basic stats
    q_count = sum(1 for l in lines if l.strip().startswith('Q:'))
    a_count = sum(1 for l in lines if l.strip().startswith('A:'))
    result.stats = {'lines': len(lines), 'questions': q_count, 'answers': a_count}
    
    # Check Q/A balance
    if q_count != a_count:
        result.warnings.append(f"Mismatched Q/A: {q_count} questions, {a_count} answers")
    
    # Dataset size check
    if q_count < 100:
        result.warnings.append(f"Small dataset ({q_count} pairs). Recommend 1000+ for good results.")
    
    # Tool call validation
    tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
    tool_results = re.findall(r'<tool_result>(.*?)</tool_result>', content, re.DOTALL)
    
    result.stats['tool_calls'] = len(tool_calls)
    
    if len(tool_calls) != len(tool_results):
        result.warnings.append(f"Mismatched tool tags: {len(tool_calls)} calls, {len(tool_results)} results")
    
    # Validate tool JSON
    tools_used = Counter()
    for i, tc in enumerate(tool_calls):
        try:
            data = json.loads(tc.strip())
            if 'tool' in data:
                tools_used[data['tool']] += 1
            else:
                result.warnings.append(f"Tool call #{i+1} missing 'tool' field")
        except json.JSONDecodeError as e:
            result.errors.append(f"Tool call #{i+1} invalid JSON: {e}")
    
    result.stats['tools'] = dict(tools_used)
    
    # Check for wrong formats
    wrong_formats = {'User:': 'Q:', 'Human:': 'Q:', 'Assistant:': 'A:'}
    for line_num, line in enumerate(lines, 1):
        for wrong, correct in wrong_formats.items():
            if line.strip().startswith(wrong):
                result.warnings.append(f"Line {line_num}: '{wrong}' should be '{correct}'")
    
    # Check unbalanced tags
    open_tags = content.count('<tool_call>') + content.count('<tool_result>')
    close_tags = content.count('</tool_call>') + content.count('</tool_result>')
    if open_tags != close_tags:
        result.errors.append(f"Unbalanced tags: {open_tags} opening, {close_tags} closing")
    
    return result


def print_result(filename: str, result: ValidationResult) -> None:
    """Print validation results."""
    print(f"\n{'='*50}")
    print(f"[FILE] {filename}")
    print('='*50)
    
    print(f"\nStats: {result.stats.get('questions', 0)} Q/A pairs, "
          f"{result.stats.get('tool_calls', 0)} tool calls")
    
    if result.stats.get('tools'):
        print(f"   Tools: {', '.join(result.stats['tools'].keys())}")
    
    if result.errors:
        print(f"\n[ERROR] Errors ({len(result.errors)}):")
        for err in result.errors[:5]:
            print(f"   - {err}")
        if len(result.errors) > 5:
            print(f"   ... and {len(result.errors) - 5} more")
    
    if result.warnings:
        print(f"\n[WARN] Warnings ({len(result.warnings)}):")
        for warn in result.warnings[:5]:
            print(f"   - {warn}")
        if len(result.warnings) > 5:
            print(f"   ... and {len(result.warnings) - 5} more")
    
    print(f"\n{'[OK] Valid' if result.is_valid else '[FAIL] Has errors'}")


def main() -> int:
    """CLI entry point."""
    print("="*50)
    print("ForgeAI Training Data Validator")
    print("="*50)
    
    if len(sys.argv) > 1:
        # Validate specific file
        filepath = Path(sys.argv[1])
        result = validate_file(filepath)
        print_result(filepath.name, result)
        return 0 if result.is_valid else 1
    
    # Validate all training files
    try:
        from forge_ai.config import CONFIG
        data_dir = Path(CONFIG.get("data_dir", "data"))
    except ImportError:
        data_dir = Path("data")
    
    training_files = [
        "training.txt", "tool_training_data.txt", "combined_action_training.txt",
        "user_training.txt"
    ]
    
    results = {}
    for filename in training_files:
        filepath = data_dir / filename
        if filepath.exists():
            results[filename] = validate_file(filepath)
    
    if not results:
        print("\nNo training data files found!")
        return 1
    
    for filename, result in results.items():
        print_result(filename, result)
    
    total_errors = sum(len(r.errors) for r in results.values())
    print(f"\n{'='*50}")
    print(f"Files: {len(results)} | Errors: {total_errors}")
    print('='*50)
    
    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
