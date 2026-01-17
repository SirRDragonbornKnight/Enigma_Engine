#!/usr/bin/env python3
"""
Training Data Validator - Check training data for common issues.

Validates:
1. Correct Q:/A: format
2. Tool call JSON format
3. Balanced tags
4. Minimum data size
5. Common mistakes

Usage:
    python scripts/validate_training_data.py
    python scripts/validate_training_data.py path/to/data.txt
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ValidationResult:
    """Result of validation."""
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.stats: Dict[str, int] = {}
    
    def add_error(self, msg: str, line: int = None):
        if line:
            self.errors.append(f"Line {line}: {msg}")
        else:
            self.errors.append(msg)
    
    def add_warning(self, msg: str, line: int = None):
        if line:
            self.warnings.append(f"Line {line}: {msg}")
        else:
            self.warnings.append(msg)
    
    def add_info(self, msg: str):
        self.info.append(msg)
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_file(filepath: Path) -> ValidationResult:
    """Validate a training data file."""
    result = ValidationResult()
    
    if not filepath.exists():
        result.add_error(f"File not found: {filepath}")
        return result
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        result.add_error(f"Could not read file: {e}")
        return result
    
    # Basic stats
    result.stats['total_lines'] = len(lines)
    result.stats['non_empty_lines'] = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
    
    # Count Q/A pairs
    q_count = sum(1 for l in lines if l.strip().startswith('Q:'))
    a_count = sum(1 for l in lines if l.strip().startswith('A:'))
    result.stats['questions'] = q_count
    result.stats['answers'] = a_count
    
    if q_count != a_count:
        result.add_warning(f"Mismatched Q/A pairs: {q_count} questions, {a_count} answers")
    
    # Minimum data check
    if q_count < 100:
        result.add_warning(f"Very small dataset ({q_count} Q/A pairs). Recommend 1000+ for good results.")
    elif q_count < 500:
        result.add_info(f"Small dataset ({q_count} Q/A pairs). More data will improve results.")
    else:
        result.add_info(f"Good dataset size: {q_count} Q/A pairs")
    
    # Tool call validation
    tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    tool_result_pattern = re.compile(r'<tool_result>(.*?)</tool_result>', re.DOTALL)
    
    tool_calls = tool_call_pattern.findall(content)
    tool_results = tool_result_pattern.findall(content)
    
    result.stats['tool_calls'] = len(tool_calls)
    result.stats['tool_results'] = len(tool_results)
    
    if len(tool_calls) != len(tool_results):
        result.add_warning(f"Mismatched tool_call/tool_result: {len(tool_calls)} calls, {len(tool_results)} results")
    
    # Validate tool call JSON
    tools_used = Counter()
    for i, tc in enumerate(tool_calls):
        tc = tc.strip()
        try:
            data = json.loads(tc)
            if 'tool' not in data:
                result.add_warning(f"Tool call #{i+1} missing 'tool' field")
            else:
                tools_used[data['tool']] += 1
            if 'params' not in data:
                result.add_warning(f"Tool call #{i+1} missing 'params' field")
        except json.JSONDecodeError as e:
            result.add_error(f"Tool call #{i+1} has invalid JSON: {e}")
    
    result.stats['tools_used'] = dict(tools_used)
    
    # Validate tool result JSON
    for i, tr in enumerate(tool_results):
        tr = tr.strip()
        try:
            data = json.loads(tr)
            if 'success' not in data:
                result.add_warning(f"Tool result #{i+1} missing 'success' field")
        except json.JSONDecodeError as e:
            result.add_error(f"Tool result #{i+1} has invalid JSON: {e}")
    
    # Check for wrong formats
    wrong_formats = [
        (r'User:', "Found 'User:' - should use 'Q:' instead"),
        (r'Human:', "Found 'Human:' - should use 'Q:' instead"),
        (r'Assistant:', "Found 'Assistant:' - should use 'A:' instead"),
        (r'AI:', "Found 'AI:' - should use 'A:' instead (unless in middle of text)"),
    ]
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        for pattern, msg in wrong_formats:
            if stripped.startswith(pattern.rstrip(':')):
                result.add_warning(msg, line_num)
    
    # Check for unbalanced tags
    open_tags = content.count('<tool_call>') + content.count('<tool_result>')
    close_tags = content.count('</tool_call>') + content.count('</tool_result>')
    if open_tags != close_tags:
        result.add_error(f"Unbalanced tool tags: {open_tags} opening, {close_tags} closing")
    
    # Check for very long lines (might cause issues)
    for line_num, line in enumerate(lines, 1):
        if len(line) > 2000:
            result.add_warning(f"Very long line ({len(line)} chars) - may cause issues", line_num)
    
    # Summary info
    if tools_used:
        result.add_info(f"Tools trained: {', '.join(tools_used.keys())}")
    
    return result


def validate_all_training_data() -> Dict[str, ValidationResult]:
    """Validate all training data files in the data directory."""
    from forge_ai.config import CONFIG
    
    data_dir = Path(CONFIG.get("data_dir", "data"))
    results = {}
    
    training_files = [
        "training.txt",
        "tool_training_data.txt",
        "personality_development.txt",
        "self_awareness_training.txt",
        "combined_action_training.txt",
        "sacrifice_training.txt",
        "user_training.txt",
    ]
    
    for filename in training_files:
        filepath = data_dir / filename
        if filepath.exists():
            results[filename] = validate_file(filepath)
    
    return results


def print_result(filename: str, result: ValidationResult):
    """Print validation result nicely."""
    print(f"\n{'='*60}")
    print(f"📄 {filename}")
    print('='*60)
    
    # Stats
    print(f"\n📊 Statistics:")
    print(f"   Lines: {result.stats.get('total_lines', 0)} ({result.stats.get('non_empty_lines', 0)} non-empty)")
    print(f"   Q/A pairs: {result.stats.get('questions', 0)}")
    print(f"   Tool calls: {result.stats.get('tool_calls', 0)}")
    
    if result.stats.get('tools_used'):
        print(f"   Tools: {', '.join(result.stats['tools_used'].keys())}")
    
    # Errors
    if result.errors:
        print(f"\n❌ Errors ({len(result.errors)}):")
        for err in result.errors[:10]:  # Limit to 10
            print(f"   • {err}")
        if len(result.errors) > 10:
            print(f"   ... and {len(result.errors) - 10} more")
    
    # Warnings
    if result.warnings:
        print(f"\n⚠️ Warnings ({len(result.warnings)}):")
        for warn in result.warnings[:10]:
            print(f"   • {warn}")
        if len(result.warnings) > 10:
            print(f"   ... and {len(result.warnings) - 10} more")
    
    # Info
    if result.info:
        print(f"\nℹ️ Info:")
        for info in result.info:
            print(f"   • {info}")
    
    # Summary
    if result.is_valid:
        print(f"\n✅ Valid (no errors)")
    else:
        print(f"\n❌ Has errors - fix before training!")


def main():
    print("=" * 60)
    print("ForgeAI Training Data Validator")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Validate specific file
        filepath = Path(sys.argv[1])
        result = validate_file(filepath)
        print_result(filepath.name, result)
    else:
        # Validate all training files
        results = validate_all_training_data()
        
        if not results:
            print("\nNo training data files found!")
            return 1
        
        for filename, result in results.items():
            print_result(filename, result)
        
        # Overall summary
        total_errors = sum(len(r.errors) for r in results.values())
        total_warnings = sum(len(r.warnings) for r in results.values())
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"Files checked: {len(results)}")
        print(f"Total errors: {total_errors}")
        print(f"Total warnings: {total_warnings}")
        
        if total_errors > 0:
            print("\n❌ Fix errors before training!")
            return 1
        else:
            print("\n✅ All files valid!")
            return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
