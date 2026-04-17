"""
Script to automatically remove EXISTS-CHECK and STRING-IN-SOURCE tests
from test files, while preserving REAL tests and anti-pattern checks.

Uses AST to find bad test functions and removes them from source. 
"""
import ast
import re
import sys
from pathlib import Path


def classify_test(node: ast.FunctionDef, source_lines: list[str]) -> str:
    """Classify a test function as REAL, STRING-IN-SOURCE, or EXISTS-CHECK."""
    # Get the source text for this function
    start = node.lineno - 1
    end = node.end_lineno
    func_source = "\n".join(source_lines[start:end])
    func_name = node.name.lower()
    
    # --- SOURCE-GREP DETECTION (check FIRST) ---
    has_getsource = "getsource" in func_source
    has_read_text = "read_text" in func_source and "assert" in func_source
    has_open_read = ('open(' in func_source and '.read()' in func_source 
                     and 'assert' in func_source)
    is_source_reader = has_getsource or has_read_text or has_open_read
    
    if is_source_reader:
        # Check if it also does real behavioral testing
        has_real_call = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_text = ast.get_source_segment(
                    "\n".join(source_lines), child) or ""
                if any(skip in call_text for skip in 
                       ['getsource', 'read_text', 'open(', 'split(', 
                        'count(', 'strip(', 'startswith(', 'lower(',
                        're.search', 're.compile', 're.findall', 're.match',
                        're.DOTALL', 're.MULTILINE', 'json.loads',
                        'ast.parse', 'ast.walk', 'len(', 'any(', 'all(',
                        'enumerate(', 'zip(', 'list(', 'set(', 'dict(',
                        'sorted(', 'max(', 'min(', 'sum(', 'range(',
                        'isinstance(', 'hasattr(', 'callable(',
                        'Path(', 'str(', 'int(', 'float(',
                        '.group(', '.groups(', '.match(', '.findall(',
                        '.items(', '.keys(', '.values(', '.get(',
                        '.replace(', '.join(', '.encode(', '.decode(',
                        '.endswith(', '.name', '.parent',
                        'inspect.signature', 'inspect.getmembers',
                        'print(', 'format(',
                        '.find(', '.index(', '.rfind(', '.resolve(',
                        '.splitlines(', '__import__(', 'getattr(',
                        'next(', '.upper(', '.lstrip(', '.rstrip(',
                        'type(', 'bool(', 'tuple(', 'map(', 'filter(',
                        'ord(', 'abs(', '.exists(', '.is_file(',
                        'importlib']):
                    continue
                has_real_call = True
                break
        
        if has_real_call:
            return "REAL TEST"
        
        # It's a source-grep test. Now check if it's a PURE anti-pattern guard.
        # Anti-pattern guards ONLY assert that bad things DON'T exist.
        # If the test has ANY positive assertions ("X" in source), it's a fake.
        assert_lines = [l.strip() for l in func_source.split("\n") 
                        if l.strip().startswith("assert")]
        
        has_positive_assert = False
        has_negative_assert = False
        for aline in assert_lines:
            # Positive: assert "X" in source (without "not")
            if re.search(r'assert\s+["\'].*["\']\s+in\s+\w+', aline) and "not in" not in aline:
                has_positive_assert = True
            # Negative: assert "X" not in Y, or assert not Y
            if "not in" in aline or aline.startswith("assert not "):
                has_negative_assert = True
        
        # Pure anti-pattern: only negative assertions, or name indicates guard
        anti_pattern_name_kws = [
            "no_dead", "no_unused", "no_bare", "no_stale", "not_called",
            "removed", "no_hardcoded", "not_in_", "no_direct",
            "not_silence", "not_redirected", "no_print",
            "no_asyncio", "no_eager", "no_simpledialog",
            "no_module_level", "no_cors", "no_immediate_yield",
            "no_top_level", "does_not_",
        ]
        is_anti_pattern_name = any(kw in func_name for kw in anti_pattern_name_kws)
        
        if not has_positive_assert and (has_negative_assert or is_anti_pattern_name):
            # Pure anti-pattern guard — keep it
            return "REAL TEST"
        
        # Has positive assertions → it's a source-grep regardless of negatives
        return "STRING-IN-SOURCE"
    
    # EXISTS-CHECK: assert X is not None / hasattr / callable / isinstance
    # without any real behavior testing
    body_stmts = node.body
    # Skip docstring
    real_stmts = [s for s in body_stmts 
                  if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    
    # Signature-only check: inspect.signature() + assert param in parameters
    if "signature(" in func_source and "parameters" in func_source:
        # Only a signature check if no real calls exist
        sig_only = True
        for stmt in real_stmts:
            stmt_src = "\n".join(source_lines[stmt.lineno - 1:
                                              (stmt.end_lineno or stmt.lineno)])
            if any(kw in stmt_src for kw in ['monkeypatch', 'tmp_path', 'torch.',
                                              'model(', '.execute(', '.query(']):
                sig_only = False
                break
        if sig_only:
            return "EXISTS-CHECK"

    # hasattr loops: for attr in (...): assert hasattr(Class, attr)
    if "hasattr(" in func_source:
        assert_stmts = [s for s in real_stmts if isinstance(s, ast.Assert)]
        non_import = [s for s in real_stmts 
                      if not isinstance(s, (ast.Import, ast.ImportFrom, ast.Assign))]
        if assert_stmts:
            all_hasattr = all("hasattr(" in 
                            "\n".join(source_lines[s.lineno - 1:(s.end_lineno or s.lineno)])
                            for s in assert_stmts)
            if all_hasattr:
                return "EXISTS-CHECK"
    
    # For loops that only check hasattr
    for stmt in real_stmts:
        if isinstance(stmt, ast.For):
            for_src = "\n".join(source_lines[stmt.lineno - 1:
                                             (stmt.end_lineno or stmt.lineno)])
            if "hasattr(" in for_src and "assert" in for_src:
                # Check if this is the only substantial thing
                other_stmts = [s for s in real_stmts 
                              if s is not stmt and not isinstance(s, (ast.Import, ast.ImportFrom, ast.Assign))]
                if not other_stmts:
                    return "EXISTS-CHECK"

    all_trivial = True
    for stmt in real_stmts:
        stmt_source = "\n".join(source_lines[stmt.lineno - 1:
                                              (stmt.end_lineno or stmt.lineno)])
        # Import statements are OK
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        # Simple assignments (from imports)
        if isinstance(stmt, ast.Assign):
            # Check if assignment has a real behavioral call (not just import/constructor)
            assign_src = stmt_source.strip()
            has_behavioral_call = any(
                kw in assign_src for kw in [
                    '.encode(', '.decode(', '.execute(', '.query(',
                    '.train(', '.generate(', '.save(', '.load(',
                    '.add(', '.fit(', '.predict(', '.transform(',
                ])
            if has_behavioral_call:
                all_trivial = False
                break
            continue
        # Assert statements
        if isinstance(stmt, ast.Assert):
            test_expr = ast.get_source_segment(
                "\n".join(source_lines), stmt.test) or ""
            # Trivial patterns
            trivial_patterns = [
                r'^\s*\w+\s+is\s+not\s+None$',
                r'^\s*hasattr\(',
                r'^\s*callable\(',
                r'^\s*isinstance\(\w+,\s*\w+\)$',
            ]
            is_trivial = any(re.match(p, test_expr.strip()) 
                           for p in trivial_patterns)
            if not is_trivial:
                all_trivial = False
                break
        elif isinstance(stmt, ast.Expr):
            # Function calls that aren't just imports
            all_trivial = False
            break
        else:
            all_trivial = False
            break
    
    if all_trivial and len(real_stmts) > 0:
        return "EXISTS-CHECK"
    
    return "REAL TEST"


def find_bad_tests(filepath: Path) -> list[tuple[str, str, int, int, str]]:
    """Find bad tests in a file. Returns list of (class, method, start, end, category)."""
    source = filepath.read_text(encoding='utf-8')
    source_lines = source.split('\n')
    tree = ast.parse(source)
    
    bad_tests = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                    category = classify_test(item, source_lines)
                    if category != "REAL TEST":
                        bad_tests.append((
                            class_name, item.name, 
                            item.lineno, item.end_lineno,
                            category
                        ))
        elif isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            # Top-level test function
            if not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree)):
                category = classify_test(node, source_lines)
                if category != "REAL TEST":
                    bad_tests.append((
                        "<module>", node.name,
                        node.lineno, node.end_lineno,
                        category
                    ))
    
    return bad_tests


def remove_bad_tests(filepath: Path, bad_tests: list, dry_run: bool = False) -> int:
    """Remove bad tests from a file. Returns count of removed tests."""
    source = filepath.read_text(encoding='utf-8')
    lines = source.split('\n')
    
    # Build set of line ranges to remove
    lines_to_remove = set()
    for class_name, method_name, start, end, category in bad_tests:
        # Include blank lines before the method (up to 2)
        actual_start = start
        while actual_start > 1 and lines[actual_start - 2].strip() == '':
            actual_start -= 1
            if actual_start <= start - 2:
                break
        
        for i in range(actual_start, end + 1):
            lines_to_remove.add(i)
    
    # Check if removing tests leaves an empty class
    tree = ast.parse(source)
    classes_to_remove = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Count remaining test methods after removal
            remaining = 0
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                    if item.lineno not in lines_to_remove:
                        remaining += 1
            if remaining == 0:
                # Remove entire class including decorators
                class_start = node.lineno
                # Check for decorators
                if node.decorator_list:
                    class_start = min(d.lineno for d in node.decorator_list)
                # Extend backwards for blank lines
                actual_start = class_start
                while actual_start > 1 and lines[actual_start - 2].strip() == '':
                    actual_start -= 1
                    if actual_start <= class_start - 3:
                        break
                classes_to_remove.add(node.name)
                for i in range(actual_start, node.end_lineno + 1):
                    lines_to_remove.add(i)
    
    if dry_run:
        return len(bad_tests) + sum(1 for _ in classes_to_remove)
    
    # Build new source, skipping removed lines
    new_lines = []
    for i, line in enumerate(lines, 1):
        if i not in lines_to_remove:
            new_lines.append(line)
    
    # Clean up excessive blank lines (3+ in a row → 2)
    cleaned = []
    blank_count = 0
    for line in new_lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)
    
    filepath.write_text('\n'.join(cleaned), encoding='utf-8')
    return len(bad_tests)


def main():
    test_dir = Path(__file__).parent / "tests"
    target_files = sys.argv[1:] if len(sys.argv) > 1 else [
        "test_new_features.py",
        "test_research_upgrades.py", 
        "test_training.py",
        "test_gui.py",
        "test_core.py",
    ]
    
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        target_files = [f for f in target_files if f != "--dry-run"]
    
    total_removed = 0
    for fname in target_files:
        filepath = test_dir / fname
        if not filepath.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        
        bad_tests = find_bad_tests(filepath)
        if not bad_tests:
            print(f"  {fname}: 0 bad tests")
            continue
        
        if dry_run:
            print(f"\n  {fname}: {len(bad_tests)} bad tests to remove:")
            for cls, method, start, end, cat in bad_tests:
                print(f"    {cat:20s} {cls}.{method} (L{start}-{end})")
        else:
            count = remove_bad_tests(filepath, bad_tests)
            total_removed += count
            print(f"  {fname}: removed {count} bad tests")
    
    if not dry_run:
        print(f"\n  TOTAL REMOVED: {total_removed}")
    

if __name__ == "__main__":
    main()
