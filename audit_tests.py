"""Audit test quality: categorize tests by what they actually verify."""
import ast
import re
from pathlib import Path
from collections import Counter

TESTS_DIR = Path("tests")
categories = Counter()
all_tests = {"source_grep": [], "exists_check": [], "real_test": []}

REAL_KEYWORDS = [
    "torch.", "model(", "model.generate", "reg.execute", "mem.add",
    "result =", "index.", "vec.fit", "tok.encode", "tok.decode",
    "chunks =", "profile =", ".train(", "encoder(", "parse_",
    "tmp_path", "monkeypatch", ".query(", ".build(", ".save(",
    ".load(", "== 0", "== 1", "== 2", "== 3", "shape ==",
    ".item()", "counts[", "torch.randn", "torch.zeros", "torch.ones",
    ".backward()", "CommandResult", "json.loads", "json.dumps",
    "expand_model_weights", ".execute(", "handle_message",
    "write_text", "CommandRegistry()", "PersistentMemory(",
    "TrainingConfig(", "ForgeConfig(", "Trainer._dpo_loss",
    ".encode(", ".decode(", "preprocess_image", "encode_image",
    "BPETokenizer", "SimpleTokenizer", "AdvancedBPETokenizer",
]


def _is_source_grep(body_source: str) -> bool:
    """Detect source-grep patterns including variable-indirection."""
    has_getsource = "getsource" in body_source
    has_read_text = "read_text" in body_source
    has_source_provider = has_getsource or has_read_text

    # Direct pattern: assert "x" in source / assert "x" not in source
    has_string_in_source = bool(re.search(
        r'assert\s+["\'].*["\']\s+(not\s+)?in\s+\w*source\w*', body_source
    ))
    # Variable pattern: source = ...; later assert "x" in source
    assigns_source_var = bool(re.search(
        r'\b\w*source\w*\s*=\s*.*(getsource|read_text)', body_source
    ))
    # "in src" variant
    has_string_in_src = bool(re.search(
        r'assert\s+["\'].*["\']\s+(not\s+)?in\s+\w*src\w*', body_source
    ))
    assigns_src_var = bool(re.search(
        r'\b\w*src\w*\s*=\s*.*(getsource|read_text)', body_source
    ))

    if has_source_provider and has_string_in_source:
        return True
    if assigns_source_var and has_string_in_source:
        return True
    if assigns_src_var and has_string_in_src:
        return True
    if has_source_provider and has_string_in_src:
        return True
    # Catch: read file then string-check on any variable storing the text
    if has_read_text and bool(re.search(
        r'assert\s+["\'].*["\']\s+(not\s+)?in\s+\w+', body_source
    )):
        return True
    if has_getsource and bool(re.search(
        r'assert\s+["\'].*["\']\s+(not\s+)?in\s+\w+', body_source
    )):
        return True
    return False


def _is_exists_check(body_source: str, body_lines: list[str]) -> bool:
    """Detect exists-check patterns including hasattr loops and signature-only."""
    # Classic patterns
    classic = (
        "is not None" in body_source
        or "callable(" in body_source
    )
    # hasattr in any form (including loops)
    has_hasattr = "hasattr(" in body_source
    # Signature-only: inspect.signature() but no actual call to the function
    sig_only = ("inspect.signature" in body_source or "getsource" not in body_source) and \
               "signature(" in body_source and \
               bool(re.search(r'assert\s+["\'].*["\']\s+in\s+.*parameters', body_source))

    # All assertions are hasattr
    assert_lines = [l for l in body_lines if l.startswith("assert ")]
    all_hasattr = assert_lines and all("hasattr(" in l for l in assert_lines)
    if all_hasattr and has_hasattr:
        return True

    # Short test with classic pattern
    if len(body_lines) <= 6 and (classic or has_hasattr):
        return True

    # Signature-only
    if sig_only and not any(kw in body_source for kw in REAL_KEYWORDS):
        return True

    return False

for pyfile in sorted(TESTS_DIR.glob("test_*.py")):
    source = pyfile.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        continue

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue

        body_source = ast.get_source_segment(source, node)
        if body_source is None:
            continue

        has_getsource = "getsource" in body_source
        has_read_text = "read_text" in body_source
        is_source_grep = _is_source_grep(body_source)
        has_real_logic = any(kw in body_source for kw in REAL_KEYWORDS)

        # Get body lines (skip docstring, comments, blank)
        lines = body_source.split("\n")
        body_lines = [
            l.strip() for l in lines[1:]
            if l.strip() and not l.strip().startswith("#") and not l.strip().startswith('"""')
        ]

        if is_source_grep and not has_real_logic:
            cat = "source_grep"
        elif has_real_logic:
            cat = "real_test"
        elif _is_exists_check(body_source, body_lines):
            cat = "exists_check"
        elif has_getsource or has_read_text:
            cat = "source_grep"
        else:
            cat = "real_test"

        categories[cat] += 1
        all_tests[cat].append(f"{pyfile.name}::{node.name}")


print("=== TEST QUALITY AUDIT ===\n")
total = sum(categories.values())
labels = {
    "source_grep": "STRING-IN-SOURCE  (inspect.getsource + assert 'x' in source)",
    "exists_check": "EXISTS-CHECK      (assert X is not None / hasattr / callable)",
    "real_test": "REAL TEST         (exercises actual logic with inputs/outputs)",
}
for cat in ["real_test", "source_grep", "exists_check"]:
    count = categories[cat]
    pct = count / total * 100 if total else 0
    print(f"  {count:4d} ({pct:4.1f}%)  {labels[cat]}")

print(f"\n  TOTAL: {total} test functions across {len(list(TESTS_DIR.glob('test_*.py')))} files\n")

# Show some source_grep examples
print("--- SAMPLE source-grep tests (string-matching, not real verification): ---")
for t in all_tests["source_grep"][:15]:
    print(f"  {t}")

print(f"\n--- SAMPLE exists-check tests (trivial import/exists): ---")
for t in all_tests["exists_check"][:15]:
    print(f"  {t}")

# Per-file breakdown
print("\n\n=== PER-FILE BREAKDOWN ===\n")
file_cats = {}
for cat, tests in all_tests.items():
    for t in tests:
        fname = t.split("::")[0]
        if fname not in file_cats:
            file_cats[fname] = Counter()
        file_cats[fname][cat] += 1

print(f"{'File':<35} {'Real':>6} {'SrcGrp':>7} {'Exists':>7} {'Total':>6} {'Bad%':>5}")
print("-" * 75)
for fname in sorted(file_cats.keys()):
    c = file_cats[fname]
    t = sum(c.values())
    bad = c["source_grep"] + c["exists_check"]
    pct = bad / t * 100 if t else 0
    print(f"{fname:<35} {c['real_test']:>6} {c['source_grep']:>7} {c['exists_check']:>7} {t:>6} {pct:>4.0f}%")
