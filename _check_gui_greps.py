"""Quick script to classify remaining source-greps in test_gui.py."""
import ast
import re

with open("tests/test_gui.py", "r", encoding="utf-8") as f:
    source = f.read()

tree = ast.parse(source)
lines = source.splitlines()

anti = []
positive = []

for node in ast.walk(tree):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
    if not node.name.startswith("test_"):
        continue
    body_src = "\n".join(lines[node.lineno - 1 : node.end_lineno])
    if "getsource" not in body_src and "read_text" not in body_src:
        continue
    is_anti = bool(
        re.search(r"assert .+ not in ", body_src)
        or re.search(r"assert not ", body_src)
    )
    if is_anti:
        anti.append(node.name)
    else:
        positive.append(node.name)

print(f"Total source-grep tests in test_gui.py: {len(anti) + len(positive)}")
print(f"Anti-pattern guards (keep): {len(anti)}")
print(f"Positive source-greps (remove): {len(positive)}")
print()

if positive:
    print("=== POSITIVE SOURCE-GREPS (should have been removed) ===")
    for name in positive:
        print(f"  {name}")
    print()

if anti:
    print("=== ANTI-PATTERN GUARDS (keep) - first 10 ===")
    for name in anti[:10]:
        print(f"  {name}")
