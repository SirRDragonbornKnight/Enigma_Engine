"""
Quality Gate — run before every commit.

Usage:
    python check.py          # lint + tests
    python check.py --fix    # auto-fix safe lint issues, then test
    python check.py --lint   # lint only (no tests)
    python check.py --test   # tests only (no lint)
"""
from __future__ import annotations

import subprocess
import sys
import time


def _run(cmd: list[str], label: str) -> bool:
    """Run a command and return True if it succeeded."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}\n")
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=".")
    elapsed = time.perf_counter() - start
    ok = result.returncode == 0
    status = "PASS" if ok else "FAIL"
    print(f"\n  [{status}] {label} ({elapsed:.1f}s)")
    return ok


def main():
    args = set(sys.argv[1:])
    do_fix = "--fix" in args
    lint_only = "--lint" in args
    test_only = "--test" in args

    results: list[tuple[str, bool]] = []

    # ── Lint ────────────────────────────────────────────────────
    if not test_only:
        lint_cmd = [
            sys.executable, "-m", "ruff", "check", "enigma_engine/",
        ]
        if do_fix:
            lint_cmd.append("--fix")

        ok = _run(lint_cmd, "Ruff Lint (bugs + security)")
        results.append(("Ruff Lint", ok))

    # ── Tests ───────────────────────────────────────────────────
    if not lint_only:
        ok = _run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            "Pytest (full suite)",
        )
        results.append(("Pytest", ok))

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    all_ok = True
    for name, ok in results:
        icon = "PASS" if ok else "FAIL"
        print(f"  [{icon}] {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All checks passed.\n")
    else:
        print("\n  Some checks failed — fix before committing.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
