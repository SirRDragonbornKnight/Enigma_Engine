"""Bridge: run the avatar's JS unit tests (node --test) inside the pytest suite.

The engine's pure logic lives in JavaScript (fit math, name matching, the model
library), so the real assertions are in ``mods/avatar/tests/*.test.js``. This wrapper
runs them via the portable Node interpreter so ``pytest`` covers them too, alongside
``tests/test_avatar_bone_data.py``. Skips cleanly if Node isn't available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

AVATAR_DIR = Path(__file__).resolve().parents[1] / "mods" / "avatar"


def _find_node() -> str | None:
    portable = Path(os.environ.get("LOCALAPPDATA", "")) / "node-portable" / "node.exe"
    if portable.exists():
        return str(portable)
    return shutil.which("node")


@pytest.mark.integration
def test_avatar_js_rig_tests_pass() -> None:
    node = _find_node()
    if not node:
        pytest.skip("node not found (portable Node or PATH) — JS rig tests skipped")
    if not (AVATAR_DIR / "node_modules" / "three").exists():
        pytest.skip("avatar node_modules/three not installed — run npm install in mods/avatar")
    proc = subprocess.run(
        [node, "--test"],  # auto-discovers tests/*.test.js (skips node_modules)
        cwd=str(AVATAR_DIR),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"avatar JS rig tests failed:\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
