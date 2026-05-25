"""Wire-site tests for the GUI-ARCH-0b ``--baseline`` flag.

Confirms the flag propagates from CLI argparse → ``run_gui_app`` →
``run_gui`` → ``EnigmaGUI.__init__`` without launching a real tk
mainloop.  Structural because constructing an ``EnigmaGUI`` requires
CustomTkinter + a display; the wire-site assertions are paired with
behavioural tests on the pure helper in
``tests/test_baseline_instrument.py``.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import enigma_engine.gui.desktop as desktop_mod

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestRunPyBaselineFlag:
    def test_argparse_registers_baseline_flag(self) -> None:
        src = _read(REPO_ROOT / "run.py")
        # Must register a literal --baseline action="store_true"
        # under the same argparse parser used by --gui.
        assert re.search(
            r'add_argument\(\s*"--baseline"\s*,\s*action\s*=\s*"store_true"',
            src,
        ), "run.py must register --baseline as a store_true argparse flag"

    def test_dispatch_forwards_baseline_to_run_gui_app(self) -> None:
        src = _read(REPO_ROOT / "run.py")
        # The --gui dispatch branch must forward args.baseline.
        assert re.search(
            r"run_gui_app\(\s*args\.model\s*,\s*baseline\s*=\s*args\.baseline\s*\)",
            src,
        ), "run.py --gui dispatch must forward args.baseline to run_gui_app"

    def test_process_start_captured_at_module_top(self) -> None:
        src = _read(REPO_ROOT / "run.py")
        # _PROCESS_START must exist and be the earliest reasonable
        # anchor for M1.  It is referenced by run_gui_app when
        # baseline=True.
        assert "_PROCESS_START = time.perf_counter()" in src
        assert "process_start=_PROCESS_START" in src


class TestRunGuiSignature:
    def test_run_gui_accepts_baseline_and_process_start(self) -> None:
        sig = inspect.signature(desktop_mod.run_gui)
        assert "baseline" in sig.parameters
        assert "process_start" in sig.parameters
        # Both must default to a falsy value so existing callers
        # (auto-spawn, tests) keep working unchanged.
        assert sig.parameters["baseline"].default is False
        assert sig.parameters["process_start"].default is None


class TestEnigmaGUIWireSites:
    def test_init_signature_accepts_baseline_kwargs(self) -> None:
        sig = inspect.signature(desktop_mod.EnigmaGUI.__init__)
        assert "baseline" in sig.parameters
        assert "process_start" in sig.parameters

    def test_init_imports_baseline_monitor_under_flag(self) -> None:
        src = inspect.getsource(desktop_mod.EnigmaGUI.__init__)
        # Import is gated inside the ``if baseline:`` branch — proves
        # zero overhead when the flag is off (no module import cost).
        assert "from enigma_engine.gui.baseline_instrument import" in src
        assert "BaselineMonitor" in src
        assert "self._baseline_monitor" in src

    def test_init_schedules_m1_emit_via_after(self) -> None:
        src = inspect.getsource(desktop_mod.EnigmaGUI.__init__)
        # M1 must be scheduled via after(0, ...) so it fires only
        # once the tk event loop reaches idle — measures the full
        # cold-start window, not the mid-__init__ snapshot.
        assert re.search(
            r"self\.after\(\s*0\s*,\s*self\._baseline_monitor\.emit_m1\s*\)",
            src,
        ), "M1 must be scheduled with after(0, monitor.emit_m1)"

    def test_init_schedules_frame_tick(self) -> None:
        src = inspect.getsource(desktop_mod.EnigmaGUI.__init__)
        assert re.search(
            r"self\.after\(\s*16\s*,\s*self\._baseline_frame_tick\s*\)",
            src,
        ), "M5 frame tick must be scheduled with after(16, ...)"

    def test_switch_page_times_transition_when_baseline_on(self) -> None:
        src = inspect.getsource(desktop_mod.EnigmaGUI._switch_page)
        # Capture-before-work and call to time_page_switch must both
        # be present, gated on the monitor attribute.
        assert "time.perf_counter()" in src
        assert "self._baseline_monitor.time_page_switch(" in src

    def test_frame_tick_method_exists_and_calls_monitor(self) -> None:
        assert hasattr(desktop_mod.EnigmaGUI, "_baseline_frame_tick")
        src = inspect.getsource(desktop_mod.EnigmaGUI._baseline_frame_tick)
        assert "frame_tick()" in src
        assert "self.after(" in src  # self-rescheduling
