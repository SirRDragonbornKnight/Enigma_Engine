"""Tests for the GUI baseline measurement helper.

GUI-ARCH-0b instrumentation: M1 cold-start, M2 page-switch, M5
frame-stall.  Pure helper class — testable without launching a real
tk mainloop.
"""

from __future__ import annotations

import time

import pytest

from enigma_engine.gui.baseline_instrument import BaselineMonitor


class TestBaselineMonitorM1:
    def test_emit_m1_prints_elapsed_seconds(self, capsys: pytest.CaptureFixture[str]) -> None:
        # process_start in the past should produce a positive elapsed.
        ps = time.perf_counter() - 0.5
        mon = BaselineMonitor(process_start=ps)
        mon.emit_m1()
        out = capsys.readouterr().out
        assert "[BASELINE] M1_cold_start_s=" in out
        # value should be >= 0.5s (the gap we forced)
        line = next(ln for ln in out.splitlines() if "M1_cold_start_s" in ln)
        value = float(line.split("=", 1)[1])
        assert value >= 0.4  # generous lower bound for slow CI

    def test_emit_m1_is_idempotent(self, capsys: pytest.CaptureFixture[str]) -> None:
        mon = BaselineMonitor(process_start=time.perf_counter())
        mon.emit_m1()
        mon.emit_m1()
        mon.emit_m1()
        out = capsys.readouterr().out
        # Only the FIRST call should print.  Second/third are no-ops
        # — important because the after(0, ...) callback can race
        # with operator-triggered re-emits in dev.
        assert out.count("M1_cold_start_s") == 1


class TestBaselineMonitorM2:
    def test_time_page_switch_prints_from_to_and_ms(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mon = BaselineMonitor(process_start=time.perf_counter())
        start = time.perf_counter() - 0.02  # 20ms ago
        mon.time_page_switch("CORE", "CONFIG", start)
        out = capsys.readouterr().out
        assert "[BASELINE] M2_switch" in out
        assert "from=CORE" in out
        assert "to=CONFIG" in out
        # parse the ms value
        line = next(ln for ln in out.splitlines() if "M2_switch" in ln)
        ms_token = next(t for t in line.split() if t.startswith("ms="))
        ms = float(ms_token.split("=", 1)[1])
        assert ms >= 15.0  # generous lower bound around the 20ms gap


class TestBaselineMonitorM5:
    def test_frame_tick_tracks_max_stall(self) -> None:
        mon = BaselineMonitor(process_start=time.perf_counter())
        # First tick establishes baseline; max_stall_ms is whatever
        # the gap since __init__ was — could be ~0 in fast CI.
        mon.frame_tick()
        first = mon.max_stall_ms
        # Force a known gap
        time.sleep(0.05)
        mon.frame_tick()
        second = mon.max_stall_ms
        assert second >= first
        assert second >= 40.0  # 50ms sleep, allow 20% jitter

    def test_frame_tick_returns_current_max(self) -> None:
        mon = BaselineMonitor(process_start=time.perf_counter())
        mon.frame_tick()
        time.sleep(0.03)
        returned = mon.frame_tick()
        assert returned == mon.max_stall_ms

    def test_smaller_subsequent_gap_does_not_lower_max(self) -> None:
        mon = BaselineMonitor(process_start=time.perf_counter())
        mon.frame_tick()
        time.sleep(0.05)
        mon.frame_tick()
        peak = mon.max_stall_ms
        # Now several fast ticks; max must stay at peak
        for _ in range(5):
            mon.frame_tick()
        assert mon.max_stall_ms == peak
