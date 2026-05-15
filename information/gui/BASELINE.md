# GUI Baseline Measurements — Phase 0b

**Status:** Measurement protocol shipped. Numbers pending operator run. Author: AA code maker. Date: May 13, 2026.
**Pass anchor:** GUI-ARCH-0b.

This document is the **measurement contract** for the current CustomTkinter GUI. Every metric in [ARCH_DECISION.md](ARCH_DECISION.md) §5 (rubric) and §4 (gate G3) requires a baseline number on the incumbent stack so Phase 1 POCs are scored against reality, not against arbitrary absolutes.

> **Why this matters.** Per §4 Learned Principles (`AA code maker.md`): *"Beat baseline" targets require measuring the baseline first. A gate or metric written as "≤ baseline" is vacuous if the baseline is never captured.* This file fixes that gap before Phase 1 starts.

---

## 1. Environment

Fill in once. All Phase 1 POC runs must use the **same machine** — running PySide6 on a 4090 desktop and Tauri on a laptop is not a fair comparison.

| Field | Value |
|---|---|
| Machine | _e.g. RTX 5090 desktop, 32 GB RAM_ |
| OS | Windows ___ (build ___) |
| Python | ___ (`python --version`) |
| Installed extras | `pip list \| findstr /i "torch customtkinter pyside6"` |
| GUI commit | `git rev-parse HEAD` |
| Date measured | YYYY-MM-DD |

## 2. Measurement protocol

Run each measurement **3 times**, record min/median/max, use **median** in the table. Close all non-essential apps first. Disable Windows Game Bar / Discord overlay / any FPS overlay before measuring.

### M1 — Cold start (shell ready to accept input)

**What:** Time from `Launch Enigma.bat` (or `python run.py --gui`) double-click to the moment the main window is rendered AND a keyboard event reaches the focused widget.

**How:** Wrap the entry point with a stopwatch. Insert a one-off `print(f"[BASELINE] ready {time.perf_counter():.3f}")` at the end of `EnigmaGUI.__init__` (or wherever the mainloop is entered) and a `print(f"[BASELINE] start {time.perf_counter():.3f}")` at the very top of `run.py`. Subtract.

**Record:** median of 3 cold starts (kill the process between runs; do not measure warm starts).

### M2 — Page-switch latency

**What:** Time from sidebar button press on CONFIG → page fully rendered. Repeat for FORGE.

**How:** Bind `<Button-1>` on the nav button to record `t0 = time.perf_counter()` *before* dispatch; record `t1` at the end of the page build function (`_build_config_page` or similar). Print `t1 - t0`.

**Record:** median of 3 switches each. Switch from a different page (e.g. HOME → CONFIG, not CONFIG → CONFIG).

### M3 — Idle RAM, shell process only

**What:** Resident set size (RSS) of the GUI Python process after the window has been idle for 60 seconds, **before any model is loaded** (no checkpoint selected in MODELS page).

**How:** PowerShell, with the GUI window idle on the HOME page for 60 s:
```powershell
Get-Process python | Where-Object { $_.MainWindowTitle -like "*Enigma*" } | Select-Object Id, @{n="RSS_MB";e={[math]::Round($_.WorkingSet64/1MB,1)}}
```
**Record:** median of 3 cold launches.

**Why "before model loaded":** Per rubric (§5 row 3), torch + model weights are excluded from the shell-only number. Loading a 742M model adds ~3 GB and dominates the measurement, which would make every framework look identical.

### M4 — Packaged install size

**Filled by agent (May 13, 2026, GUI-ARCH-0b).** The current GUI ships as a Python source tree, not a packaged installer. The number below is the **UI-surface estimate** (GUI source + CustomTkinter + Pillow) excluding the Python interpreter (~30 MB) and torch (~2 GB) per rubric §5 row 3.

| Component | Size (MB) |
|---|---|
| `enigma_engine/gui/` source | 2.4 |
| CustomTkinter | 1.4 |
| Pillow | 15.1 |
| **Sum** | **19.0** |

Reproduce with:
```powershell
python information/gui/measure_baseline.py --m4
```

### M5 — Frame stall during 30-second training step (Gate G3)

**Automation note.** The M3 idle RSS step is automated by [measure_baseline.py](measure_baseline.py) (`--m3 --pid <PID> --settle 60`); operator still needs to launch the GUI in a separate process first. M1/M2/M5 remain operator-instrumented (require code splicing into live `EnigmaGUI.__init__` and the page builders).


**What:** While a synthetic 30-second training run is executing on a worker thread, log the longest interval between `after(16, ...)` callbacks. A 16 ms interval = 60 FPS. Anything above ~50 ms is visible jank. The number we record is the **median of the max-stall-per-run** over 3 runs.

**How:** Add a one-off frame-monitor:
```python
import time
_last_tick = [time.perf_counter()]
_max_stall = [0.0]
def _tick():
    now = time.perf_counter()
    dt = (now - _last_tick[0]) * 1000  # ms
    if dt > _max_stall[0]:
        _max_stall[0] = dt
    _last_tick[0] = now
    self.after(16, _tick)
self.after(16, _tick)
```
Trigger a real 30 s training step from FORGE (a smoke-data SFT pass with `epochs=1`, `--batch-size 1`, ~30 s wall-clock). Read `_max_stall[0]` after.

**Record:** median of 3 max-stall values. This is the number Phase 1 POCs must not exceed on the same workload (primary G3) or improve to < 100 ms (stretch G3).

### M6 — Dev-velocity proxy (for rubric row 5)

**Soft signal only.** Not measured here. Phase 1 records dev-time-to-parity per track during the POC.

## 3. Results table (fill in after measurement)

| Metric | Cold (1) | Cold (2) | Cold (3) | Median |
|---|---|---|---|---|
| M1 — Cold start (s) | | | | |
| M2a — Page-switch HOME → CONFIG (ms) | | | | |
| M2b — Page-switch HOME → FORGE (ms) | | | | |
| M3 — Idle RAM (MB) | | | | |
| M4 — Estimated bundle size (MB) | 19.0 (one-shot) | — | — | **19.0** |
| M5 — Max frame stall during 30 s training (ms) | | | | |

## 4. Notes captured during measurement

Operator: free-form notes about anything unusual — disk I/O during cold start, GPU driver pop-ups, debounced widgets behaving oddly, anything that would invalidate a number. Document everything; future Phase 1 / Phase 2 will refer back here.

```
(empty until first run)
```

## 5. Acceptance for Phase 0b close

- [x] M4 estimate captured (agent, May 13, 2026 — 19.0 MB).
- [x] Helper script [measure_baseline.py](measure_baseline.py) automates M3 + M4.
- [ ] §1 Environment filled in (operator).
- [ ] §3 Results table has medians for M1, M2a, M2b, M3, M5 (operator).
- [ ] §4 Notes captured (or explicit "no anomalies observed") (operator).
- [ ] Numbers cross-referenced in [ARCH_DECISION.md](ARCH_DECISION.md) §9 sign-off checklist as "filled in" (operator).

Once §5 is complete, Phase 0b is closed and the Phase 1 rubric has real targets to score against.
